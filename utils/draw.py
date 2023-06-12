import sys

sys.path.insert(0, '')

import os
import torch
import cv2
import json
import argparse
import random
import numpy as np
from PIL import Image
from utils.box import patch_to_box
from utils.data import get_dataset
from torchvision import transforms
from matplotlib import pyplot as plt
from model.vit_model import vit_base_patch16_224 as create_model


def gt_patch(img_path, sequence, box):
    '''
    可视化标注信息生成的patch监督序列
    
    Parameters:
        img_path - 图片路径
        sequence - 14*14的矩阵,每个元素为0或1
        box - 14*14*4的矩阵,每个元素为xmin, ymin, xmax, ymax
    '''
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    for i in range(0, len(box), 4):
        cv2.rectangle(img, (round(box[i]), round(box[i + 1])), (round(box[i + 2]), round(box[i + 3])), (0, 255, 0), 2)
    for i in range(14):  # 行
        for j in range(14):  # 列
            if sequence[i * 14 + j] == 1:
                cv2.rectangle(img, (j * 16, i * 16), ((j + 1) * 16, (i + 1) * 16), (0, 0, 255), 2)
    cv2.imwrite('./images/draw/gt_patch.jpg', img)


def gt_box(img_path, forecast, box):
    '''
    可视化由patch监督序列恢复的box与标注信息
    
    Parameters:
        img_path - 图片路径
        sequence - 14*14的矩阵,每个元素为0或1
        box: 14*14*4的矩阵,每个元素为xmin, ymin, xmax, ymax
    '''
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    for i in range(0, len(box), 4):
        cv2.rectangle(img, (round(box[i]), round(box[i + 1])), (round(box[i + 2]), round(box[i + 3])), (0, 255, 0), 2)
    for i in range(0, len(forecast), 4):
        cv2.rectangle(img, (forecast[i], forecast[i + 1]), (forecast[i + 2], forecast[i + 3]), (0, 0, 255), 2)
    cv2.imwrite('./images/draw/gt_box.jpg', img)


def tensor2im(image_tensor, imtype=np.uint8):
    '''
    将数据预处理后tensor格式的图像转换为原RGB的numpy格式

    Parameters:
        image_tensor - 输入的tensor,维度为CHW,注意这里没有batch size的维度
        imtype - 转换后的numpy的数据类型
    
    Returns:
        image_numpy - 转换后的numpy格式的图像
    '''
    mean = [0.485, 0.456, 0.406]  # dataLoader中设置的mean参数，需要从dataloader中拷贝过来
    std = [0.229, 0.224, 0.225]  # dataLoader中设置的std参数，需要从dataloader中拷贝过来
    if not isinstance(image_tensor, np.ndarray):
        if isinstance(image_tensor, torch.Tensor):  # 如果传入的图片类型为torch.Tensor，则读取其数据进行下面的处理
            image_tensor = image_tensor.data
        else:
            return image_tensor
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):  # 反标准化，乘以方差，加上均值
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255  #反ToTensor(),从[0,1]转为[0,255]
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
    else:  # 如果传入的是numpy数组,则不做处理
        image_numpy = image_tensor
    return image_numpy.astype(imtype)


def contrast_box(img_tensor, forecast, box):
    '''
    可视化box和gt_box
    
    Parameters:
        img_tensor - 图片tensor存放预测box坐标的list
        forecast - 存放预测box坐标的list
    '''
    img = tensor2im(img_tensor)
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i in range(0, len(box), 4):
        cv2.rectangle(img_cv, (round(box[i].item()), round(box[i + 1].item())), (round(box[i + 2].item()), round(box[i + 3].item())), (0, 255, 0), 2)
    for i in range(0, len(forecast), 4):
        cv2.rectangle(img_cv, (forecast[i], forecast[i + 1]), (forecast[i + 2], forecast[i + 3]), (0, 0, 255), 2)
    cv2.imwrite('./images/draw/contrast_box.jpg', img_cv)


def forecast_box(img_tensor, output, sequence, patch_threshold):
    '''
    利用模型预测的patch生成box,并可视化box与gt box
    (设定这个函数的目的是contrast_box函数无法可视化不同阈值的效果)
    
    Parameters:
        img_tensor - 数据预处理过的tensor格式的图像
        output - 14*14的矩阵,每个元素为概率值
        sequence: 14*14的矩阵,每个元素为0或1
        patch_threshold: patch激活的阈值,是一个值
    '''
    img = tensor2im(img_tensor)
    for threshold in patch_threshold:
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        output_s = output > threshold
        forecast = patch_to_box(output_s)
        gt = patch_to_box(sequence)
        for i in range(0, len(gt), 4):
            cv2.rectangle(img_cv, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 0), 2)
        for i in range(0, len(forecast), 4):
            cv2.rectangle(img_cv, (forecast[0], forecast[1]), (forecast[2], forecast[3]), (0, 0, 255), 2)
        cv2.imwrite('./images/draw/forecast_box_{}.jpg'.format(threshold), img_cv)


def forecast_patch(img_tensor, output, sequence, patch_threshold):
    '''
    可视化模型预测的patch与对应的监督信息
    
    Parameters:
        img_tensor - 数据预处理过的tensor格式的图像
        output - 14*14的矩阵,每个元素为概率值
        sequence -  14*14的矩阵,每个元素为0或1
        patch_threshold -  一个list,里面存放不同的阈值
    '''
    img = tensor2im(img_tensor)
    sequence = sequence.detach().cpu().numpy()
    output = output.detach().cpu().numpy()
    for threshold in patch_threshold:
        # img为array img类型，需要转换为cv2的类型
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for i in range(14):  # 行
            for j in range(14):  # 列
                if sequence[i * 14 + j] == 1:
                    # 16*16的patch上色
                    cv2.rectangle(img_cv, (j * 16, i * 16), ((j + 1) * 16, (i + 1) * 16), (0, 255, 0), 2)  # 绿
                if output[i * 14 + j] > threshold:
                    if sequence[i * 14 + j] == 0:
                        # 16*16的patch上色
                        cv2.rectangle(img_cv, (j * 16, i * 16), ((j + 1) * 16, (i + 1) * 16), (255, 128, 0), 2)  # 浅蓝
                    else:
                        # 16*16的patch上色
                        cv2.rectangle(img_cv, (j * 16, i * 16), ((j + 1) * 16, (i + 1) * 16), (0, 0, 255), 2)  # 红
        cv2.imwrite('./images/draw/forecast_patch_{}.jpg'.format(threshold), img_cv)


def prob_chart(index_json, outputs):
    '''
    绘制模型分类预测的种类概率直方图
    
    Parameters:
        index_json - 存放类别名称的json文件
        outputs - 模型分类预测输出
    '''
    index_dict = {}
    with open(index_json) as f:
        data = json.load(f)
        for i in range(len(data)):
            index_dict[list(data.keys())[i]] = list(data.values())[i]

    plt.figure()
    # ouputs进行softmax
    outputs = torch.softmax(outputs, dim=1)
    # outputs中前十个最大数值的下标
    topk = torch.topk(outputs, 5)[1]
    # topk中的下标对应的类别名称
    topk_name = [index_dict[str(i.item())].split(', ')[0] for i in topk[0]]
    # topk中的下标对应的数值
    topk_value = [outputs[0][i.item()].item() for i in topk[0]]
    plt.bar(topk_name, topk_value)
    # 显示数值
    for x, y in enumerate(topk_value):
        plt.text(x, y, '%.2f' % y, ha='center', va='bottom')
    # name放不下，调整柱状图长度
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=45)
    plt.savefig('images/draw/predict_cla.jpg')


def main(args):
    if args.test_function == 'gt_patch':
        with open('/data_1/langwenpeng/imagenet/train_index.txt', 'r') as f:
            line = f.readlines()
            # 固定随机种子
            random.seed(0)
            random.shuffle(line)
            idx = 0
            while (True):
                words = line[idx].split(' ')
                img_path = os.path.join('/data_1/langwenpeng/imagenet/task2_train', words[0])
                idx += 1
                positions = []
                for i in range(2, len(words)):
                    positions.extend(words[i].split(','))
                box = []
                for i in range(0, len(positions), 4):
                    box.extend([int(positions[i]), int(positions[i + 1]), int(positions[i + 2]), int(positions[i + 3])])
                img = Image.open(img_path).convert('RGB')
                before_w, before_h = img.size
                after_w, after_h = 224, 224
                for i in range(0, len(box), 4):
                    box[i] = box[i] * after_w / before_w
                    box[i + 1] = box[i + 1] * after_h / before_h
                    box[i + 2] = box[i + 2] * after_w / before_w
                    box[i + 3] = box[i + 3] * after_h / before_h
                sequence = torch.zeros(196)
                for i in range(0, len(box), 4):
                    xmin = min(13, int(box[i] // 16 + (box[i] % 16 > 8)))
                    ymin = min(13, int(box[i + 1] // 16 + (box[i + 1] % 16 > 8)))
                    xmax = min(13, int(box[i + 2] // 16 - (box[i + 2] % 16 < 8)))
                    ymax = min(13, int(box[i + 3] // 16 - (box[i + 3] % 16 < 8)))
                    for i in range(ymin, ymax + 1):
                        for j in range(xmin, xmax + 1):
                            sequence[i * 14 + j] = 1
                gt_patch(img_path, sequence, box)
                if input() == 'q':
                    break

    elif args.test_function == 'gt_box':
        with open('/data_1/langwenpeng/imagenet/train_index.txt', 'r') as f:
            line = f.readlines()
            random.seed(0)
            # 随机打乱random
            random.shuffle(line)
            idx = 0
            while (True):
                words = line[idx].split(' ')
                img_path = os.path.join('/data_1/langwenpeng/imagenet/task2_train', words[0])
                idx += 1
                positions = []
                for i in range(2, len(words)):
                    positions.extend(words[i].split(','))
                box = []
                for i in range(0, len(positions), 4):
                    box.extend([int(positions[i]), int(positions[i + 1]), int(positions[i + 2]), int(positions[i + 3])])
                img = Image.open(img_path).convert('RGB')
                before_w, before_h = img.size
                after_w, after_h = 224, 224
                for i in range(0, len(box), 4):
                    box[i] = int(box[i] * after_w / before_w)
                    box[i + 1] = int(box[i + 1] * after_h / before_h)
                    box[i + 2] = int(box[i + 2] * after_w / before_w)
                    box[i + 3] = int(box[i + 3] * after_h / before_h)
                sequence = torch.zeros(196)
                for i in range(0, len(box), 4):
                    xmin = min(13, int(box[i] // 16 + (box[i] % 16 > 8)))
                    ymin = min(13, int(box[i + 1] // 16 + (box[i + 1] % 16 > 8)))
                    xmax = min(13, int(box[i + 2] // 16 - (box[i + 2] % 16 < 8)))
                    ymax = min(13, int(box[i + 3] // 16 - (box[i + 3] % 16 < 8)))
                    for i in range(ymin, ymax + 1):
                        for j in range(xmin, xmax + 1):
                            sequence[i * 14 + j] = 1
                b = patch_to_box(sequence)
                gt_box(img_path, b, box)
                if input() == 'q':
                    break

    elif args.test_function == 'forecast_box':
        device = torch.device('cpu')
        data_transform = {
            "train": transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }
        nw = 8
        # 实例化训练数据集
        _, val_dataset = get_dataset("imagenet", data_transform)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
        model = create_model(num_classes=1000).to(device)
        print(model.load_state_dict(torch.load(args.weights_path, map_location='cpu'), strict=False))
        with torch.no_grad():
            model.eval()
            for images, labels, sequences, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                sequences = sequences.to(device)
                ouputs = model(images)
                preds = torch.argmax(ouputs, dim=1)
                ids = torch.zeros(1, 1000)
                ids[0][preds] = 1
                outputs = model.localize(images, ids)
                forecast_box(images[0], sequences[0], outputs[0], args.patch_threshold)
                if input() == 'q':
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 设置测试函数
    parser.add_argument('--test_function', type=str, default='gt_box', help='test function[gt_patch, gt_box, forecast_box]')

    # 模型权重路径
    parser.add_argument('--weights_path', type=str, default='./weights/before/6_loc_best_0.7192_0.7439.pth', help='initial weights path')

    # 预测patch阈值
    parser.add_argument('--patch_threshold', type=list, default=[0.7, 0.5, 0.3])

    args = parser.parse_args()
    main(args)