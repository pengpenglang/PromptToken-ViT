import torch
import argparse

from tqdm import tqdm
from torchvision import transforms
from utils.data import get_dataset
from utils.box import cal_box, patch_to_box
from utils.draw import forecast_box, contrast_box
from model.vit_model import vit_base_patch16_224 as create_model


def get_box(model, images, ids, sequence, threshold, device):
    '''
    使用模型预测的patch生成box坐标
    
    Parameters:
        model - 使用的模型
        images - 输入的图像二维tensor
        ids - 输入的图像对应的类别编号
        sequence - 输入的图像patch概率值序列
        threshold - 用来两阶段生成box的阈值
        device - 运行模型的gpu
    
    Returns:
        update_box - 存储输入图像对应的box坐标列表
    '''
    outputs = model.localize(images, ids)
    outputs = outputs > threshold[0]
    # 调用utils中的函数将patch转换为box
    box = patch_to_box(outputs[0])
    update_patch = torch.zeros(196).to(device)
    update_box = []
    for i in range(0, len(box), 4):
        xmin, ymin, xmax, ymax = box[i:i + 4]
        # 对于每个box,取出其在images中的位置并填充无效值到224*224,然后再利用模型预测,得到新的box
        obj_prop = torch.zeros([1, 3, 224, 224]).to(device)
        obj_prop[:, :, ymin:ymax, xmin:xmax] = 1
        # 获得只有目标区域的图像为1
        obj_img = images * obj_prop
        obj_pred = model(obj_img)
        # if obj_pred.argmax(dim=1)[0] == ids.argmax(dim=1)[0]:
        obj_outputs = model.localize(obj_img, ids)
        # 超出box的patch直接置0
        for num in obj_outputs[0]:
            obj_xmin, obj_ymin, obj_xmax, obj_ymax = (num % 14) * 16, (torch.div(num, 14, rounding_mode='trunc')) * 16, (num % 14 + 1) * 16, torch.div(num, 14, rounding_mode='trunc') * 16
            if obj_xmin < xmin and obj_xmax > xmax and obj_ymin < ymin and obj_ymax > ymax:
                obj_outputs[0][num] = 0

        update_patch = (update_patch + obj_outputs[0])
        if outputs[0].sum() / 196 <= 0.3:
            obj_outputs = obj_outputs > threshold[1]
        elif outputs[0].sum() / 196 <= 0.5:
            obj_outputs = obj_outputs > threshold[2]
        else:
            obj_outputs = obj_outputs > threshold[3]
        update_box.extend(patch_to_box(obj_outputs[0]))
    # 画出预测的box
    forecast_box(images[0], update_patch, sequence[0], [0.3, 0.5, 0.7])
    return update_box


def main(args):
    # 运行gpu
    device = torch.device("cuda:3")

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # 加载模型
    nw = 8
    model = create_model(num_classes=1000).to(device)
    print(model.load_state_dict(torch.load(args.weights_path, map_location=device), strict=False))

    # 获取数据集
    _, val_dataset = get_dataset("imagenet", data_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=nw)
    val_loader = tqdm(val_loader, ncols=150)

    if args.test_mode == 'top1':
        ac, wa = 0, 0
        with torch.no_grad():
            for images, labels, sequences, boxes in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                sequences = sequences.to(device)
                outputs = model(images)
                if outputs.argmax(dim=1)[0] == labels[0]:
                    ids = torch.zeros([1, 1000]).to(device)
                    ids[0][labels] = 1
                    forecast = get_box(model, images.clone(), ids, sequences, args.patch_threshold, device)
                    # contrast_box(images[0], forecast, boxes[0])
                    # print(forecast)
                    # 调用utils中的函数统计当前batch预测的box正确和错误的数量
                    res = cal_box(forecast[0:4], boxes[0])
                    ac += res[0]
                    wa += res[1]
                else:
                    wa += 1
                # if input() == 'q':
                #     break'
                val_loader.set_description("Test: ac: {}, wa: {}, acc: {:.4f}".format(ac, wa, ac / (ac + wa)))

    elif args.test_mode == 'top5':
        ac, wa = 0, 0
        with torch.no_grad():
            for images, labels, sequences, boxes in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                sequences = sequences.to(device)
                outputs = model(images)
                if labels[0] in outputs.topk(5, dim=1)[1]:
                    ids = torch.zeros([1, 1000]).to(device)
                    ids[0][labels] = 1
                    forecast = get_box(model, images.clone(), ids, sequences, args.patch_threshold, device)
                    # contrast_box(images[0], forecast, boxes[0])
                    # print(forecast)
                    res = cal_box(forecast[0:4], boxes[0])
                    ac += res[0]
                    wa += res[1]
                else:
                    wa += 1
                # if input() == 'q':
                #     break'
                val_loader.set_description("Test: ac: {}, wa: {}, acc: {:.4f}".format(ac, wa, ac / (ac + wa)))

    elif args.test_mode == 'gt':
        ac, wa = 0, 0
        with torch.no_grad():
            for images, labels, sequences, boxes in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                sequences = sequences.to(device)
                ids = torch.zeros([1, 1000]).to(device)
                ids[0][labels] = 1
                forecast = get_box(model, images.clone(), ids, sequences, args.patch_threshold, device)
                # contrast_box(images[0], forecast, boxes[0])
                # print(forecast)
                res = cal_box(forecast[0:4], boxes[0])
                ac += res[0]
                wa += res[1]
                # if input() == 'q':
                #     break'
                val_loader.set_description("Test: ac: {}, wa: {}, acc: {:.4f}".format(ac, wa, ac / (ac + wa)))
    else:
        print('test_mode error!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 数据集种类数量
    parser.add_argument('--num_classes', type=int, default=1000)

    # 数据集路径
    parser.add_argument('--data_path', type=str, default="/data_1/langwenpeng/imagenet")

    # 模型权重路径
    parser.add_argument('--weights_path', type=str, default='weights/0_cla_best_0.8272_0.7021.pth', help='initial weights path')

    # patch生成box的两阶段阈值设置
    parser.add_argument('--patch_threshold', type=list, default=[0.3, 0.5, 0.5, 0.5], help='index: 0-first, 1-second tiny, 2-second middle, 3-second large')

    # 评测指标
    parser.add_argument('--test_mode', type=str, default='gt', help='[top1, top5, gt]')

    opt = parser.parse_args()
    main(opt)