import os
import json
import torch
import argparse
import socket
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from torchsummary import summary
from ptflops import get_model_complexity_info
from utils.data import get_dataset
from torchvision import transforms
from matplotlib import pyplot as plt
from visualizer import get_local
from utils.visualize import visualize_grid_to_grid_with_cls, visualize_head

get_local.activate()
from model.vit_model import vit_base_patch16_224 as create_model
from flashtorch.saliency import Backprop
from torch.utils.tensorboard import SummaryWriter


def kernel_weights(model):
    # 获取卷积核权重并转移到cpu
    kernal_filters = model.patch_embed.proj.weight
    kernal_filters = kernal_filters.detach().cpu()

    # 每次可视化36个卷积核
    now = 0
    while True:
        now += 36
        for step, i in enumerate(range(now, now + 36)):
            mean = kernal_filters[i].mean()
            std = kernal_filters[i].std() if kernal_filters[i].std() != 0 else 1e-7
            standardized = kernal_filters[i].sub(mean).div(std).mul(0.15)
            output = standardized.add(0.5).clamp(0, 1)
            formatted = output.clone().squeeze(0).permute(1, 2, 0).detach()
            plt.subplot(6, 6, step + 1)
            plt.axis('off')
            plt.imshow(formatted, cmap='gray')
        plt.savefig('./images/analyse/kernal_weights.jpg')
        if input() == 'q':
            break


def guided_backprop(model, val_loader, device):
    for images, labels, _, _ in val_loader:
        images = images.to(device)
        # 必须设置为True否则无法计算梯度
        images.requires_grad = True
        # 目标类别作为guide
        labels = labels.to(device)
        backprop = Backprop(model)
        backprop.visualize(images, labels[0].item(), guided=True)
        plt.savefig('./images/analyse/grad_cam.jpg')
        if input() == 'q':
            break


def attention_visualize(model, images_path, label, class_indices):
    '''
    可视化transformer编码层的关注区域
    
    Parameters:
        model - 使用的预测模型
        images_path - plot需要加载图像这里设置传入图像路径
        labl - 图像对应的标签
        class_indices - 类别索引字典
    '''
    for i in range(len(label)):
        # 读入RGB图像并resize
        image = Image.open(images_path[i]).convert('RGB')
        image = image.resize((224, 224))

        # 设置图像预处理
        trans = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # 获得图像的tensor
        input_tensor = trans(image).unsqueeze(0)

        # 清空之前缓存并运行模型获取当前的统计结果
        get_local.clear()
        with torch.no_grad():
            outputs = model(input_tensor)
        print('label: {}, predict: {}'.format(class_indices[label[i]], class_indices[outputs.argmax().item()]))
        cache = get_local.cache
        attention_maps = cache['Attention.forward']
        print('attention_maps:', len(attention_maps), 'shape:', attention_maps[0].shape)

        # 清空当前图像
        plt.clf()

        # 可视化最后编码层的第一个head的196个patch关联热力图
        visualize_head(attention_maps[11][0, 1])
        plt.savefig('./images/analyse/attention_head_cla.jpg')

        # 每个编码层的head值合并
        attention_maps = np.sum(attention_maps, axis=2, keepdims=True)

        # 可视化最后编码层指定位置与其他patch的特征图
        visualize_grid_to_grid_with_cls(attention_maps[11][0, 0, :, :], 103, image)
        plt.savefig('./images/analyse/attention_qkv_cla.jpg')

        # 上面是分类可视化再执行一次定位可视化
        get_local.clear()
        with torch.no_grad():
            ids = torch.zeros(1, 1000)
            ids[0][label[i]] = 1
            outputs = model.localize(input_tensor, ids)
        cache = get_local.cache
        attention_maps = cache['Attention.forward']
        plt.clf()
        visualize_head(attention_maps[11][0, 1])
        plt.savefig('./images/analyse/attention_head_loc.jpg')
        visualize_head(attention_maps[11][0, 1])
        attention_maps = np.sum(attention_maps, axis=2, keepdims=True)  # 合并head
        visualize_grid_to_grid_with_cls(attention_maps[11][0, 0, :, :], 103, image)
        plt.savefig('./images/analyse/attention_qkv_loc.jpg')

        if input() == 'q':
            break


def parameter_analysis(model):
    # 设置日志文件名称
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(current_time + "_" + socket.gethostname())
    writer = SummaryWriter(log_dir='./logs/parameter_analysis/{}'.format(log_dir))

    # 统计模型参数
    for name, param in model.named_parameters():
        writer.add_histogram(name + '_data', param)


def model_graph(model):
    # 设置日志文件名称
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(current_time + "_" + socket.gethostname())
    writer = SummaryWriter(log_dir='./logs/model_graph/{}'.format(log_dir))

    # 随机一个输入
    image = torch.rand(1, 3, 224, 224)
    writer.add_graph(model, image)
    writer.flush()


def dataset_stats(loader):
    cnt = [0, 0, 0, 0, 0]
    loader = tqdm(loader)
    for _, _, sequences, _ in loader:
        # 使用patch为1的数量作为比例
        rate = sequences[0].sum() / 196
        if rate < 0.2:
            cnt[0] += 1
        elif rate < 0.4:
            cnt[1] += 1
        elif rate < 0.6:
            cnt[2] += 1
        elif rate < 0.8:
            cnt[3] += 1
        else:
            cnt[4] += 1
        loader.set_description('rate: {}'.format(cnt))


def main(args):
    # 运行设备
    device = torch.device("cpu")

    # 数据预处理
    data_transform = {"train": transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()]), "val": transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])}

    # 加载模型
    nw = 8
    model = create_model(num_classes=1000).to(device)
    print(model.load_state_dict(torch.load(args.weights_path, map_location='cpu'), strict=False))

    # 加载json文件
    with open('/data_1/langwenpeng/imagenet/class_indices.json', 'r') as f:
        class_indices = json.load(f)
        class_indices = {int(k): v.split(', ')[0] for k, v in class_indices.items()}

    model.eval()

    if args.kernel_weights:
        kernel_weights(model)

    if args.guided_backprop:
        _, val_dataset = get_dataset("imagenet", data_transform)
        torch.manual_seed(0)  # 固定随机数
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=nw)
        guided_backprop(model, val_loader, device)

    if args.attention_visualize:
        images_path = []
        label = []
        with open('/data_1/langwenpeng/imagenet/val_index.txt', 'r') as f:
            random.seed(0)  # 固定随机数
            for line in random.sample(f.readlines(), 1000):
                line = line.strip()
                words = line.split(' ')
                images_path.append('/data_1/langwenpeng/imagenet/val' + '/' + words[0])
                label.append(int(words[1]))
        attention_visualize(model, images_path, label, class_indices)

    if args.parameter_analysis:
        parameter_analysis(model)

    if args.model_graph:
        model_graph(model)

    if args.model_size:
        summary(model, (3, 224, 224), device='cpu')
        print('\n\n')
        ops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)

    if args.dataset_stats:
        _, dataset = get_dataset("imagenet", data_transform)
        # 固定随机数
        torch.manual_seed(0)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=nw)
        dataset_stats(loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 数据集种类数量
    parser.add_argument('--num_classes', type=int, default=1000)

    # 模型权重路径
    # parser.add_argument('--weights_path', type=str, default='./model/vit_base_patch16_224.pth', help='initial weights path')
    parser.add_argument('--weights_path', type=str, default='weights/合并训练/12_loc_best_0.7871_0.7892.pth', help='initial weights path')

    # patch预测0/1的阈值设定
    parser.add_argument('--patch_threshold', type=list, default=[0.7, 0.5, 0.3])

    # 图像嵌入层的CNN卷积核可视化
    parser.add_argument('--kernel_weights', type=bool, default=False)

    # guided backprop可视化CNN关注区域
    parser.add_argument('--guided_backprop', type=bool, default=False)

    # 可视化transformer编码层的关注区域
    parser.add_argument('--attention_visualize', type=bool, default=False)

    # 模型各层参数分布分析
    parser.add_argument('--parameter_analysis', type=bool, default=False)

    # 模型图生成
    parser.add_argument('--model_graph', type=bool, default=False)

    # 模型规格统计
    parser.add_argument('--model_size', type=bool, default=False)

    # 统计数据集目标不同尺寸数量
    parser.add_argument('--dataset_stats', type=bool, default=True)

    opt = parser.parse_args()
    main(opt)