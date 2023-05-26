import os
import json
import torch
import argparse
import matplotlib.pyplot as plt

from utils.data import get_dataset
from torchvision import transforms
from model.vit_model import vit_base_patch16_224 as create_model
from utils.draw import forecast_patch, prob_chart
from test_box import get_box


def main(args):
    # 设定运行设备
    device = torch.device("cpu")

    # 定义数据预处理方式
    data_transform = {
        "train": transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # 实例化验证数据集
    _, val_dataset = get_dataset("imagenet", data_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

    # 实例化模型并加载权重
    model = create_model(num_classes=1000).to(device)
    print(model.load_state_dict(torch.load(args.weights_path, map_location=device), strict=False))

    with torch.no_grad():
        model.eval()
        for images, labels, sequences, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            sequences = sequences.to(device)
            outputs = model(images)
            prob_chart(args.index_json, outputs)
            pred_cla = torch.argmax(outputs, dim=1)
            ids = torch.zeros(1, 1000)
            ids[0][pred_cla] = 1
            pred_loc = model.localize(images, ids)
            forecast_patch(images[0].clone(), pred_loc[0], sequences[0], [0.3, 0.5, 0.7])
            get_box(model, images.clone(), ids, sequences, args.patch_threshold, device)
            if input() == 'q':
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 模型权重路径
    parser.add_argument('--weights_path', type=str, default='weights/loc_best_0.8054_0.7755.pth')

    # 种类索引json文件路径
    parser.add_argument('--index_json', type=str, default='/data_1/langwenpeng/imagenet/class_indices.json', help='index json file')

    # patch预测阈值
    parser.add_argument('--patch_threshold', type=list, default=[0.3, 0.5, 0.5, 0.5], help='index: 0-active, 1-tiny, 2-middle, 3-large')

    args = parser.parse_args()
    main(args)