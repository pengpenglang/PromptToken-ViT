import os
import torch
import argparse

from tqdm import tqdm
from utils.data import get_dataset
from torchvision import transforms
from model.vit_model import vit_base_patch16_224 as create_model


def main(args):
    # 运行设备
    device = torch.device("cuda:5")

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # 加载模型
    nw = 8
    _, val_dataset = get_dataset(args.dataset, data_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    model = create_model(num_classes=args.num_classes).to(device)
    print(model.load_state_dict(torch.load(args.weights_path, map_location=device), strict=False))

    if args.test_mode == 'cla_top1':
        cla_acc = 0.0
        val_loader = tqdm(val_loader, ncols=100)
        with torch.no_grad():
            model.eval()
            for step, (images, labels, _, _) in enumerate(val_loader):
                images = images.to(device)
                batch_size = images.shape[0]
                labels = labels.to(device)
                outputs = model(images)
                pred = torch.max(outputs, dim=1)[1]
                pred_num = torch.eq(pred, labels).sum().float()
                cla_acc = (cla_acc * step + pred_num / batch_size) / (step + 1)
                val_loader.set_description("cla acc: {:.4f}".format(cla_acc))

    if args.test_mode == 'cla_top5':
        cla_acc = 0.0
        val_loader = tqdm(val_loader, ncols=100)
        with torch.no_grad():
            model.eval()
            for step, (images, labels, _, _) in enumerate(val_loader):
                images = images.to(device)
                batch_size = images.shape[0]
                labels = labels.to(device)
                outputs = model(images)
                # 取出前5个索引
                pred = torch.topk(outputs, 5, dim=1)[1]
                # 对于labels逐行判断是否在pred对应行中
                pred_num = torch.sum(torch.eq(pred, labels.unsqueeze(1)).any(1).float())
                cla_acc = (cla_acc * step + pred_num / batch_size) / (step + 1)
                val_loader.set_description("cla acc: {:.4f}".format(cla_acc))

    if args.test_mode == 'loc_gt':
        loc_acc = {threshold: 0 for threshold in args.patch_threshold}
        with torch.no_grad():
            model.eval()
            for step, (images, labels, sequences, _) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                sequences = sequences.to(device)
                batch_size = images.shape[0]
                ids = []
                for i in range(batch_size):
                    tmp_id = torch.zeros(1000).to(device)
                    tmp_id[labels[i]] = 1
                    ids.append(tmp_id)
                ids = torch.stack(ids).to(device)
                outputs = model.localize(images, ids)
                for threshold in args.patch_threshold:
                    pred = outputs > threshold
                    in_num = (pred * sequences).sum(dim=1)
                    out_num = ((pred + sequences) - (pred * sequences)).sum(dim=1)
                    iou = (in_num / out_num).sum() / batch_size
                    loc_acc[threshold] = (loc_acc[threshold] * step + iou) / (step + 1)
                val_loader.set_description("loc acc: {}".format({k: "{:.4f}".format(v) for k, v in loc_acc.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 定义数据集种类数量
    parser.add_argument('--num_classes', type=int, default=1000)

    # 定义检测并行数
    parser.add_argument('--batch_size', type=int, default=512)

    # 选择检测数据集
    parser.add_argument('--dataset', type=str, default="imagenet", help='dataset name in [imagenet, cifar100]]')

    # 定义权重路径
    # parser.add_argument('--weights_path', type=str, default='model/vit_base_patch16_224.pth', help='initial weights path')
    parser.add_argument('--weights_path', type=str, default='weights/0_loc_best_0.7984_0.7821.pth', help='initial weights path')

    # 定义patch阈值
    parser.add_argument('--patch_threshold', type=list, default=[0.7, 0.5, 0.3])

    # 评测指标
    parser.add_argument('--test_mode', type=str, default='cla_top1', help='[cla_top1, cla_top5, loc_gt]')
    opt = parser.parse_args()
    main(opt)