import os
import torch
import argparse

from model.vit_model import vit_base_patch16_224 as create_model
from torchvision import transforms
from utils.data import get_dataset
from utils.train_epoch import cla_train_one_epoch, cla_evaluate, init_distributed_mode, clean_up
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter


def main(args):

    # 初始化分布式模式
    init_distributed_mode(args)

    # 获取当前gpu设备
    device = torch.device("cuda", args.gpu)

    # 每个进程的数据加载器的进程数
    args.num_workers = min([os.cpu_count() // args.world_size, 8])

    # 主进程创建tensorboard和权重保存文件夹
    if args.rank == 0:
        print(args)
        tb_writer = SummaryWriter()
        if os.path.exists("./weights/transfer") is False:
            os.makedirs("./weights/transfer")
        print('Using {} dataloader workers every process'.format(args.num_workers))

    # 定义数据预处理方式
    data_transform = {
        "train": transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # 加载数据集
    train_dataset, val_dataset = get_dataset(args.dataset, data_transform)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    # 创建模型
    model = create_model(num_classes=args.num_classes).to(device)

    # 加载预训练权重
    if os.path.exists(args.weights_path):
        weights_dict = torch.load(args.weights_path, map_location=device)  # 主进程加载权重后会自动广播到其他进程
        del_weights = ['head.weight', 'head.bias']
        for key in del_weights:
            del weights_dict[key]
        load_weights_dict = {k: v for k, v in weights_dict.items() if k in model.state_dict()}  # 只加载模型中存在的权重
        if args.rank == 0:  # 只有主进程进行操作
            print(model.load_state_dict(load_weights_dict, strict=False))  # strict=False表示加载的权重不完全匹配也不会报错

    # 冻结权重
    for name, param in model.named_parameters():
        for name, param in model.named_parameters():
            if name in ['head.weight', 'head.bias']:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # 定义优化器、学习率衰减策略与损失函数
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(pg, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=args.patience, verbose=True, eps=1e-12, min_lr=1e-12)
    loss_functoin = torch.nn.CrossEntropyLoss()

    # 模型转换为分布式模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # 开始训练
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_loss, train_acc = cla_train_one_epoch(model, optimizer, loss_functoin, train_loader, device, epoch)
        val_sampler.set_epoch(epoch)
        val_loss, val_acc = cla_evaluate(model, loss_functoin, val_loader, device, epoch)
        scheduler.step(train_acc)

        # 保存训练日志
        if args.rank == 0:
            tags = ['lr', 'train_acc', 'train_loss', 'val_acc', 'val_loss']
            tb_writer.add_scalar(tags[0], optimizer.param_groups[0]['lr'], epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], train_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], val_loss, epoch)
            if val_acc > best_acc:
                best_acc = val_acc
                os.system("rm ./weights/transfer/best_*.pth")
                torch.save(model.state_dict(), "./weights/transfer/best_{:.4f}.pth".format(val_acc))

    if args.rank == 0:
        tb_writer.close()
        clean_up(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 超参数设置
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for classification head')
    parser.add_argument('--epochs', type=int, default=55)
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for classification head')
    parser.add_argument('--dataset', type=str, default="dtd", help='dataset name in [cifar100, caltech256, dtd]')
    parser.add_argument('--lr', type=float, default=0.0001)

    # 数据集种类数量
    parser.add_argument('--num_classes', type=int, default=100, help='number of classes for classification head')

    # 模型权重路径
    # parser.add_argument('--weights_path', type=str, default='./model/vit_base_patch16_224.pth', help='path to pretrained weights')
    parser.add_argument('--weights_path', type=str, default='weights/cla_best_0.8284_0.7154.pth', help='path to pretrained weights')

    # scheduler自动调整学习率的patience
    parser.add_argument('--patience', type=int, default=2)

    # DDP运行参数(默认不修改)
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    opt = parser.parse_args()
    main(opt)