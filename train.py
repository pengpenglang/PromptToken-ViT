import os
import torch
import argparse

from model.vit_model import vit_base_patch16_224 as create_model
from torchvision import transforms
from utils.loss import BCEFocalLoss
from utils.train_epoch import train_one_epoch, evaluate, init_distributed_mode, clean_up
from utils.data import get_dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler


def main(args):
    # 多卡训练没有GPU报错
    if torch.cuda.is_available() is False:
        raise EnvironmentError('GPU is not available')

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
        if os.path.exists("./weights/") is False:
            os.makedirs("./weights")
        print('Using {} dataloader workers every process'.format(args.num_workers))

    # 定义数据预处理方式
    data_transform = {
        "train": transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # 实例化训练数据集
    train_dataset, val_dataset = get_dataset(args.dataset, data_transform)
    # 将数据集分成world_size份，每个进程只加载自己的数据
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    # 实例化模型
    model = create_model(num_classes=args.num_classes).to(device)

    # 加载预训练权重
    if os.path.exists(args.weights_path):
        weights_dict = torch.load(args.weights_path, map_location=device)  # 主进程加载权重后会自动广播到其他进程
        load_weights_dict = {k: v for k, v in weights_dict.items() if k in model.state_dict()}  # 只加载模型中存在的权重
        if args.rank == 0:  # 只有主进程进行操作
            print(model.load_state_dict(load_weights_dict, strict=False))  # strict=False表示加载的权重不完全匹配也不会报错

    # 冻结层
    if args.freeze_layers:
        for name, param in model.named_parameters():
            for name, param in model.named_parameters():
                if name in []:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    else:  # 只有不冻结层且存在BN层的时候才需要使用SyncBN
        if args.syncBN:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        for param in model.parameters():
            param.requires_grad = True

    # 定义分类优化器、学习率调整器、损失函数
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(pg, lr=args.lr, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=args.patience, verbose=True, eps=1e-12, min_lr=1e-12)
    cla_loss_function = torch.nn.CrossEntropyLoss()
    loc_loss_function = torch.nn.BCELoss()

    # 加载checkpoint
    if os.path.exists(args.checkpoint_path) and args.rank == 0:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        print(model.load_state_dict(checkpoint['weights']))
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0

    # 将模型转换为分布式模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)  # 将模型转换为分布式模型

    # 开始训练
    cla_best_acc = 0.0
    loc_best_acc = 0.0
    for epoch in range(start_epoch, args.epochs):

        train_sampler.set_epoch(epoch)
        cla_train_loss, cla_train_acc, loc_train_loss, loc_train_acc = train_one_epoch(model, optimizer, cla_loss_function, loc_loss_function, train_loader, device, epoch, args.patch_threshold)

        val_sampler.set_epoch(epoch)
        cla_val_loss, cla_val_acc, loc_val_loss, loc_val_acc = evaluate(model, cla_loss_function, loc_loss_function, val_loader, device, epoch, args.patch_threshold)

        # 设置学习率调整依据
        scheduler.step(loc_train_acc)

        # 主进程保存checkpoint与训练日志
        if args.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'weights': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            if cla_val_acc > cla_best_acc:
                cla_best_acc = cla_val_acc
                os.system("rm ./weights/cla_best_*.pth")
                torch.save(model.module.state_dict(), os.path.join("./weights", "cla_best_{:.4f}_{:.4f}.pth".format(cla_best_acc, loc_val_acc)))
            if loc_val_acc > loc_best_acc:
                loc_best_acc = loc_val_acc
                os.system("rm ./weights/loc_best_*.pth")
                torch.save(model.module.state_dict(), os.path.join("./weights", "loc_best_{:.4f}_{:.4f}.pth".format(cla_val_acc, loc_best_acc)))
            os.system('rm ./weights/checkpoint_*.pth')
            torch.save(checkpoint, './weights/checkpoint_{}_{:.4f}_{:.4f}.pth'.format(epoch, cla_val_acc, loc_val_acc))

            tags = [
                "lr",
                "cla_train_loss",
                "cla_train_acc",
                "cla_val_loss",
                "cla_val_acc",
                "loc_train_loss",
                "loc_train_acc",
                "loc_val_loss",
                "loc_val_acc",
            ]
            tb_writer.add_scalar(tags[0], optimizer.param_groups[0]['lr'], epoch)
            tb_writer.add_scalar(tags[1], cla_train_loss, epoch)
            tb_writer.add_scalar(tags[2], cla_train_acc, epoch)
            tb_writer.add_scalar(tags[3], cla_val_loss, epoch)
            tb_writer.add_scalar(tags[4], cla_val_acc, epoch)
            tb_writer.add_scalar(tags[5], loc_train_loss, epoch)
            tb_writer.add_scalar(tags[6], loc_train_acc, epoch)
            tb_writer.add_scalar(tags[7], loc_val_loss, epoch)
            tb_writer.add_scalar(tags[8], loc_val_acc, epoch)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    tb_writer.add_histogram(name + '_grad', param.grad, epoch)
                    tb_writer.add_histogram(name + '_data', param, epoch)

    if args.rank == 0:
        tb_writer.close()
        clean_up()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 训练超参数设置
    parser.add_argument('--num_workers', type=int, default=8, help='the largest number of workers for dataloader')
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=0.00001)

    # 模型权重设置
    parser.add_argument('--weights_path', type=str, default="./model/vit_base_patch16_224.pth", help='initial weights path')

    # 是否在checkpoint处继续训练
    parser.add_argument('--checkpoint_path', type=str, default="")

    # scheduler自动调整学习率的patience
    parser.add_argument('--patience', type=int, default=2)

    # patch预测0/1的阈值设定
    parser.add_argument('--patch_threshold', type=list, default=[0.5, 0.7, 0.3])

    # 是否冻结层
    parser.add_argument('--freeze_layers', type=bool, default=True)

    # 是否启用SyncBN
    parser.add_argument('--syncBN', type=bool, default=True)

    # DDP运行参数(默认不修改)
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    opt = parser.parse_args()
    main(opt)