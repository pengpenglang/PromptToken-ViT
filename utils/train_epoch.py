import os
import torch
from tqdm import tqdm
import torch.distributed as dist
from utils.draw import forecast_patch


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    # 设置命令记录是否分布式
    args.distributed = True

    # 设置当前进程绑定到gpu
    torch.cuda.set_device(args.gpu)

    # 设置分布式通信后端
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)

    # 初始化进程组
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # 阻塞等待所有进程都到达这里
    dist.barrier()


def is_main_process():
    return dist.get_rank() == 0


def reduce_value(value, average=True):
    if not dist.is_initialized():
        return value

    with torch.no_grad():
        # 阻塞所有进程，直到所有进程都执行了这个函数
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if average:
            value /= dist.get_world_size()
    return value


def clean_up():
    if dist.is_initialized():
        dist.destroy_process_group()


def cla_train_one_epoch(model, optimizer, loss_function, train_loader, device, epoch):
    model.train()
    mean_loss = torch.zeros(1).to(device)
    mean_acc = 0
    if is_main_process():
        train_loader = tqdm(train_loader, ncols=150)

    for step, (images, labels, _, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.shape[0]
        outputs = model(images)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        # 想要显示的是平均loss
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
        pred = torch.max(outputs, dim=1)[1]
        pred_num = torch.eq(pred, labels).sum().float()
        pred_num = reduce_value(pred_num, average=True)
        mean_acc = (mean_acc * step + pred_num / batch_size) / (step + 1)
        if is_main_process():
            train_loader.set_description("[cla_train {}] loss: {:.4f}, acc: {:.4f}".format(epoch, mean_loss.item(), mean_acc))
        optimizer.step()

    # 主进程传递梯度后每个进程异步进行梯度更新，因此此处需要同步
    torch.cuda.synchronize()
    return mean_loss.item(), mean_acc


@torch.no_grad()
def cla_evaluate(model, loss_function, val_loader, device, epoch):
    model.eval()
    mean_loss = torch.zeros(1).to(device)
    mean_acc = 0
    if is_main_process():
        val_loader = tqdm(val_loader, ncols=150)

    for step, (images, labels, _, _) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.shape[0]
        outputs = model(images)
        loss = loss_function(outputs, labels)
        # 想要显示的是平均loss
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
        pred = torch.max(outputs, dim=1)[1]
        pred_num = torch.eq(pred, labels).sum().float()
        pred_num = reduce_value(pred_num, average=True)
        mean_acc = (mean_acc * step + pred_num / batch_size) / (step + 1)
        if is_main_process():
            val_loader.set_description("[cla_valid {}] loss: {:.4f}, acc: {:.4f}".format(epoch, mean_loss.item(), mean_acc))

    # 主进程传递梯度后每个进程异步进行梯度更新因此此处需要同步
    torch.cuda.synchronize()
    return mean_loss.item(), mean_acc


def loc_train_one_epoch(model, optimizer, loss_function, train_loader, device, epoch, patch_threshold):
    model.train()
    mean_loss = torch.zeros(1).to(device)
    mean_acc = 0
    if is_main_process():
        train_loader = tqdm(train_loader, ncols=150)

    for step, (images, labels, sequences, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        sequences = sequences.to(device)
        batch_size = images.shape[0]
        ids = []
        for i in range(batch_size):
            tmp_id = torch.zeros(1000).to(device)
            tmp_id[labels[i]] = 1
            ids.append(tmp_id)
        ids = torch.stack(ids)
        outputs = model.module.localize(images, ids)
        if step % 10 == 0:
            forecast_patch(images[0], outputs[0], sequences[0], patch_threshold)
        loss = loss_function(outputs, sequences.to(torch.float))
        optimizer.zero_grad()
        loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
        # outputs和threshold会进行元素级别的比较
        pred = outputs > patch_threshold
        # 计算样本和labels向量数值为1的IoU
        in_num = (pred * sequences).sum(dim=1)
        out_num = ((pred + sequences) - (pred * sequences)).sum(dim=1)
        iou = (in_num / out_num).sum() / batch_size
        mean_iou = reduce_value(iou, average=True)
        mean_acc = (mean_acc * step + mean_iou) / (step + 1)
        if is_main_process():
            train_loader.set_description("[loc_train {}] loss: {:.4f}, acc: {:.4f}".format(epoch, mean_loss.item(), mean_acc))
        optimizer.step()

    torch.cuda.synchronize()
    return mean_loss.item(), mean_acc


@torch.no_grad()
def loc_evaluate(model, loss_function, val_loader, device, epoch, patch_threshold):
    model.eval()
    mean_loss = torch.zeros(1).to(device)
    mean_acc = 0
    if is_main_process():
        val_loader = tqdm(val_loader, ncols=150)

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
        ids = torch.stack(ids)
        outputs = model.module.localize(images, ids)
        if step % 10 == 0:
            forecast_patch(images[0], outputs[0], sequences[0], patch_threshold)
        loss = loss_function(outputs, sequences.to(torch.float))
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
        pred = outputs > patch_threshold[0]
        in_num = (pred * sequences).sum(dim=1)
        out_num = ((pred + sequences) - (pred * sequences)).sum(dim=1)
        iou = (in_num / out_num).sum() / batch_size
        mean_iou = reduce_value(iou, average=True)
        mean_acc = (mean_acc * step + mean_iou) / (step + 1)
        if is_main_process():
            val_loader.set_description("[loc_valid {}] loss: {:.4f}, acc: {:.4f}".format(epoch, mean_loss.item(), mean_acc))

    return mean_loss.item(), mean_acc


def train_one_epoch(model, optimizer, cla_loss_function, loc_loss_function, train_loader, device, epoch, patch_threshold):
    model.train()
    loc_mean_acc = 0
    cla_mean_acc = 0
    cla_mean_loss = torch.tensor(0).to(device)
    loc_mean_loss = torch.tensor(0).to(device)
    if is_main_process():
        train_loader = tqdm(train_loader, ncols=150)

    for step, (images, labels, sequences, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        sequences = sequences.to(device)
        batch_size = images.shape[0]
        optimizer.zero_grad()
        # --------------------- 分类 ----------------------
        cla_outputs = model(images)
        cla_loss = cla_loss_function(cla_outputs, labels)
        cla_loss.backward()
        cla_mean_loss = (cla_mean_loss * step + reduce_value(cla_loss, average=True).detach()) / (step + 1)
        cla_pred = torch.max(cla_outputs, dim=1)[1]
        cla_pred_num = torch.eq(cla_pred, labels).sum().float()
        cla_pred_num = reduce_value(cla_pred_num, average=True)
        cla_mean_acc = (cla_mean_acc * step + cla_pred_num / batch_size) / (step + 1)
        # --------------------- 检测 ----------------------
        ids = []
        for i in range(batch_size):
            tmp_id = torch.zeros(1000).to(device)
            tmp_id[labels[i]] = 1
            ids.append(tmp_id)
        ids = torch.stack(ids)
        loc_outputs = model.module.localize(images, ids)
        if step % 10 == 0:
            forecast_patch(images[0], loc_outputs[0], sequences[0], patch_threshold)
        loc_loss = loc_loss_function(loc_outputs, sequences.to(torch.float))
        loc_loss.backward()
        loc_mean_loss = (loc_mean_loss * step + reduce_value(loc_loss, average=True).detach()) / (step + 1)
        loc_pred = loc_outputs > patch_threshold[0]
        loc_in_num = (loc_pred * sequences).sum(dim=1)
        loc_out_num = ((loc_pred + sequences) - (loc_pred * sequences)).sum(dim=1)
        iou = (loc_in_num / loc_out_num).sum() / batch_size
        mean_iou = reduce_value(iou, average=True)
        loc_mean_acc = (loc_mean_acc * step + mean_iou) / (step + 1)

        optimizer.step()
        if is_main_process():
            train_loader.set_description("[train {}] cla_loss: {:.4f}, cla_acc: {:.4f}, loc_loss: {:.4f}, loc_acc: {:.4f}".format(epoch, cla_mean_loss.item(), cla_mean_acc, loc_mean_loss.item(),
                                                                                                                                  loc_mean_acc))

    torch.cuda.synchronize()
    return cla_mean_loss.item(), cla_mean_acc, loc_mean_loss.item(), loc_mean_acc


@torch.no_grad()
def evaluate(model, cla_loss_function, loc_loss_function, val_loader, device, epoch, patch_threshold):
    model.eval()
    loc_mean_acc = 0
    cla_mean_acc = 0
    cla_mean_loss = torch.zeros(1).to(device)
    loc_mean_loss = torch.zeros(1).to(device)
    if is_main_process():
        val_loader = tqdm(val_loader, ncols=150)

    for step, (images, labels, sequences, _) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        sequences = sequences.to(device)
        batch_size = images.shape[0]
        # --------------------- 分类 ----------------------
        cla_outputs = model(images)
        cla_loss = cla_loss_function(cla_outputs, labels)
        cla_mean_loss = (cla_mean_loss * step + reduce_value(cla_loss, average=True).detach()) / (step + 1)
        cla_pred = torch.max(cla_outputs, dim=1)[1]
        cla_pred_num = torch.eq(cla_pred, labels).sum().float()
        cla_pred_num = reduce_value(cla_pred_num, average=True)
        cla_mean_acc = (cla_mean_acc * step + cla_pred_num / batch_size) / (step + 1)
        # --------------------- 检测 ----------------------
        ids = []
        for i in range(batch_size):
            tmp_id = torch.zeros(1000).to(device)
            tmp_id[labels[i]] = 1
            ids.append(tmp_id)
        ids = torch.stack(ids)
        loc_outputs = model.module.localize(images, ids)
        if step % 10 == 0:
            forecast_patch(images[0], loc_outputs[0], sequences[0], patch_threshold)
        loc_loss = loc_loss_function(loc_outputs, sequences.to(torch.float))
        loc_mean_loss = (loc_mean_loss * step + reduce_value(loc_loss, average=True).detach()) / (step + 1)
        loc_pred = loc_outputs > patch_threshold[0]
        loc_in_num = (loc_pred * sequences).sum(dim=1)
        loc_out_num = ((loc_pred + sequences) - (loc_pred * sequences)).sum(dim=1)
        iou = (loc_in_num / loc_out_num).sum() / batch_size
        mean_iou = reduce_value(iou, average=True)
        loc_mean_acc = (loc_mean_acc * step + mean_iou) / (step + 1)

        if is_main_process():
            val_loader.set_description("[valid {}] cla_loss: {:.4f}, cla_acc: {:.4f}, loc_loss: {:.4f}, loc_acc: {:.4f}".format(epoch, cla_mean_loss.item(), cla_mean_acc, loc_mean_loss.item(),
                                                                                                                                loc_mean_acc))

    torch.cuda.synchronize()
    return cla_mean_loss.item(), cla_mean_acc, loc_mean_loss.item(), loc_mean_acc