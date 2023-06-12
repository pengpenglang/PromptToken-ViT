import torch


def cla_mAP(outputs, labels):
    '''
    分类mAP计算函数
    计算方法参考:https://blog.csdn.net/liubo187/article/details/77406654
    
    Parameters:
        outputs - 模型预测输出,shape为[batch_size, num_classes]
        labls - 样本标签,shape为[batch_size, num_classes]
    
    Returns:
        res - mAP值
    '''
    _, indices = torch.sort(outputs, dim=0, descending=True)
    res = 0
    for i in range(outputs.shape[1]):
        pos_cnt = 0.000000001
        tot_cnt = 0
        precision = 0
        for j in indices[:, i]:
            target = labels[j][i]
            if target == 1:
                pos_cnt += 1
            tot_cnt += 1
            if target == 1:
                precision += pos_cnt / tot_cnt
        precision /= pos_cnt
        res += precision
    return res / outputs.shape[1]


if __name__ == '__main__':
    outputs = torch.tensor([[0.1, 0.2], [0.9, 0.5]])
    labels = torch.tensor([[0, 0], [1, 1]])
    print(cla_mAP(outputs, labels))