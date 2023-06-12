import sys

sys.path.insert(0, '')

import cv2
import torch
import numpy as np


def patch_to_box(output):
    '''
    将预测的patch转换为box
    
    Parameters:
        output - 模型预测的patch结果,一维张量
    
    Returns:
        box - 坐标列表
    '''
    pmap = np.zeros((224, 224, 3))
    for i in range(14):
        for j in range(14):
            if output[i * 14 + j] == 1:
                pmap[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16, :] = 255
    pmap = pmap.astype(np.uint8)
    pmap = cv2.cvtColor(pmap, cv2.COLOR_BGR2GRAY)

    _, _, stats, centroids = cv2.connectedComponentsWithStats(pmap, connectivity=8)
    stats = np.delete(stats, 0, 0)  # 删除背景
    centroids = np.delete(centroids, 0, 0)  # 删除背景
    box = []
    # 按照面积降序排列
    stats = stats[np.argsort(stats[:, 4])[::-1]]
    for prop in stats:
        x, y, w, h, _ = prop
        box.extend([x, y, x + w, y + h])
    return box


def cal_box(forecast, box):
    '''
    统计box预测正确与错误的数量
    
    Parameters:
        forecast - 预测的box坐标列表
        box - gt box坐标列表
    
    Returns:
        ac - 预测正确的数量
        wa - 预测错误的数量
    '''

    def cal_iou(box1, box2):
        '''
        计算两个box的iou

        Parameters:
            box1 - [fxmin, fymin, fxmax, fymax]
            box2 - [xmin, ymin, xmax, ymax]
        '''
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
        w = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
        h = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
        inter = w * h
        union = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - inter
        return inter / union

    ac = 0
    for i in range(0, len(forecast), 4):
        fxmin, fymin, fxmax, fymax = forecast[i:i + 4]
        for j in range(0, len(box), 4):
            xmin, ymin, xmax, ymax = box[j:j + 4]
            iou = cal_iou([fxmin, fymin, fxmax, fymax], [xmin, ymin, xmax, ymax])
            if iou > 0.5:
                ac += 1
                break
    wa = len(forecast) // 4 - ac
    return ac, wa