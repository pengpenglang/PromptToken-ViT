import torch

class BCEFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        # pt = torch.sigmoid(predict)  # sigmoide获取概率
        pt = predict
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = -self.alpha * (1 - pt)**self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt**self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss