import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  

class MentorNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        """
        MentorNet 网络：输入样本的损失、损失差值和 epoch，输出样本的权重 v
        :param input_dim: 输入维度，包含 (loss, lossdiff, epoch)
        :param hidden_dim: 隐藏层维度
        """
        super(MentorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).cuda()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).cuda()
        self.fc3 = nn.Linear(hidden_dim, 1).cuda()

    def forward(self, input_data):
        """
        前向传播
        :param input_data: [batch_size, input_dim]，包含 (loss, lossdiff, epoch)
        :return: 样本权重 v，值在 (0,1) 之间
        """
        input_data = input_data.to('cuda')
        x = F.relu(self.fc1(input_data))
        x = F.relu(self.fc2(x))
        v = torch.sigmoid(self.fc3(x))  # 限制 v 在 [0,1]
        return v

def mixup_data(x, y, alpha=0.2):
    """ 生成 Mixup 训练样本 """
    if isinstance(alpha, torch.Tensor):  # 确保 alpha 不是张量
        alpha = alpha.mean().item()  # 取均值转换成标量

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()  # 随机打乱 batch 内的样本

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y
