import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  

class MentorNet(nn.Module):
    def __init__(self, label_embedding_size=8, epoch_embedding_size=6, 
                 num_label_embedding=751, num_fc_nodes=100):
        super(MentorNet, self).__init__()

        # 标签和 Epoch Embedding 层
        self.label_embedding = nn.Embedding(num_label_embedding, label_embedding_size)
        self.epoch_embedding = nn.Embedding(100, epoch_embedding_size)

        # 双向 LSTM 层
        self.lstm = nn.LSTM(input_size=2, hidden_size=1, 
                            num_layers=1, batch_first=True, bidirectional=True)

        # 全连接层
        self.fc1 = nn.Linear(label_embedding_size + epoch_embedding_size + 4, num_fc_nodes)
        self.fc2 = nn.Linear(num_fc_nodes, 1)

    def forward(self, input_features):
        """
        input_features: [batch_size, 4] -> (loss, lossdiff, labels, epoch)
        """
        losses = input_features[:, 0].unsqueeze(-1)  # [batch_size, 1]
        loss_diffs = input_features[:, 1].unsqueeze(-1)  # [batch_size, 1]
        labels = input_features[:, 2].long()  # 转换为整数索引
        epochs = input_features[:, 3].long().clamp(max=99)  # 限制最大值 99

        # 标签和 epoch 的 embedding
        label_inputs = self.label_embedding(labels)  # [batch_size, label_embedding_size]
        epoch_inputs = self.epoch_embedding(epochs)  # [batch_size, epoch_embedding_size]

        # LSTM 计算
        lstm_inputs = torch.cat([losses, loss_diffs], dim=-1).unsqueeze(1)  # [batch_size, 1, 2]
        _, (hidden_fw, hidden_bw) = self.lstm(lstm_inputs)  # 双向 LSTM 输出

        # 拼接隐藏状态
        h = torch.cat([hidden_fw.permute(1, 0, 2).view(32, -1), hidden_bw.permute(1, 0, 2).view(32, -1)], dim=-1)  # [batch_size, 2]

        # 拼接特征
        feat = torch.cat([label_inputs, epoch_inputs, h], dim=-1)  # [batch_size, label_size + epoch_size + 2]

        # 全连接层计算权重 v
        fc1_out = torch.tanh(self.fc1(feat))  # [batch_size, num_fc_nodes]
        v = self.fc2(fc1_out)  # [batch_size, 1]
        
        return v
    
def sigmoid(x):
    # if torch.isnan(x).any() or torch.isinf(x).any():  
    #     print("Detected NaN or Inf in x!")
    #     print(x)
    #     exit()
    x = x.detach().cpu().numpy()  # 先分离计算图，再移到 CPU 并转换为 numpy 数组
    return 1 / (1 + np.exp(-x)) 

# class MentorNet(nn.Module):
#     def __init__(self, input_dim=3, hidden_dim=32):
#         """
#         MentorNet 网络：输入样本的损失、损失差值和 epoch，输出样本的权重 v
#         :param input_dim: 输入维度，包含 (loss, lossdiff, epoch)
#         :param hidden_dim: 隐藏层维度
#         """
#         super(MentorNet, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim).cuda()
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim).cuda()
#         self.fc3 = nn.Linear(hidden_dim, 1).cuda()

#     def forward(self, input_data):
#         """
#         前向传播
#         :param input_data: [batch_size, input_dim]，包含 (loss, lossdiff, epoch)
#         :return: 样本权重 v，值在 (0,1) 之间
#         """
#         input_data = input_data.to('cuda')
#         x = F.relu(self.fc1(input_data))
#         x = F.relu(self.fc2(x))
#         v = torch.sigmoid(self.fc3(x))  # 限制 v 在 [0,1]
#         return v

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
