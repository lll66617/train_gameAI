import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        #三层BP神经网络，输入层，隐藏层，输出层
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # 定义网络的前向传播
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    #把模型的权重保存下来，在model文件夹下
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

# Q-learning训练器类
class QTrainer:
    def __init__(self, model, lr, gamma):
        # 学习率和折扣因子
        self.lr = lr
        self.gamma = gamma
        # 待训练的模型
        self.model = model
        #优化器使用了PyTorch库中的Adam优化器（Adam optimizer）。
        #Adam是一种常用的自适应学习率优化算法，通常用于训练神经网络模型。
        #optim.Adam表示创建了一个Adam优化器对象。
        #model.parameters()指定了优化器需要优化的模型参数。
        #lr=self.lr指定了学习率（learning rate），即优化器在更新模型参数时使用的步长或者说学习率大小。
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        #损失函数的度量
        #使用了PyTorch库中的均方误差损失函数
        #均方误差损失函数是一种常用的损失函数，通常用于回归问题。对于每个样本，
        #均方误差损失函数计算模型预测值与真实值之间的差异的平方，并将所有样本的平方差取平均作为最终的损失值。
        #优化器通过最小化损失函数的值来调整模型参数，使得模型的预测值更接近真实值。
        self.criterion = nn.MSELoss()

    # 单步训练，针对单个转换
    def train_step(self, state, action, reward, next_state, done):
        #将相应数据转换为适合pytorch使用的数据类型
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        # 如果状态是单个样本，则添加一个额外维度
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 使用当前状态预测的Q值
        pred = self.model(state)

        target = pred.clone()
        #循环遍历所有的样本，其中done是一个布尔型数组，表示每个样本是否为终止状态。
        for idx in range(len(done)):
            # 使用贝尔曼方程计算目标Q值
            Q_new = reward[idx]#将Q_new初始化为当前状态下的即时奖励。
            if not done[idx]: #检查当前样本是否为非终止状态。如果是终止状态，则不需要计算未来奖励。
                #如果当前样本不是终止状态，则使用该公式来计算未来奖励。
                #是折扣因子，用于平衡当前奖励和未来奖励的重要性
                #用于计算下一个状态对应的最大Q值，即预测下一个状态中最有利的行动所对应的Q值。
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            #将Q_new赋值给target张量中对应动作的位置，以便后续计算损失和更新模型参数
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        #这行代码将优化器中的梯度清零。
        #在PyTorch中，每次反向传播之前，需要手动将之前的梯度清零，否则梯度会累加，导致错误的参数更新。
        self.optimizer.zero_grad()
        #这行代码计算了模型预测值 pred 和目标值 target 之间的损失。
        #self.criterion 是之前创建的损失函数对象（通常是均方误差损失函数 nn.MSELoss()）。
        loss = self.criterion(target, pred)
        #计算损失函数对模型参数的梯度。
        loss.backward()

        #这行代码使用优化器来更新模型参数。优化器根据之前计算得到的参数梯度，
        #以及指定的学习率等参数，来更新模型中的参数。
        #这个过程通常是使用梯度下降法或其变种来最小化损失函数的过程，从而使模型的预测值更接近真实值。
        self.optimizer.step()



