import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000 # 记录100 000次实验的state, action, reward, next_state, done情况 为何要储存这么多呢 要训练！
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() 记录训练的每一轮信息，经验回放区
        self.model = Linear_QNet(8, 256, 3)#模型，3层BP神经网路
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)#训练，Q-learning训练器

    #让模型拿到这11个值
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)#头左边坐标
        point_r = Point(head.x + 20, head.y)#头右边坐标
        point_u = Point(head.x, head.y - 20)#头上面坐标
        point_d = Point(head.x, head.y + 20)#头下面坐标
        
        dir_l = game.direction == Direction.LEFT#朝向是否为左
        dir_r = game.direction == Direction.RIGHT#朝向是否为右
        dir_u = game.direction == Direction.UP#朝向是否为上
        dir_d = game.direction == Direction.DOWN#朝向是否为下

        state = [
            #通过朝向和下一步判断是否危险
            # Danger straight,前方是否危险，朝向和下一步一致
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right，右边是否危险
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left，左边是否危险
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            #因为后面一定危险所以不用判断
            
            # Move direction，移动方向
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            #长度是否增加
            game.count==4

            # Food location 
            #game.food.x < game.head.x,  # food left
            #game.food.x > game.head.x,  # food right
            #game.food.y < game.head.y,  # food up
            #game.food.y > game.head.y  # food down
            ]

        #print(state)
        return np.array(state, dtype=int)

    #remember就是把每步的信息append起来
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    #依据remember中的信息进行强化训练
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
            # 从历史的 state中随机选择BATCH_SIZE数目的state训练
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games#探索率，随着游戏局数的增加而减小
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)#将状态转为模型能识别的类型
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() # 模型输出三个值 取最大的 作为移动方向(前进 ← →)
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []  # 存储每局游戏的分数
    plot_mean_scores = []  # 存储每局游戏分数的平均值
    total_score = 0  # 总分数
    record = 0  # 最高分记录
    agent = Agent()  # 创建智能体
    game = SnakeGameAI()  # 创建贪吃蛇游戏实例
    while True:
         # 获取当前状态
        state_old = agent.get_state(game)

        # 获取移动方向
        final_move = agent.get_action(state_old)

        # 执行移动并获取新状态、奖励和游戏是否结束
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 对短期记忆进行训练
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 记忆当前状态、移动方向、奖励、新状态和游戏结束标志
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:  # 如果游戏结束（撞墙或者碰到自身）
            # 训练长期记忆，并绘制结果图
            game.reset()  # 重置游戏
            agent.n_games += 1  # 游戏局数加一
            agent.train_long_memory()  # 训练长期记忆

            if score > record:  # 如果当前分数高于最高记录
                record = score  # 更新最高记录
                print("save")
                agent.model.save()  # 保存模型权重

            print('Game', agent.n_games, 'Score', score, 'Record:', record)  # 打印游戏局数、分数和最高记录

            plot_scores.append(score)  # 将当前分数添加到分数列表中
            total_score += score  # 更新总分数
            mean_score = total_score / agent.n_games  # 计算平均分数
            plot_mean_scores.append(mean_score)  # 将平均分数添加到平均分数列表中
            plot(plot_scores, plot_mean_scores)  # 绘制分数图和平均分数图



if __name__ == '__main__':
    agent = Agent()  # 创建智能体
    
    game = SnakeGameAI()  # 创建贪吃蛇游戏实例

    while True:
         # 获取当前状态
        state_old = agent.get_state(game)
        agent.model.load_state_dict(torch.load('model\\model.pth'))#加载模型文件
        # 获取移动方向
        final_move = [0,0,0]
        state0 = torch.tensor(state_old, dtype=torch.float)
        prediction = agent.model(state0)
        move = torch.argmax(prediction).item() # 模型输出三个值 取最大的 作为移动方向(前进 ← →)
        final_move[move] = 1
        # 执行移动并获取新状态、奖励和游戏是否结束
        reward, done, score = game.play_step(final_move)
        
        if done:
            break
    #train()
    #导出模型
    
    #dummy_input = torch.randn(8)
    #bool_input = dummy_input==a
    #state_old = agent.get_state(game)
    #print(state_old)
    #state0 = torch.tensor(state_old, dtype=torch.float)
    #agent.model.load_state_dict(torch.load('model\\model.pth'))#加载模型文件
    #print(state0)
    #prediction = agent.model(state0)
    #print(prediction)
    #print(torch.argmax(prediction).item())
    #torch.onnx.export(agent.model, dummy_input, "model.onnx")