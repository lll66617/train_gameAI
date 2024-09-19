import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#创建一个字体对象，使用名为'arial.ttf'的字体文件，并设置字体大小为25像素。
#创建了字体对象后，你就可以使用它来渲染文本，以便在Pygame中显示文本内容。
#font = pygame.font.SysFont('arial', 25)

#定义方向
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')#每一节点都是一个x，y坐标

# rgb colors，颜色信息
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN = (0, 255, 0)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20#一个小块就是20*20像素
SPEED = 10

class SnakeGameAI:

    #先定义画布的高和宽并将其渲染出来
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state，初始方向是朝右
        self.direction = Direction.RIGHT

        #蛇从中间出生，蛇的信息用list存储
        self.head = Point(self.w/2-10*BLOCK_SIZE, self.h/2)
        self.snake = [self.head,
                      Point(BLOCK_SIZE, BLOCK_SIZE)]

        #分数
        self.score = 0
        self.food = None
        self._place_block()
        self._place_food()

        self.frame_iteration = 0
        #记录回合数
        self.count=0

    #食物随机生成
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or self.food in self.block_list:
            self._place_food()

    #生成障碍物
    def _place_block(self):
        self.block_list=[]
        for i in range(50):
            x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            self.block=Point(x,y)
            if self.block in self.snake:
                i-=1
                continue
            self.block_list.append(self.block)

    def play_step(self, action):
        self.count+=1
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        #self.frame_iteration > 100*len(self.snake)是为了防止游戏陷入死循环
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        #如果进了死胡同扣分！
        x=self.head.x
        y=self.head.y
        pt1=Point(x-BLOCK_SIZE,y)
        pt2=Point(x+BLOCK_SIZE,y)
        pt3=Point(x,y-BLOCK_SIZE)
        pt4=Point(x,y+BLOCK_SIZE)
        if self.is_collision(pt1) and self.is_collision(pt2) and self.is_collision(pt3) and self.is_collision(pt4):
            print("你上当啦")
            reward = -10
            return reward, game_over, self.score

        # 长度增加
        if self.count%5==0:
            self.score += 1
            reward = 10
            #self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    #是否发生碰撞
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:#不包含蛇头
            return True
        #撞到障碍物
        if pt in self.block_list:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            #使用rect函数画出贪吃蛇
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        #食物
        #pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        #障碍物
        brown = (139, 69, 19)
        for pt in self.block_list:
            pygame.draw.rect(self.display,brown,pygame.Rect(pt.x,pt.y,BLOCK_SIZE,BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    #移动函数
    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)