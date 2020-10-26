'''
    Simulate the sharing bikes running between blocks

    System parameters definition:
                 T : system  clock by minutes
             event : events in the system
     _trans_matrix : possibility matrix of normal bikes transisting from one block to another
              _miu : arriving rate for each blocks
           _lambda : parameters of the distribution for the time for a bike riden to another
       _life_clock : life duration for a just repaired bike
    _time_interval : dicision time interval
           _blocks : list for all the blocks
'''

import numpy as np
import pandas as pd
from numba import jit

class Block():
    def __init__(self, name, event, bikes):
        self.name = name
        self.normal_bike = bikes  # 好车FIFO
        self.arriving_bike = {}  # 正在来到这个block的车的list
        self.customer_arriving = np.random.exponential(_miu[self.name - 1], 1)[0]
        self.add_event(event, self.customer_arriving, self.name, 1, -1)
        self.broken_bike = {}  # 坏车一视同仁
        self.served, self.lost = 0, 0

    def add_event(self, event, time, dest, typeofa, bike):
        if event.time == 0:
            event.next = Event(self.customer_arriving, self.name, 1, -1)
        elif time <= event.time:
            event.next = Event(event.time, event.obj, event.type, event.bike)
            event.time, event.obj, event.type, event.bike = time, dest, typeofa, bike
        else:
            temp = event
            while temp:
                if temp.next == None:
                    temp.next = Event(time, dest, typeofa, bike)
                    break
                elif time <= temp.next.time:
                    _ = temp.next
                    temp.next = Event(time, dest, typeofa, bike)
                    temp.next.next = _
                    break
                else:
                    temp = temp.next

    def execute(self, event, time, activity, bike):
        if activity == 1:
            if self.normal_bike:
                # bike和block served += 1
                self.normal_bike[0].served += 1
                self.served += 1
                # 生成下一个到达，并添加至Event类中
                self.customer_arriving = np.random.exponential(_miu[self.name - 1], 1)[0] + time
                #print('customer arriving: ' + str(self.customer_arriving))
                self.add_event(event, self.customer_arriving, self.name, 1, -1)
                # 转移到什么地方去，调用对方instance添加一个arriving事件
                destination = [1, 2][2 - self.name]
                t = time + np.random.exponential(_lambda[self.name - 1], 1)[0]
                #print(t)
                self.add_event(event, t, destination, 2, self.normal_bike[0])
                # 车出库，从list中删掉
                blocks[destination].arriving_bike[self.normal_bike[0].id] = self.normal_bike[0]
                del self.normal_bike[0]
            else:
                # lose这个顾客
                self.lost += 1
                # 生成下一个到达事件
                self.customer_arriving = np.random.exponential(_miu[self.name - 1], 1)[0] + time
                self.add_event(event, self.customer_arriving, self.name, 1, -1)
        else:
            # 那就是有一辆车进入
            # normal list要添加
            #print(event.time, event.obj, event.type, event.bike)
            self.normal_bike.append(bike)
            del self.arriving_bike[bike.id]


class Bike():
    def __init__(self, name):
        self.id = name
        self.state = True  # True for normal, False for broken
        # self.t = T #上一次处理这个bike的时间
        self.served, self.lost = 0, 0

    def deteriorate(self):  # 每次维修完成后按一定参数的指数分布生成其使用寿命，也即下次损坏的的时间
        self.deat_time = T + np.random.exponential(_life_clock, 1)[0]


class Event():
    def __init__(self, time, objct, activity, bike):
        '''
        type:1 for customer arriving, 2 for bike arriving
        '''
        self.time = time
        self.obj = objct
        self.type = activity
        self.bike = bike
        self.next = None


if __name__ == '__main__':

    '''
    'N: number of bikes
    'B: number of blocks
    '''
    T = 0  # 系统时钟
    event = Event(0, -1, -1, -1)
    _trans_matrix = [1, 1]  # 转移概率矩阵
    _miu = [2, 2]  # 每个block中乘客的到达速率
    _lambda = [5, 5]  # block之间骑行时间参数矩阵
    _life_clock = 2000  # 刚修好的一辆车寿命服从的指数分布的参数
    _time_interval = 30
    blocks = {}

    b1 = [Bike(1), Bike(2), Bike(3)]
    blocks[1] = Block(1, event, b1)
    event = event.next
    b2 = [Bike(4), Bike(5), Bike(6)]
    blocks[2] = Block(2, event, b2)

    N, B = 10, 2

    _periods = 10000

    for i in range(_periods):
        T += _time_interval
        while event.time <= T:

            blocks[event.obj].execute(event, event.time, event.type, event.bike)
            # temp = event
            # while event:
            #     print(event.time, event.obj, event.type, event.bike)
            #     event = event.next
            # print('\n\n')
            # event = temp
            event = event.next
            if not event: break
    print(blocks[1].served, blocks[1].lost, blocks[2].served, blocks[2].lost)