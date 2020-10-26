from math import *
import numpy as np
import scipy.stats as stats
import time
import pandas as pd
import gc
import numba
gc.disable()

_broken = 0.30 #自然损坏概率
_valve = 0.90 #判定是否为坏车的概率阈值
_minlat = 35.549969284159104 #最小纬度
_minlng = 116.940193389599 #最小经度
_maxlat = 35.6243685047049 #最大纬度
_maxlng = 117.072528405772 #最大经度
_vert = 0.0014879844109158568 #纬度区间长度的1/50
_horiz = 0.0026467003234600384 #经度区间长度的1/50

df = pd.read_csv('/Users/valarian/SJTU/SJTU/毕业论文/Data/data/单车数据/曲阜数据/5-26 to 6-13/qufu_526_613_utc.csv', \
                    names = ['bid', 'lat', 'lng', 'time'], usecols = [0, 2, 3, 4])
df = df[:10000]

class Bike():
    def __init__(self, datum, blockid):
        self.id = datum[0]
        self.lat = datum[1]
        self.lng = datum[2]
        self.time = datum[3]
        self.block = blockid #表示车在哪个位置的tuple
        self.number = 1 
        self.broken = stats.truncnorm(0-_broken, 1-_broken, _broken, 1).rvs(1)[0] #应该设置成期望为某个概率的某种分布
#         self.alpha = 3
#         self.p = np.random.beta(self.alpha,7)
        self.state = [0,1][self.broken<_valve] #1 indicates functioning, 0 indicates broken
        if self.state == 0:
            broke_list.append([self.id, self.time, self.block])
            
    def update(self, datum, blockid): #对应了已经被实例化的正常车发生使用事件
        self.lat = datum[1]
        self.lng = datum[2]
        self.time = datum[3]
        self.block = blockid #表示车在哪个位置的tuple
        self.number += 1
        self.broken = stats.truncnorm(0-_broken, 1-_broken, _broken, 1).rvs(1)[0]
#         self.alpha = 3
#         self.p = np.random.beta(self.alpha,7)
        self.state = [0,1][self.broken<_valve] #1 indicates functioning, 0 indicates broken
        if self.state == 0:
            broke_list.append([self.id, self.time, self.block])
 
    def update_p(self, N): #有车离开了，剩下的车进行概率更新
        self.p = N*(N-1)*self.broken/((N-1)*(N-1+self.broken))
        self.state = [0,1][self.broken<_valve]
        if self.state == 0:
            broke_list.append([self.id, self.time, self.block])
        
class Block():
    def __init__(self, leftpoint, rightpoint):
        self.llat = leftpoint[1]
        self.llng = leftpoint[0]
        self.rlat = rightpoint[1]
        self.rlng = rightpoint[0]
        self.N = 0
        self.bikes = {}
        
    def bikein(self, datum, block):
        self.N += 1
        self.bikes[datum[0]] = bike_dic[datum[0]]
        bike_dic[datum[0]].update(datum, block)
    def bikeout(self, bike):
        del self.bikes[bike.id]
        for key in self.bikes:
            self.bikes[key].update_p(self.N)
        self.N -= 1
#        update the possibilities of the bikes likely to be broken

# dictionaries for bikes and blocks
broke_list = []
bike_dic = {} #保存所有已经出现的自行车的字典
block_dic = {} #保存所有block的字典
for vert in range(51):
    for horiz in range(51):
        block_dic[(vert, horiz)] = Block((_minlng+_horiz*horiz, _minlat+_vert*vert),
                                         (_minlng+_horiz*(horiz+1), _minlat+_vert*(vert+1)))


#if __name__ == '__main__':
    #ids = set(qufu.bid)
@profile
@numba.jit
def run():
    df = pd.read_csv('/Users/valarian/SJTU/SJTU/毕业论文/Data/data/单车数据/曲阜数据/5-26 to 6-13/qufu_526_613_utc.csv', \
                    names = ['bid', 'lat', 'lng', 'time'], usecols = [0, 2, 3, 4])
    df = df[:10000]
    for line in df.index:
        datum = df.loc[line]
        block = (int((datum[2]-_minlng)//_horiz), int((datum[1]-_minlat)//_vert))
        if datum[0] in bike_dic: #这辆车已经出现过
            bike = bike_dic[datum[0]]
            #bikeout
            block_dic[bike.block].bikeout(bike)
            #bikein
            block_dic[block].bikein(datum, block)
        else:  #这辆车第一次出现
            #generate new instance
            bike = Bike(datum,block)
            #update block
            bike_dic[datum[0]] = bike
            block_dic[block].bikein(datum,block)
    
    df_result = pd.DataFrame(broke_list)
    df_result.to_csv('/Users/valarian/SJTU/SJTU/毕业论文/Data/result/brokenbikes_ofo.csv')
#     with open('/Users/valarian/SJTU/SJTU/毕业论文/Data/result/brokenbikes_0.9.csv', 'w') as fout:
#         for _ in broke_list:
#             fout.writelines('%s\n'%(_))
run()
gc.enable()