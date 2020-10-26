#windows version code to run on remote machine


import numpy as np
import scipy.stats as stats
#import time
import pandas as pd
import gc
from numba import jit
#import cProfile
from tqdm import tqdm
#gc.disable()

_broken = 0.30 #自然损坏概率
_valve = 0.90 #判定是否为坏车的概率阈值

#ofo范围
_minlat = 31.1098957062 #最小纬度
_minlng = 121.3264205349 #最小经度
_maxlat = 31.372089535900002 #最大纬度
_maxlng = 121.6624752258 #最大经度
#曲阜范围
# _minlat = 35.549969284159104 #最小纬度
# _minlng = 116.940193389599 #最小经度
# _maxlat = 35.6243685047049 #最大纬度
# _maxlng = 117.072528405772 #最大经度

_vert = (_maxlat-_minlat)/50 #纬度区间长度的1/50
_horiz = (_maxlng-_minlng)/50 #经度区间长度的1/50


class Bike():
    def __init__(self, datum, blockid, fo):
        self.id = datum.bid
        self.lat = datum.lat
        self.lng = datum.lng
        self.time = datum.time
        self.block = blockid #表示车在哪个位置的tuple
        self.fout = fo
        self.number = 1 
        self.broken = stats.truncnorm(0-_broken, 1-_broken, _broken, 1).rvs(1)[0] #应该设置成期望为某个概率的某种分布
#         self.alpha = 3
#         self.p = np.random.beta(self.alpha,7)
        self.state = [0,1][self.broken<_valve] #1 indicates functioning, 0 indicates broken
        if self.state == 0:
            self.fout.write('%s,%s,%s\n'%(self.id, self.time, self.block))
            #broke_list.append([self.id, self.time, self.block])
            
    def update(self, datum, blockid): #对应了已经被实例化的正常车发生使用事件
        self.lat = datum.lat
        self.lng = datum.lng
        self.time = datum.time
        self.block = blockid #表示车在哪个位置的tuple
        self.number += 1
        self.broken = stats.truncnorm(0-_broken, 1-_broken, _broken, 1).rvs(1)[0]
#         self.alpha = 3
#         self.p = np.random.beta(self.alpha,7)
        self.state = [0,1][self.broken<_valve] #1 indicates functioning, 0 indicates broken
        if self.state == 0:
            self.fout.write('%s,%s,%s\n'%(self.id, self.time, self.block))
            #broke_list.append([self.id, self.time, self.block])
 
    def update_p(self, N): #有车离开了，剩下的车进行概率更新
        self.p = N*(N-1)*self.broken/((N-1)*(N-1+self.broken))
        self.state = [0,1][self.broken<_valve]
        if self.state == 0:
            self.fout.write('%s,%s,%s\n'%(self.id, self.time, self.block))
            #broke_list.append([self.id, self.time, self.block])
        
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
        self.bikes[datum.bid] = bike_dic[datum.bid]
        bike_dic[datum.bid].update(datum, block)
    def bikeout(self, bike):
        del self.bikes[bike.id]
        for key in self.bikes:
            self.bikes[key].update_p(self.N)
        self.N -= 1
#        update the possibilities of the bikes likely to be broken

# dictionaries for bikes and blocks
#broke_list = []
bike_dic = {} #保存所有已经出现的自行车的字典
block_dic = {} #保存所有block的字典
for vert in range(51):
    for horiz in range(51):
        block_dic[(vert, horiz)] = Block((_minlng+_horiz*horiz, _minlat+_vert*vert),
                                         (_minlng+_horiz*(horiz+1), _minlat+_vert*(vert+1)))


#if __name__ == '__main__':
    #ids = set(qufu.bid)
#@jit
def run():
    df = pd.read_csv('C:/Rebalancing/data/单车数据/ofo_car_6-19_7-3/ofo_car_73_utc.csv', \
                    names = ['bid', 'lat', 'lng', 'time'], usecols = [0, 2, 3, 4])
    with open('C:/Rebalancing/data/单车数据/ofo_car_6-19_7-3/ofo_car_73_utc_broken.csv', 'w') as fout:
#     df = pd.read_csv('/Users/valarian/SJTU/SJTU/毕业论文/Data/data/单车数据/曲阜数据/5-26 to 6-13/qufu_526_613_utc.csv', \
#                     names = ['bid', 'lat', 'lng', 'time'], usecols = [0, 2, 3, 4])
#     with open('/Users/valarian/SJTU/SJTU/毕业论文/Data/result/broken_test.csv', 'w') as fout:
        for index, row in tqdm(df.iterrows()):
            #datum = df.loc[line]
            block = (int((row.lng-_minlng)//_horiz), int((row.lat-_minlat)//_vert))
            if row.bid in bike_dic: #这辆车已经出现过
                bike = bike_dic[row.bid]
                #bikeout
                block_dic[bike.block].bikeout(bike)
                #bikein
                block_dic[block].bikein(row, block)
            else:  #这辆车第一次出现
                #generate new instance
                bike = Bike(row,block, fout)
                #update block
                bike_dic[row.bid] = bike
                block_dic[block].bikein(row,block)
    
    #df_result = pd.DataFrame(broke_list)
    #df_result.to_csv('/Users/valarian/SJTU/SJTU/毕业论文/Data/result/brokenbikes_ofo.csv')
#     with open('/Users/valarian/SJTU/SJTU/毕业论文/Data/result/brokenbikes_0.9.csv', 'w') as fout:
#         for _ in broke_list:
#             fout.writelines('%s\n'%(_))

if __name__ == '__main__':
    run()

#gc.enable()