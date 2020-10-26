'''
create_date : 21 Sep 2020
created_by : Zhongyu Guo
email : guozhongyu@sjtu.edu.cn
Description: 
    This is a python version implementation of the Ranking and Selection method called 
    Fully Sequential Indifference Zone Method which was proposed by Kim and Nelson in 2001.
'''
import numpy as np

class KNRanking_and_Selection():
    def __init__(self, c, k, n0, ALPHA, DELTA, setting):
        self.c = c
        # constant
        self.ALPHA = ALPHA
        self.CL = 1 - ALPHA # confidence level
        self.DELTA = DELTA
        self.setting = setting
        
        self.init_k = k # number of settings
        self.init_n0 = n0
        self.init_Ita = self.get_ita(ALPHA, k, n0)
            
    def initialize(self):
        self.k = self.init_k 
        self.n0 = self.init_n0 
        self.Ita = self.init_Ita
        self.I = {} # keep a dict to identify the initial index of each row left in the X
        [self.I.update({_:_}) for _ in range(self.k)] 
        self.X = np.array([np.random.normal(loc=self.setting[i][0],scale=self.setting[i][1],size=self.n0) for i in range(self.k)]) # initialize the observations

        # calculate some parameters
        self.h2 = self.get_h2(self.c, self.Ita, self.n0)
        self.S2 = self.get_S2(self.X, self.n0, self.k)
        self.Nil = self.get_Nil(self.h2, self.S2, self.DELTA, self.k)
        self.Ni = self.get_Ni(self.Nil)
        self.Xbar = self.get_Xbar(self.X, self.n0)
        
        self.counter = self.n0*self.k # record the total observations used
    
    def initial_check(self): # check if the algorithm stops before screening
        if self.n0 > max(self.Ni): 
            # return the one with highest sample mean
            t  = np.argmax(self.Xbar)
            self.I = {0:t}
            return 1
        else: 
            self.r = self.n0
            return 0
        
    # calc ita
    def get_ita(self, alpha, k , n0):
        return 0.5 * ((2*alpha / (k-1)) ** (-2/(n0-1))-1)
    # calc h2
    def get_h2(self, c, ita, n0):
        return 2 * c * ita * (n0-1)
    # calc S2il
    def get_S2(self, X, n0, k):
        X = X[:,:n0]
        return [[np.var(X[i]-X[l], ddof=1) for l in range(k)] for i in range(k)]
    # calc Nil
    def get_Nil(self, h2, s2, delta, k):
        return [[int(h2 * s2[i][l] / delta ** 2) for l in range(k)] for i in range(k)]
    # calc Ni
    def get_Ni(self, Nil):
        return [max(_) for _ in Nil]
    # calc Xbar
    def get_Xbar(self, X, r):
        return [np.mean(_) for _ in X[:,:r]]
    # calc Wil
    def get_Wil(self, s2, h2, delta, c, r, I_):
        return [[max(0, delta/2/c/r*(h2*s2[i][l]/delta**2 - r)) for l in I_.values()] for i in I_.values()]
    # update X
    def add_X(self, X, addon):
        #[np.append(X[i],v) for i,v in enumerate(addon)]
        return np.c_[X, addon]
    # update r
    def add_r(self, r):
        return r+1
    # delete i from I
    def update_I(self, I, pop):
        new_I = {}
        [new_I.update({n:v}) for n, v in enumerate(set(I.values())-set(pop))]
        return new_I
    def update_X(self, X, pop):
        new_X = X
        new_X = np.delete(new_X, pop, axis=0)
        return new_X
    # check stop condition
    def check_stop(self, I):
        if len(I)==1: return 0
        else: return 1
    # check each I
    def screen(self, I, X, Xbar, Wil):
        pop_I, pop_X = [], []
        for i in I:
            if sum([Xbar[i]>=Xbar[l] - Wil[i][l] for l in I if i!=l]) == len(Wil[0])-1: continue
            else: 
                pop_I.append(I[i])
                pop_X.append(i)
        I = self.update_I(I, pop_I)
        X = self.update_X(X, pop_X)
        return I, X
    # screening
    def screening(self, I, X, Xbar, S2, h2, delta, c, r):
        Wil = self.get_Wil(S2, h2, delta, c, r, I)
        I, X = self.screen(I, X, Xbar, Wil)
        return I, X
    
    def loop_filter(self):
        while self.check_stop(self.I): 
            x_addon = [np.random.normal(loc=self.setting[i][0],scale=self.setting[i][1],size=1) for i in self.I.values()] # add obervation to instances remained
            self.counter += len(x_addon)
            self.X, self.r = self.add_X(self.X, x_addon), self.add_r(self.r)
            self.Xbar = self.get_Xbar(self.X, self.r)
            if self.r == 1 + max([self.Ni[i] for i in self.I.values()]):
                self.I = {0:self.I[np.argmax(self.Xbar)]}
                break
            else:
                self.I, self.X = self.screening(self.I, self.X, self.Xbar, self.S2, self.h2, self.DELTA, self.c, self.r)
                self.k = len(self.I)

    def main(self):
        self.initialize()
        if not self.initial_check():
            self.loop_filter()
        value = [_ for _ in self.I.values()][0]
        return value, self.counter
        
        