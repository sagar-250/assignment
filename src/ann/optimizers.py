import numpy as np

class SGD:
    def __init__(self,lr=0.01):
        self.lr=lr
    def update(self,w,gw):
        return w-self.lr*gw

class Momentum:
    def __init__(self,lr=0.01,gamma=0.9):
        self.lr=lr
        self.gamma=gamma
        self.v=None
    def update(self,w,gw):
        if self.v is None:
            self.v=np.zeros_like(w)   
        self.v=self.gamma*self.v+self.lr*gw
        return w-self.v

class RMSprop:
    def __init__(self,lr=0.001,beta=0.9,epsilon=1e-8):
        self.lr=lr
        self.beta=beta
        self.eps=epsilon
        self.v=None
    def update(self,w,gw):
        if self.v is None:
            self.v=np.zeros_like(w)
        self.v=self.beta*self.v+(1-self.beta)*(gw**2)
        return w-self.lr*gw/(np.sqrt(self.v)+self.eps)

class NAG:
    def __init__(self,lr=0.01,gamma=0.9):
        self.lr=lr
        self.gamma=gamma
        self.v=None
    
    def update(self,w,gw):
        if self.v is None:
            self.v=np.zeros_like(w)
        
        self.v=self.gamma*self.v+self.lr*gw #using nesterov approximation like pytorch, calc gradient at current pos and apply momentum correction during update
        return w-(self.gamma*self.v+self.lr*gw)