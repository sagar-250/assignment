import numpy as np

class Identity:
    """Linear activation - returns input as-is (for output layer to get logits)"""
    def __init__(self):
        pass
    def forward(self, x):
        return x
    def backward(self, gout):
        return gout

class Sigmoid:
    def __init__(self):
        self.out=None
    def forward(self,x):
        self.out=1/(1+np.exp(-x))
        return self.out
    def backward(self,gout):
        return gout*(self.out*(1-self.out))
    
class ReLU:
    def __init__(self):
        self.x=None
    def forward(self,x):
        self.x=x
        return np.maximum(0,x)
    def backward(self,gout):
        gin=gout.copy()
        gin[self.x<=0]=0
        return gin
    
class Tanh:
    def __init__(self):
        self.out=None
    def forward(self,x):
        self.out=np.tanh(x)
        return self.out
    def backward(self,gout):
        return gout*(1-self.out**2)

class Softmax:
    def __init__(self):
        self.out=None
    def forward(self,x):
        ex=np.exp(x-np.max(x,axis=1,keepdims=True))
        self.out=ex/np.sum(ex,axis=1,keepdims=True)
        return self.out
    def backward(self,gout): #havent written derivative here cuz we combine it with cross entropy for easier gradient
        return gout