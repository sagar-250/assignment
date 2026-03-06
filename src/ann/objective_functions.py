import numpy as np

class MSELoss:
    def __init__(self):
        self.ytrue=None
        self.ypred=None

    def loss(self,ytrue,ypred):
        self.ytrue=ytrue
        self.ypred=ypred
        return np.mean(np.square(ytrue-ypred))

    def derivative(self):
        n=self.ytrue.shape[0]
        return (2/n)*(self.ypred-self.ytrue)

class CrossEntropyLoss:
    def __init__(self):
        self.ytrue=None
        self.ypred=None
        self.eps=1e-15

    def loss(self,ytrue,ypred):
        self.ytrue=ytrue
        self.ypred=np.clip(ypred,self.eps,1-self.eps)
        return -np.sum(ytrue*np.log(self.ypred))/ytrue.shape[0]

    def derivative(self): #combined with softmax derivative for easier gradient
        return (self.ypred-self.ytrue)/self.ytrue.shape[0] 
    
