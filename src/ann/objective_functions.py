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
        self.ypred_softmax=None
        self.eps=1e-15

    def loss(self,ytrue,ypred):
        self.ytrue=ytrue
        self.ypred=ypred
        # Apply softmax to logits
        exp_logits=np.exp(ypred-np.max(ypred,axis=1,keepdims=True))
        self.ypred_softmax=exp_logits/np.sum(exp_logits,axis=1,keepdims=True)
        # Clip for numerical stability
        probs=np.clip(self.ypred_softmax,self.eps,1-self.eps)
        return -np.sum(ytrue*np.log(probs))/ytrue.shape[0]

    def derivative(self): #combined with softmax derivative for easier gradient
        return (self.ypred_softmax-self.ytrue)/self.ytrue.shape[0] 
    
