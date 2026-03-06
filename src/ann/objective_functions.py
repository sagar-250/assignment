import numpy as np


def _to_one_hot(y, num_classes):
    oh=np.zeros((y.shape[0],num_classes))
    oh[np.arange(y.shape[0]),y.astype(int)]=1
    return oh

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
        # Accept both one-hot (N,C) and class-index (N,) labels
        if ytrue.ndim==1:
            ytrue=_to_one_hot(ytrue, ypred.shape[1])
        self.ytrue=ytrue
        self.ypred=ypred
        # Apply softmax to logits
        exp_logits=np.exp(ypred-np.max(ypred,axis=1,keepdims=True))
        ypred_softmax=exp_logits/np.sum(exp_logits,axis=1,keepdims=True)
        # Clip for numerical stability
        probs=np.clip(ypred_softmax,self.eps,1-self.eps)
        return -np.sum(ytrue*np.log(probs))/ytrue.shape[0]

    def derivative(self): #combined with softmax derivative for easier gradient
        # Always recompute softmax from current ypred to avoid stale state
        exp_logits=np.exp(self.ypred-np.max(self.ypred,axis=1,keepdims=True))
        ypred_softmax=exp_logits/np.sum(exp_logits,axis=1,keepdims=True)
        return (ypred_softmax-self.ytrue)/self.ytrue.shape[0]

