import numpy as np

class Layer:
    def __init__(self,in_dim,out_dim,act_class,weight_init='xavier'):
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.act=act_class()
        if weight_init=='xavier':
            std=np.sqrt(2/(in_dim+out_dim)) #using 2/(nin+nout) instead of 1/nin cuz its symmetric for forward and backward pass
            self.W=np.random.randn(in_dim,out_dim)*std
        else:
            self.W=np.random.randn(in_dim,out_dim)*0.01   
        self.b=np.zeros((1,out_dim))
        self.grad_W=None
        self.grad_b=None
        self.inp=None
        self.z=None

    def forward(self,x):
        self.inp=x
        self.z=np.dot(x,self.W)+self.b
        return self.act.forward(self.z)

    def backward(self,gout):
        gz=self.act.backward(gout)
        batch_size = self.inp.shape[0]
        self.grad_W=np.dot(self.inp.T,gz) / batch_size
        self.grad_b=np.sum(gz,axis=0,keepdims=True) / batch_size
        gin=np.dot(gz,self.W.T)
        return gin