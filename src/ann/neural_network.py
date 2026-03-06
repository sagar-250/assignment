"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
import copy

from .neural_layer import Layer
from .activations import Sigmoid, ReLU, Tanh, Softmax
from .objective_functions import CrossEntropyLoss, MSELoss


acts={'relu':ReLU,'sigmoid':Sigmoid,'tanh':Tanh,'softmax':Softmax}

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self,cli_args):
        self.in_size=cli_args.input_size
        self.hid_sizes=cli_args.hidden_size
        self.out_size=cli_args.output_size
        self.act=cli_args.activation
        self.w_init=cli_args.weight_init
        self.lyrs=[]
        self.loss_fn=CrossEntropyLoss() if cli_args.loss=='cross_entropy' else MSELoss()
        self._build_network()
        self.layers=self.lyrs

    def _build_network(self):
        act_cls=acts[self.act.lower()]
        sizes=[self.in_size]+self.hid_sizes+[self.out_size]
        for i in range(len(sizes)-1):
            a=act_cls if i<len(sizes)-2 else Softmax
            self.lyrs.append(Layer(sizes[i],sizes[i+1],a,weight_init=self.w_init))

    def forward(self,X):
        out=X
        for lyr in self.lyrs:
            out=lyr.forward(out)
        return out

    def backward(self,y_true,y_pred):
        g=self.loss_fn.derivative()
        for lyr in reversed(self.lyrs):
            g=lyr.backward(g)

        gw_list=[lyr.grad_W for lyr in reversed(self.lyrs)]
        gb_list=[lyr.grad_b for lyr in reversed(self.lyrs)]

        self.grad_W=np.empty(len(gw_list),dtype=object) #making object arrays so numpy doesnt broadcast
        self.grad_b=np.empty(len(gb_list),dtype=object)
        for i,(gw,gb) in enumerate(zip(gw_list,gb_list)):
            self.grad_W[i]=gw
            self.grad_b[i]=gb

        return self.grad_W,self.grad_b

    def update_weights(self):
        for i,lyr in enumerate(self.lyrs):
            lyr.W=self.w_opts[i].update(lyr.W,lyr.grad_W)
            lyr.b=self.b_opts[i].update(lyr.b,lyr.grad_b)

    def train(self,X_train,y_train,optimizer,epochs=1,batch_size=32,xval=None,yval=None,callback=None,X_val=None,y_val=None,epoch_callback=None):
        if X_val is not None:
            xval=X_val
        if y_val is not None:
            yval=y_val
        if epoch_callback is not None:
            callback=epoch_callback
        xtr=X_train
        ytr=y_train
        opt=optimizer
        self.w_opts=[copy.deepcopy(opt) for _ in self.lyrs]
        self.b_opts=[copy.deepcopy(opt) for _ in self.lyrs]
        n=xtr.shape[0]
        for ep in range(epochs):
            idx=np.random.permutation(n)
            xs=xtr[idx]
            ys=ytr[idx]
            ep_loss=0
            nbatch=0
            for start in range(0,n,batch_size):
                xb=xs[start:start+batch_size]
                yb=ys[start:start+batch_size]
                ypred=self.forward(xb)
                ep_loss+=self.loss_fn.loss(yb,ypred)
                self.backward(yb,ypred)
                self.update_weights()
                nbatch+=1
            tr_loss=ep_loss/nbatch
            
            tr_pred=self.forward(xtr)
            tr_acc=np.mean(np.argmax(tr_pred,axis=1)==np.argmax(ytr,axis=1))
            tr_f1=compute_f1_score(ytr,tr_pred)
            
            metrics={'epoch':ep+1,'train_loss':tr_loss,'train_acc':tr_acc,'train_f1':tr_f1}
            if xval is not None and yval is not None:
                val_loss,val_acc,val_f1=self.evaluate(xval,yval)
                metrics['val_loss']=val_loss
                metrics['val_acc']=val_acc
                metrics['val_f1']=val_f1
                print(f"Epoch {ep+1}/{epochs} - train_loss: {tr_loss:.4f} | train_acc: {tr_acc:.4f} | train_f1: {tr_f1:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f}")
            else:
                print(f"Epoch {ep+1}/{epochs} - train_loss: {tr_loss:.4f} | train_acc: {tr_acc:.4f} | train_f1: {tr_f1:.4f}")
            if callback:
                callback(metrics)

    def evaluate(self,X,y):
        ypred=self.forward(X)
        loss=self.loss_fn.loss(y,ypred)
        acc=np.mean(np.argmax(ypred,axis=1)==np.argmax(y,axis=1))
        f1=compute_f1_score(y,ypred)
        return loss,acc,f1

    def get_weights(self):
        d={}
        for i,lyr in enumerate(self.lyrs):
            d[f"W{i}"]=lyr.W.copy()
            d[f"b{i}"]=lyr.b.copy()
        return d

    def set_weights(self,wdict):
        for i,lyr in enumerate(self.lyrs):
            wk=f"W{i}"
            bk=f"b{i}"
            if wk in wdict:
                lyr.W=wdict[wk].copy()
            if bk in wdict:
                lyr.b=wdict[bk].copy()
                
def compute_f1_score(ytrue,ypred,num_classes=10):
    if len(ytrue.shape)>1 and ytrue.shape[1]>1:
        ytrue=np.argmax(ytrue,axis=1)
    if len(ypred.shape)>1 and ypred.shape[1]>1:
        ypred=np.argmax(ypred,axis=1)
    
    scores=[]
    for cls in range(num_classes):
        tp=np.sum((ytrue==cls)&(ypred==cls))
        fp=np.sum((ytrue!=cls)&(ypred==cls))
        fn=np.sum((ytrue==cls)&(ypred!=cls))
        
        prec=tp/(tp+fp) if (tp+fp)>0 else 0
        rec=tp/(tp+fn) if (tp+fn)>0 else 0
        
        f1=2*(prec*rec)/(prec+rec) if (prec+rec)>0 else 0
        scores.append(f1)
    
    return np.mean(scores)
