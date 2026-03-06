import argparse
import numpy as np
import json
import sys
import os
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data

def one_hot(y,num_classes=10):
    out=np.zeros((y.shape[0],num_classes))
    out[np.arange(y.shape[0]),y]=1
    return out

def parse_arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('-d','--dataset',type=str,default='mnist')
    parser.add_argument('--model_path',type=str,default='best_model.npy')
    parser.add_argument('--config_path',type=str,default='best_config.json')
    return parser.parse_args()


def load_model(model_path):
    data=np.load(model_path,allow_pickle=True).item()
    return data


def main():
    args=parse_arguments()
    
    print(f"loading cfg {args.config_path}")
    with open(args.config_path,'r') as f:
        cfg=json.load(f)
    
    print(f"loading {args.dataset} data")
    (xtr,ytr),(xtest,ytest)=load_data(dataset=args.dataset)
    ytest_oh=one_hot(ytest)
    
    print(f"making model {cfg['hidden_sizes']}")
    
    class Args:
        pass
    
    model_args=Args()
    model_args.input_size=cfg['input_size']
    model_args.hidden_size=cfg['hidden_sizes']
    model_args.output_size=cfg['output_size']
    model_args.activation=cfg['activation']
    model_args.weight_init=cfg['weight_init']
    model_args.loss=cfg['loss']
    
    model=NeuralNetwork(model_args)
    
    print(f"loading weights {args.model_path}")
    weights=load_model(args.model_path)
    model.set_weights(weights)
    
    print("testing")
    ypred=model.forward(xtest)
    
    ypred_cls=np.argmax(ypred,axis=1)
    ytrue_cls=np.argmax(ytest_oh,axis=1)
    
    acc=accuracy_score(ytrue_cls,ypred_cls)
    prec=precision_score(ytrue_cls,ypred_cls,average='macro',zero_division=0)
    rec=recall_score(ytrue_cls,ypred_cls,average='macro',zero_division=0)
    f1=f1_score(ytrue_cls,ypred_cls,average='macro',zero_division=0)
    
    per_prec=precision_score(ytrue_cls,ypred_cls,average=None,zero_division=0)
    per_rec=recall_score(ytrue_cls,ypred_cls,average=None,zero_division=0)
    per_f1=f1_score(ytrue_cls,ypred_cls,average=None,zero_division=0)
    
    conf=confusion_matrix(ytrue_cls,ypred_cls)
    
    print(f"\n{args.dataset} - {xtest.shape[0]} samples")
    print(f"acc {acc} prec {prec} rec {rec} f1 {f1}")
    
    print("\nper class:")
    for i in range(len(per_prec)):
        print(f"{i}: p={per_prec[i]} r={per_rec[i]} f1={per_f1[i]}")
    
    results={'accuracy':float(acc),'precision':float(prec),'recall':float(rec),'f1_score':float(f1),'per_class_precision':per_prec.tolist(),'per_class_recall':per_rec.tolist(),'per_class_f1':per_f1.tolist(),'confusion_matrix':conf.tolist(),'logits':ypred.tolist() if xtest.shape[0]<=100 else "too big"}
    
    return results


if __name__=='__main__':
    main()
