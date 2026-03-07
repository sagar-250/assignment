import argparse
import numpy as np
import sys
import os
import wandb
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD, Momentum, RMSprop, NAG
from utils.data_loader import load_data

OPTIMIZER_MAP = {
    'sgd': SGD,
    'momentum': Momentum,
    'rmsprop': RMSprop,
    'nag': NAG,
}

def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out

def load_model(model_path):
    """
    Load trained model weights from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data

def compute_f1_score(y_true, y_pred, num_classes=10):

    # Convert to class indices if one-hot encoded
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    f1_scores = []
    for class_idx in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == class_idx) & (y_pred == class_idx))
        fp = np.sum((y_true != class_idx) & (y_pred == class_idx))
        fn = np.sum((y_true == class_idx) & (y_pred != class_idx))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return np.mean(f1_scores)  # macro F1

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'], help='Choose between mnist and fashion_mnist')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'mse'], help='Choice of mean_squared_error or cross_entropy')
    parser.add_argument('-o', '--optimizer', type=str, default='sgd', choices=['sgd', 'momentum', 'nag', 'rmsprop'], help='One of sgd, momentum, nag, rmsprop')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='Weight decay for L2 regularization')
    parser.add_argument('-nhl', '--num_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=None, help='Number of neurons in each hidden layer (list of values)')
    parser.add_argument('-a', '--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'], help='Choice of sigmoid, tanh, relu for every hidden layer')
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier', choices=['xavier', 'random'], help='Choice of random or xavier')
    parser.add_argument('--wandb_project', type=str, default='da6401-assignment1_try2')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--save_model', action='store_true', help='Save model weights after training')
    parser.add_argument('--model_save_path', type=str, default='models/model.pkl')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep instead of single training')
    return parser.parse_args()


def train_with_wandb(config=None, project='da6401-assignment1_try2', entity=None):
    with wandb.init(config=config, project=project, entity=entity):
        cfg = wandb.config
        
        print(f"\nSuccessfully connected to W&B")
        print(f"  Project: {wandb.run.project}")
        print(f"  Run ID: {wandb.run.id}")
        print(f"  Run URL: {wandb.run.url}\n")

        (X_train_full, y_train_full), (X_test, y_test) = load_data(dataset=cfg.dataset)

        val_size = int(0.1 * X_train_full.shape[0])
        X_val, y_val = X_train_full[:val_size], y_train_full[:val_size]
        X_train, y_train = X_train_full[val_size:], y_train_full[val_size:]

        y_train_oh = one_hot(y_train)
        y_val_oh   = one_hot(y_val)
        y_test_oh  = one_hot(y_test)

        if hasattr(cfg, 'hidden_size') and cfg.hidden_size:
            hidden_sizes = cfg.hidden_size
        elif hasattr(cfg, 'num_neurons'):
            hidden_sizes = [cfg.num_neurons] * cfg.num_layers
        else:
            hidden_sizes = [128] * cfg.num_layers
        class Arg:
            pass
        
        model_args = Arg()
        model_args.input_size = X_train.shape[1]
        model_args.hidden_size = hidden_sizes
        model_args.output_size = 10
        model_args.activation = cfg.activation
        model_args.weight_init = cfg.weight_init
        model_args.loss = cfg.loss
        
        model = NeuralNetwork(model_args)

        optimizer_cls = OPTIMIZER_MAP[cfg.optimizer]
        optimizer = optimizer_cls(lr=cfg.learning_rate)

        hidden_size_str = 'x'.join(map(str, hidden_sizes)) if hidden_sizes else 'default'
        run_name = (f"{cfg.dataset}_opt_{cfg.optimizer}_lr_{cfg.learning_rate}_"
                   f"layers_{cfg.num_layers}_sz_{hidden_size_str}_"
                   f"act_{cfg.activation}_init_{cfg.weight_init}_loss_{cfg.loss}_bs_{cfg.batch_size}")
        wandb.run.name = run_name

        best_val_acc = 0.0
        best_val_loss = float('inf')
        best_val_f1 = 0.0
        best_test_f1 = 0.0
        convergence_epoch = 0
        
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        train_f1s = []
        val_f1s = []

        def enhanced_epoch_callback(metrics):
            epoch = metrics['epoch']
            
            wandb.log({
                'epoch': epoch,
                'train_loss': metrics.get('train_loss', 0),
                'train_acc': metrics.get('train_acc', 0),
                'train_f1': metrics.get('train_f1', 0),
                'val_loss': metrics.get('vapython inference.py -d fashion_mnist --model_path models/fashion_mnist/best_model.npy --config_path models/fashion_mnist/config.jsonl_loss', 0),
                'val_acc': metrics.get('val_acc', 0),
                'val_f1': metrics.get('val_f1', 0),
            })
            
            train_losses.append(metrics.get('train_loss', 0))
            val_losses.append(metrics.get('val_loss', 0))
            train_accs.append(metrics.get('train_acc', 0))
            val_accs.append(metrics.get('val_acc', 0))
            train_f1s.append(metrics.get('train_f1', 0))
            val_f1s.append(metrics.get('val_f1', 0))
            
            current_val_acc = metrics.get('val_acc', 0)
            current_val_loss = metrics.get('val_loss', 0)
            current_val_f1 = metrics.get('val_f1', 0)
            
            nonlocal best_val_acc, best_val_loss, best_val_f1, convergence_epoch
            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                convergence_epoch = epoch
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
            
            wandb.log({
                'overfitting_gap': metrics.get('train_loss', 0) - metrics.get('val_loss', 0),
                'accuracy_gap': metrics.get('train_acc', 0) - metrics.get('val_acc', 0),
                'f1_gap': metrics.get('train_f1', 0) - metrics.get('val_f1', 0),
                'best_val_acc_so_far': best_val_acc,
                'best_val_loss_so_far': best_val_loss,
                'best_val_f1_so_far': best_val_f1,
            })

        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"  Dataset: {cfg.dataset}")
        print(f"  Optimizer: {cfg.optimizer}")
        print(f"  Learning Rate: {cfg.learning_rate}")
        print(f"  Epochs: {cfg.epochs}")
        print(f"  Batch Size: {cfg.batch_size}")
        print(f"  Architecture: {hidden_sizes}")
        print(f"  Activation: {cfg.activation}")
        print(f"  Weight Init: {cfg.weight_init}")
        print(f"  Loss: {cfg.loss}")
        print(f"{'='*60}\n")

        model.train(
            X_train, y_train_oh, optimizer,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            X_val=X_val, y_val=y_val_oh,
            epoch_callback=enhanced_epoch_callback
        )

        test_loss, test_acc, test_f1 = model.evaluate(X_test, y_test_oh)
        
        final_metrics = {
            'final_test_loss': test_loss,
            'final_test_acc': test_acc,
            'final_test_f1': test_f1,
            'final_train_loss': train_losses[-1] if train_losses else 0,
            'final_train_acc': train_accs[-1] if train_accs else 0,
            'final_train_f1': train_f1s[-1] if train_f1s else 0,
            'final_val_loss': val_losses[-1] if val_losses else 0,
            'final_val_acc': val_accs[-1] if val_accs else 0,
            'final_val_f1': val_f1s[-1] if val_f1s else 0,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'best_val_f1': best_val_f1,
            'convergence_epoch': convergence_epoch,
            'final_overfitting_gap': train_losses[-1] - val_losses[-1] if train_losses and val_losses else 0,
        }
        wandb.log(final_metrics)
        
        # Create dataset-specific folder
        dataset_folder = f'models/{cfg.dataset}'
        os.makedirs(dataset_folder, exist_ok=True)
        
        best_model_path = f'{dataset_folder}/best_model.npy'
        best_config_path = f'{dataset_folder}/config.json'
        best_f1_file = f'{dataset_folder}/best_val_f1.txt'
        
        current_best_val_f1 = 0.0
        if os.path.exists(best_f1_file):
            with open(best_f1_file, 'r') as f:
                current_best_val_f1 = float(f.read().strip())
        
        # Save model based on validation F1 
        if best_val_f1 > current_best_val_f1:
            with open(best_f1_file, 'w') as f:
                f.write(str(best_val_f1))
            
            # Save model weights using get_weights()
            best_weights = model.get_weights()
            np.save(best_model_path, best_weights)
            
            # Save model configuration
            config = {
                'dataset': cfg.dataset,
                'input_size': X_train.shape[1],
                'hidden_sizes': hidden_sizes,
                'output_size': 10,
                'activation': cfg.activation,
                'weight_init': cfg.weight_init,
                'loss': cfg.loss,
                'optimizer': cfg.optimizer,
                'learning_rate': cfg.learning_rate,
                'batch_size': cfg.batch_size,
                'epochs': cfg.epochs,
                'weight_decay': cfg.weight_decay,
                'best_test_f1': float(test_f1),
                'best_val_f1': float(best_val_f1),
                'best_val_acc': float(best_val_acc),
                'convergence_epoch': convergence_epoch,
                'wandb_run_id': wandb.run.id,
            }
            with open(best_config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"\n NEW BEST MODEL! Val F1: {best_val_f1:.4f} (prev: {current_best_val_f1:.4f})")
            print(f"  -> Corresponding Test F1: {test_f1:.4f}")
            wandb.summary['best_val_f1_overall'] = best_val_f1
            wandb.summary['best_test_f1_for_best_val'] = test_f1
            wandb.summary['best_run_id'] = wandb.run.id
        else:
            print(f"\n  Current best Val F1 remains: {current_best_val_f1:.4f} (this run: {best_val_f1:.4f})")
        
        print(f"\n✓ Training completed!")
        print(f"  Final Test Loss: {test_loss:.4f}")
        print(f"  Final Test Accuracy: {test_acc:.4f}")
        print(f"  Final Test F1: {test_f1:.4f}")
        print(f"  Best Val F1: {best_val_f1:.4f} (epoch {convergence_epoch})")


def main():
    args = parse_arguments()

    if args.sweep:
        # Run hyperparameter sweep
        sweep_config = {
            'method': 'random',
            'metric': {'name': 'final_val_f1', 'goal': 'maximize'},
            'parameters': {
                'dataset':       {'value': args.dataset},
                'epochs':        {'values': [10,5,3,7]},
                'batch_size':    {'values': [16,32,64]},
                'learning_rate': {'values': [0.001, 0.005, 0.01, 0.05]},
                'optimizer':     {'values': ['sgd','momentum','rmsprop','nag']},
                'hidden_size':   {'values': [          
                    [128, 64, 32],          
                    [128, 128, 64],           
                    [128, 64, 32, 16],     
                    [128, 128, 128],
                ]},
                'num_layers':    {'value': 2},
                'activation':    {'values': ['relu','sigmoid','tanh']},
                'loss':          {'values': ['cross_entropy', 'mse']},  
                'weight_init':   {'values': ['xavier','random']},
                'weight_decay':  {'value': args.weight_decay},
            }
        }
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
        wandb.agent(sweep_id, function=lambda: train_with_wandb(project=args.wandb_project, entity=args.wandb_entity), count=100)
    else:
        # Run single training with command line args
        class Config:
            pass
        
        config = Config()
        config.dataset = args.dataset
        config.epochs = args.epochs
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        config.optimizer = args.optimizer
        config.hidden_size = args.hidden_size if args.hidden_size else [128] * args.num_layers
        config.num_layers = args.num_layers
        config.activation = args.activation
        config.loss = args.loss
        config.weight_init = args.weight_init
        config.weight_decay = args.weight_decay
        
        train_with_wandb(config, project=args.wandb_project, entity=args.wandb_entity)


if __name__ == '__main__':
    main()
