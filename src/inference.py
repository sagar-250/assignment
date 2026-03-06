"""
Inference Script
Load a trained NumPy MLP and evaluate it on a test set.

Outputs: Accuracy, Precision, Recall, F1-score (macro), Loss, and Logits.

Usage example:
    python inference.py --model_path best_model.npy -d fashion_mnist
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained NumPy MLP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Same CLI as train.py 
    parser.add_argument("-d", "--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", default=30)
    parser.add_argument("-b", "--batch_size", default=128)
    parser.add_argument("-l", "--loss", default="cross_entropy", choices=["cross_entropy", "mse"])
    parser.add_argument("-o", "--optimizer", default="rmsprop", choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr", "--learning_rate", default=0.001553536703042097)
    parser.add_argument("-wd", "--weight_decay", default=0.0001)
    parser.add_argument("-nhl", "--num_layers", default=3)
    parser.add_argument("-sz", "--hidden_size", nargs="+", default=[128, 128, 128])
    parser.add_argument("-a", "--activation", default="tanh", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-w_i", "--weight_init", default="xavier", choices=["xavier", "random"])
    parser.add_argument("-w_p", "--wandb_project", default="da6401_assignment_try2")
    parser.add_argument("--model_path", default="best_model.npy")
    parser.add_argument("--config_path", default="best_config.json")
    parser.add_argument("--val_split", default=0.1)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--no_wandb", action="store_true", default=True)

    return parser.parse_args()


def load_model(model_path: str) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: '{model_path}'. Run train.py first or check the path.")
    data = np.load(model_path, allow_pickle=True).item()
    return data

def override_args_from_config(args, config_path: str):
    if not os.path.exists(config_path):
        print(f"[Config] '{config_path}' not found — using CLI arguments.")
        return args

    with open(config_path) as f:
        cfg = json.load(f)

    # load architecture params from config
    arch_keys = ["dataset", "num_layers", "activation", "weight_init", "loss", "optimizer", "learning_rate", "weight_decay", "batch_size", "input_size", "output_size"]
    for key in arch_keys:
        if key in cfg:
            setattr(args, key, cfg[key])
    
    # handle both hidden_size and hidden_sizes
    if "hidden_sizes" in cfg:
        setattr(args, "hidden_size", cfg["hidden_sizes"])
    elif "hidden_size" in cfg:
        setattr(args, "hidden_size", cfg["hidden_size"])

    print(f"[Config] Loaded architecture from '{config_path}'.")
    return args


def evaluate_model(model: NeuralNetwork, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    bs = 512
    logits_list = []
    n = X_test.shape[0]
    for start in range(0, n, bs):
        batch = X_test[start: start + bs]
        logits_list.append(model.forward(batch))

    logits = np.concatenate(logits_list, axis=0)  
    preds = np.argmax(logits, axis=1)           

    # convert to one-hot for loss
    y_oh = np.zeros((y_test.shape[0], 10))
    y_oh[np.arange(y_test.shape[0]), y_test] = 1
    
    loss = model.loss_fn.loss(y_oh, logits)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, preds)

    return {"logits": logits, "predictions": preds, "loss": float(loss), "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "confusion_matrix": cm}


def main() -> dict:
    args = parse_arguments()
    np.random.seed(args.seed)
    args = override_args_from_config(args, args.config_path)
    
    # load data
    (x_tr, y_tr), (x_te, y_te) = load_data(dataset=args.dataset)
    
    # get class names
    if args.dataset == 'mnist':
        labels = [str(i) for i in range(10)]
    else:
        labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # set input/output if not in config
    if not hasattr(args, 'input_size'):
        args.input_size = x_te.shape[1]
    if not hasattr(args, 'output_size'):
        args.output_size = 10

    # build model and load weights
    mdl = NeuralNetwork(args)
    wts = load_model(args.model_path)
    mdl.set_weights(wts)
    print(f"[Inference] Loaded weights from '{args.model_path}'.")

    # run evaluation
    res = evaluate_model(mdl, x_te, y_te)

    # print results
    print("\n" + "=" * 45)
    print("          INFERENCE RESULTS")
    print("=" * 45)
    print(f"  Dataset   : {args.dataset}")
    print(f"  Samples   : {x_te.shape[0]}")
    print(f"  Loss      : {res['loss']:.4f}")
    print(f"  Accuracy  : {res['accuracy']:.4f}")
    print(f"  Precision : {res['precision']:.4f}  (macro)")
    print(f"  Recall    : {res['recall']:.4f}  (macro)")
    print(f"  F1-Score  : {res['f1']:.4f}  (macro)")
    print("=" * 45)

    from sklearn.metrics import classification_report
    print("\nPer-class report:")
    print(classification_report(y_te, res["predictions"], target_names=labels, zero_division=0))

    print("\nEvaluation complete!")
    return res


if __name__ == "__main__":
    main()