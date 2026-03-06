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
from utils.data_loader import load_dataset


def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(
        description="Run inference with a trained NumPy MLP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Same CLI as train.py 
    parser.add_argument(
        "-d", "--dataset",
        type=str, default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int, default=30,
        help="(Unused in inference; kept for CLI parity with train.py.)",
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int, default=128,
        help="Batch size for inference forward passes.",
    )
    parser.add_argument(
        "-l", "--loss",
        type=str, default="cross_entropy",
        choices=["cross_entropy", "mse"],
        help="Loss function used during training (must match the saved model).",
    )
    parser.add_argument(
        "-o", "--optimizer",
        type=str, default="rmsprop",
        choices=["sgd", "momentum", "nag", "rmsprop"],
        help="(Unused in inference; kept for CLI parity.)",
    )
    parser.add_argument(
        "-lr", "--learning_rate",
        type=float, default=0.001553536703042097,
        help="(Unused in inference; kept for CLI parity.)",
    )
    parser.add_argument(
        "-wd", "--weight_decay",
        type=float, default=0.0001,
        help="(Unused in inference; kept for CLI parity.)",
    )
    parser.add_argument(
        "-nhl", "--num_layers",
        type=int, default=3,
        help="Number of hidden layers (must match the saved model).",
    )
    parser.add_argument(
        "-sz", "--hidden_size",
        type=int, nargs="+", default=[128, 128, 128],
        help="Hidden layer sizes (must match the saved model).",
    )
    parser.add_argument(
        "-a", "--activation",
        type=str, default="tanh",
        choices=["relu", "sigmoid", "tanh"],
        help="Activation function (must match the saved model).",
    )
    parser.add_argument(
        "-w_i", "--weight_init",
        type=str, default="xavier",
        choices=["xavier", "random"],
        help="(Used to build model skeleton; actual weights come from the file.)",
    )
    parser.add_argument(
        "-w_p", "--wandb_project",
        type=str, default="da6401_assignment1",
        help="(Unused in inference; kept for CLI parity.)",
    )

    # Inference specific arguments
    parser.add_argument(
        "--model_path",
        type=str, default="best_model.npy",
        help="Relative path to the saved model weights (.npy file).",
    )
    parser.add_argument(
        "--config_path",
        type=str, default="best_config.json",
        help="Optional: load architecture config from JSON instead of CLI flags.",
    )
    parser.add_argument(
        "--val_split",
        type=float, default=0.1,
        help="Validation split fraction (unused; evaluation is on the test split).",
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        default=True,
        help="Disable W&B logging during inference (default: True).",
    )

    return parser.parse_args()


# load model weights from .npy file
def load_model(model_path: str) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: '{model_path}'. "
            "Run train.py first or check the path."
        )
    data = np.load(model_path, allow_pickle=True).item()
    return data

# Override CLI args with config from JSON file (if it exists)
def override_args_from_config(args, config_path: str):
    if not os.path.exists(config_path):
        print(f"[Config] '{config_path}' not found — using CLI arguments.")
        return args

    with open(config_path) as f:
        cfg = json.load(f)

    arch_keys = [
        "dataset", "num_layers", "hidden_size", "activation",
        "weight_init", "loss", "optimizer", "learning_rate", "weight_decay",
        "batch_size",
    ]
    for key in arch_keys:
        if key in cfg:
            setattr(args, key, cfg[key])

    print(f"[Config] Loaded architecture from '{config_path}'.")
    return args


# Evaluate the model on the test set and compute metrics
def evaluate_model(model: NeuralNetwork, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Run inference and compute all required metrics.

    Args:
        model  : NeuralNetwork with loaded weights.
        X_test : Test inputs,  shape (N, 784)
        y_test : Integer labels, shape (N,)
    Returns:
        dict with keys: logits, loss, accuracy, precision, recall, f1, predictions
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix,
    )

    batch_size = 512
    all_logits = []
    N = X_test.shape[0]
    for start in range(0, N, batch_size):
        batch = X_test[start: start + batch_size]
        all_logits.append(model.forward(batch))

    logits = np.concatenate(all_logits, axis=0)  
    preds = np.argmax(logits, axis=1)           

    from ann.objective_functions import get_loss
    loss_fn = get_loss(model.cli_args.loss)
    loss = loss_fn.forward(y_test, logits)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec  = recall_score(y_test, preds, average="macro", zero_division=0)
    f1   = f1_score(y_test, preds, average="macro", zero_division=0)
    cm   = confusion_matrix(y_test, preds)

    return {
        "logits":      logits,
        "predictions": preds,
        "loss":        float(loss),
        "accuracy":    float(acc),
        "precision":   float(prec),
        "recall":      float(rec),
        "f1":          float(f1),
        "confusion_matrix": cm,
    }


def main() -> dict:
    """
    Main inference function.

    1. Parse CLI args (override with best_config.json if present)
    2. Load dataset test split
    3. Reconstruct model architecture and load weights
    4. Run evaluation
    5. Print metrics
    6. Return metrics dict

    Returns:
        dict: logits, loss, accuracy, precision, recall, f1
    """
    args = parse_arguments()
    np.random.seed(args.seed)
    args = override_args_from_config(args, args.config_path)
    _, _, _, _, X_test, y_test, label_names = load_dataset(
        args.dataset, val_split=args.val_split, seed=args.seed
    )

    # reconstruct model and load weights
    model = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)
    print(f"[Inference] Loaded weights from '{args.model_path}'.")

    # evaluate
    results = evaluate_model(model, X_test, y_test)

    # report
    print("\n" + "=" * 45)
    print("          INFERENCE RESULTS")
    print("=" * 45)
    print(f"  Dataset   : {args.dataset}")
    print(f"  Samples   : {X_test.shape[0]}")
    print(f"  Loss      : {results['loss']:.4f}")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}  (macro)")
    print(f"  Recall    : {results['recall']:.4f}  (macro)")
    print(f"  F1-Score  : {results['f1']:.4f}  (macro)")
    print("=" * 45)

    # Per-class F1
    from sklearn.metrics import classification_report
    print("\nPer-class report:")
    print(
        classification_report(
            y_test, results["predictions"],
            target_names=label_names, zero_division=0,
        )
    )

    print("\nEvaluation complete!")
    return results


if __name__ == "__main__":
    main()