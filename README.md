# DA6401 Assignment 1 — Neural Network from Scratch

This project implements a feedforward **Multi-Layer Perceptron (MLP)** entirely from scratch using **NumPy**, without any deep learning framework like PyTorch or TensorFlow. The network is trained and evaluated on the MNIST and Fashion-MNIST datasets, with all experiments tracked through **Weights & Biases**.

**W&B Report:** [View Report](https://api.wandb.ai/links/ce23b108-indian-institute-of-technology-madras/q8q4fs3f)  
**GitHub:** [sagar-250/assignment](https://github.com/sagar-250/assignment.git)

---

## Project Structure

```
da6401_assignment_1/
├── requirements.txt
├── src/
│   ├── train.py              # Training script (single run & W&B sweep)
│   ├── inference.py          # Evaluation / inference script
│   ├── best_model.npy        # Saved best model weights
│   ├── best_config.json      # Saved best model config
│   ├── ann/
│   │   ├── neural_network.py     # NeuralNetwork class (forward, backward, train)
│   │   ├── neural_layer.py       # Individual Layer class
│   │   ├── activations.py        # Sigmoid, ReLU, Tanh, Softmax, Identity
│   │   ├── optimizers.py         # SGD, Momentum, NAG, RMSprop
│   │   └── objective_functions.py # CrossEntropyLoss, MSELoss
│   └── utils/
│       └── data_loader.py        # MNIST / Fashion-MNIST loader
├── models/                   # Saved model checkpoints per dataset
└── notebooks/
    └── wandb_demo.ipynb      # W&B demo notebook
```

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/sagar-250/assignment.git
cd da6401_assignment_1
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `numpy` | Core tensor operations |
| `matplotlib` | Plotting |
| `keras` | Dataset loading (MNIST / Fashion-MNIST) |
| `wandb` | Experiment tracking and sweeps |
| `scikit-learn` | Metrics — F1, confusion matrix |

---

## Implementation Details

### Activations (`ann/activations.py`)

| Activation | Description |
|---|---|
| `ReLU` | max(0, x) |
| `Sigmoid` | 1 / (1 + e^−x) |
| `Tanh` | tanh(x) |
| `Softmax` | Numerically stable softmax |
| `Identity` | Pass-through; used on the output layer to return raw logits |

### Optimizers (`ann/optimizers.py`)

| Optimizer | Description |
|---|---|
| `SGD` | Vanilla stochastic gradient descent |
| `Momentum` | SGD with momentum (gamma = 0.9) |
| `NAG` | Nesterov Accelerated Gradient |
| `RMSprop` | Adaptive learning rate (beta = 0.9) |

### Loss Functions (`ann/objective_functions.py`)

| Loss | Description |
|---|---|
| `CrossEntropyLoss` | Fused with Softmax for numerically stable gradients |
| `MSELoss` | Mean squared error |

### Weight Initialization

| Method | Description |
|---|---|
| `xavier` | Xavier / Glorot uniform initialization |
| `random` | Small random normal initialization |

---

## Usage

### Single Training Run

```bash
python src/train.py \
  -d fashion_mnist \
  -e 30 \
  -b 128 \
  -o rmsprop \
  -lr 0.001 \
  -nhl 3 \
  -sz 128 128 128 \
  -a tanh \
  -w_i xavier \
  -l cross_entropy \
  --wandb_project da6401-assignment1
```

### Hyperparameter Sweep (W&B)

```bash
python src/train.py -d fashion_mnist --sweep --wandb_project da6401-assignment1
```

This runs a random search over the following search space:

- Epochs: `[3, 5, 7, 10]`
- Batch size: `[16, 32, 64]`
- Learning rate: `[0.001, 0.005, 0.01, 0.05]`
- Optimizer: `[sgd, momentum, rmsprop, nag]`
- Hidden architectures: `[128, 128, 128]`, `[128, 64, 32]`, `[128, 128, 64]`, `[128, 64, 32, 16]`
- Activation: `[relu, sigmoid, tanh]`
- Loss: `[cross_entropy, mse]`
- Weight init: `[xavier, random]`

### Inference / Evaluation

```bash
python src/inference.py \
  -d fashion_mnist \
  --model_path models/fashion_mnist/best_model.npy \
  --config_path models/fashion_mnist/config.json
```

Prints accuracy, precision, recall, macro F1-score, and loss on the test set.

---

## Experiment Tracking

All runs log the following metrics to Weights & Biases at each epoch:

- `train_loss`, `train_acc`, `train_f1`
- `val_loss`, `val_acc`, `val_f1`
- `overfitting_gap`, `accuracy_gap`, `f1_gap`
- `best_val_acc_so_far`, `best_val_f1_so_far`

At the end of training, final summary metrics are logged: `final_test_loss`, `final_test_acc`, `final_test_f1`, and `convergence_epoch`.

Model weights are saved only when the validation F1 score improves, so the best checkpoint per dataset is always preserved.

---

## Results

The full W&B report includes sweep results, hyperparameter importance analysis, and parallel coordinate plots:

[W&B Report](https://api.wandb.ai/links/ce23b108-indian-institute-of-technology-madras/q8q4fs3f)
