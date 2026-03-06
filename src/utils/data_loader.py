import numpy as np
import gzip
import os
from urllib.request import urlretrieve


def _download_file(url, filepath):
    """Download file from URL if not already present."""
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(f"Downloading {url}...")
        urlretrieve(url, filepath)
        print(f"Downloaded to {filepath}")


def _load_mnist_images(filepath):
    """Load MNIST images from gzip file."""
    with gzip.open(filepath, 'rb') as f:
        # First 16 bytes are magic number and dimensions
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # Reshape to (num_images, 28, 28)
    return data.reshape(-1, 28, 28)


def _load_mnist_labels(filepath):
    """Load MNIST labels from gzip file."""
    with gzip.open(filepath, 'rb') as f:
        # First 8 bytes are magic number and count
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def _download_and_load_dataset(base_url, data_dir):
    """Download and load MNIST or Fashion-MNIST dataset."""
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    # Download all files
    for key, filename in files.items():
        url = base_url + filename
        filepath = os.path.join(data_dir, filename)
        _download_file(url, filepath)
    
    # Load data
    X_train = _load_mnist_images(os.path.join(data_dir, files['train_images']))
    y_train = _load_mnist_labels(os.path.join(data_dir, files['train_labels']))
    X_test = _load_mnist_images(os.path.join(data_dir, files['test_images']))
    y_test = _load_mnist_labels(os.path.join(data_dir, files['test_labels']))
    
    return (X_train, y_train), (X_test, y_test)


def load_data(dataset='mnist', normalize=True, flatten=True):
    """
    Load MNIST or Fashion-MNIST dataset without TensorFlow/Keras.
    Downloads directly from official sources.
    
    Args:
        dataset: 'mnist' or 'fashion_mnist'
        normalize: If True, normalize pixel values to [0, 1]
        flatten: If True, flatten images to 1D vectors
    
    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    # Create data directory
    data_dir = os.path.join(os.path.expanduser('~'), '.mnist_data', dataset)
    
    if dataset.lower() == 'mnist':
        # Using GitHub mirror as primary source
        base_url = 'https://raw.githubusercontent.com/fgnt/mnist/master/'
        (X_train, y_train), (X_test, y_test) = _download_and_load_dataset(base_url, data_dir)
    elif dataset.lower() == 'fashion_mnist':
        base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
        (X_train, y_train), (X_test, y_test) = _download_and_load_dataset(base_url, data_dir)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'mnist' or 'fashion_mnist'.")
    
    if normalize:
        X_train = X_train.astype('float64') / 255.0
        X_test = X_test.astype('float64') / 255.0
    
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    
    return (X_train, y_train), (X_test, y_test)
