import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Dict, List, Optional, Tuple


from src.data.download_mnist import download_and_preprocess_mnist
from src.data.plotting_mnist import plot_mnist_images


if __name__ == "__main__":
    # Download and preprocess MNIST data
    X_train, X_test, y_train, y_test, input_dim, output_dim = download_and_preprocess_mnist()

    # simulate y_pred for demonstration
    y_pred = y_test.copy()  # In practice, this would come from a model
    #modify some predictions to simulate errors
    if len(y_pred) > 5:
        y_pred[0] = (y_pred[0] + 1) % 10
        y_pred[1] = (y_pred[1] + 2) % 10
        y_pred[2] = (y_pred[2] + 3) % 10

    # Plot some test images
    plot_mnist_images(X_test, y_test, y_pred=y_pred, num_images=10)

    