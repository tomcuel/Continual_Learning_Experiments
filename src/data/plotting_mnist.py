# =======================================
# Library Imports
# =======================================
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from typing import Dict, List, Optional, Tuple


# =======================================
# Show the MNIST images from flattened vectors
# =======================================
MNIST_LABELS = [str(i) for i in range(10)]

def plot_mnist_images(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    num_images: int = 10,
    img_shape: Tuple[int, int] = (28, 28)
):
    """
    Plot MNIST images from flattened vectors

    Parameters
    ----------
    X : np.ndarray
        Flattened MNIST images (N, 784)
    y : np.ndarray
        True labels (N,) or one-hot (N, 10)
    y_pred : np.ndarray, optional
        Predicted labels (N,)
    num_images : int
        Number of images to display
    img_shape : Tuple[int, int]
        MNIST image shape (28, 28)

    Returns
    -------
    None

    Usage Example
    --------------
    plot_mnist_images(X_test, y_test, y_pred=y_pred, num_images=15)
    """
    # Convert labels to integer if needed
    if y.ndim > 1:
        y_true = np.argmax(y, axis=1)
    else:
        y_true = y

    num_images = min(num_images, len(X))

    n_cols = min(5, num_images)
    n_rows = ceil(num_images / n_cols)

    plt.figure(figsize=(3 * n_cols, 3 * n_rows))

    for i in range(num_images):
        plt.subplot(n_rows, n_cols, i + 1)

        img = X[i].reshape(img_shape)
        plt.imshow(img, cmap="gray")
        plt.axis("off")

        title = f"True: {MNIST_LABELS[y_true[i]]}"
        if y_pred is not None:
            title += f"\nPred: {MNIST_LABELS[y_pred[i]]}"
            color = "green" if y_true[i] == y_pred[i] else "red"
            plt.title(title, color=color, fontsize=10)
        else:
            plt.title(title, fontsize=10)

    plt.tight_layout()
    plt.show()

