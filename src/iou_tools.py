import numpy as np
from scipy.ndimage import gaussian_filter1d
from collections.abc import Iterable
# graphics
import matplotlib.pyplot as plt
import seaborn as sns
from aquarel import load_theme

# KMeans
from src.TemporalKmeans import TemporalKmeans

# Graphic settings
sns.set_style(style="darkgrid")
theme = load_theme("arctic_dark")
theme.apply()


############################## INTERSECTION OVER UNION ##############################


def compute_mean_iou(
    x1_min: np.array, x1_max: np.array, x2_min: np.array, x2_max: np.array
) -> float:
    """Compute the mean intersection over union between
    two sets of intervals where each interval is defined by a lower and an upper bound.

    Args:
        x1_min (np.array): The lower bounds of the first set of intervals.
        x1_max (np.array): The upper bounds of the first set of intervals.
        x2_min (np.array): The lower bounds of the second set of intervals.
        x2_max (np.array): The upper bounds of the second set of intervals.

    Returns:
        float: The mean IoU over the two sets of intervals.
        np.array: The IoU over each intervals.
    """
    intersection = np.maximum(
        0, np.minimum(x1_max, x2_max) - np.maximum(x1_min, x2_min)
    )
    union = (x1_max - x1_min) + (x2_max - x2_min) - intersection
    iou = intersection / union

    return np.mean(iou), iou


def compute_iou_matrix(tkm: TemporalKmeans, smoothing_coef: int = 5):
    """Compute the IoU matrix.

    Args:
        tkm (TemporalKmeans): The Temporal KMeans object.
        smoothing_coef (int, optional): The smoothing coefficient. The more it is, the more it smooth. Smoothing is done by convoluting a gaussian filter. This coefficient is the standard deviation of the gaussian filter to convolve on. Defaults to 5.
        Used on noisy datasets to kill noise, because it can affect the results.

    Returns:
        (np.array) : The 2 dimensional iou matrix.
    """
    iou_matrix = np.zeros(shape=(tkm.n_clusters, tkm.n_clusters))

    for i in range(tkm.centroids.shape[1]):
        for j in range(tkm.centroids.shape[1]):
            iou_matrix[i, j], _ = compute_mean_iou(
                gaussian_filter1d(tkm.mins[:, i], sigma=smoothing_coef),
                gaussian_filter1d(tkm.maxs[:, i], sigma=smoothing_coef),
                gaussian_filter1d(tkm.mins[:, j], sigma=smoothing_coef),
                gaussian_filter1d(tkm.maxs[:, j], sigma=smoothing_coef),
            )
    return iou_matrix


def plot_iou_matrix(
    iou_matrix: np.ndarray,
    figsize: tuple = (15, 6),
    fontsize_title: int = 13,
    title: str = "Intersection over Union matrix",
    cmap="mako",
) -> None:
    """Plot the IoU matrix using seaborn heatmap.

    Args:
        iou_matrix (np.ndarray): The computed IoU matrix.
        igsize (tuple, optional): The figsize. Defaults to (15, 6).
        fontsize_title (int, optional): The fontsize of the title. Defaults to 13.
        title (str, optional): The title. Defaults to "Courbes prototypiques".
        cmap (str, optional): The colormap name to choose. Defaults to "mako".
    """
    plt.title(title, fontsize=fontsize_title, fontweight="bold")
    sns.heatmap(
        data=iou_matrix, fmt=".2f", linewidths=1, annot=True, cmap=cmap, square=True
    )
    plt.show()
