# Arrays & maths
from collections.abc import Iterable
import numpy as np
from scipy.ndimage import gaussian_filter1d

# graphics
import matplotlib.pyplot as plt
import seaborn as sns
from aquarel import load_theme

# KMeans
from src.TemporalKmeans import TemporalKmeans

# Graphic settings
theme = load_theme("arctic_dark")
theme.apply()

############### PLOTTING PROTOTYPICAL ###############


def plot_prototypes(
    index: Iterable,
    tkm: TemporalKmeans,
    smoothing_coef: int = 5,
    figsize: tuple = (15, 6),
    fontsize_title: int = 13,
    title: str = "Courbes prototypiques",
) -> None:
    """Plot the prototypes of the TemporalKMeans object.

    Args:
        index (Iterable): The index of the graphic (typically, the index of the dataframe that your studying).
        tkm (TemporalKmeans): The fitted Temporal KMeans.
        smoothing_coef (int, optional): The smoothing coefficient. The more it is, the more it smooth. Smoothing is done by convoluting a gaussian filter. This coefficient is the standard deviation of the gaussian filter to convolve on. Defaults to 5.

        # Figure parameters
        figsize (tuple, optional): The figsize. Defaults to (15, 6).
        fontsize_title (int, optional): The fontsize of the title. Defaults to 13.
        title (str, optional): The title. Defaults to "Courbes prototypiques".
    """
    if tkm.is_fitted:
        plt.figure(figsize=figsize)
        plt.title(title, fontsize=fontsize_title, fontweight="bold")

        for i in range(tkm.centroids.shape[1]):
            plt.plot(
                index,
                gaussian_filter1d(tkm.centroids[:, i], sigma=smoothing_coef),
                lw=3,
                label=[
                    f"Cluster {cluster_nb}"
                    for cluster_nb in np.arange(
                        start=1, stop=tkm.centroids.shape[1] + 1, step=1
                    )
                ][i],
            )
        plt.legend()
        plt.show()
    else:
        raise ValueError(
            "The Temporal KMeans must be fitted before plotting prototypes curves."
        )


def plot_prototypes_intervals(
    index: Iterable,
    tkm: TemporalKmeans,
    smoothing_coef: int = 5,
    figsize: tuple = (15, 6),
    fontsize_title: int = 13,
    title: str = "Courbes prototypiques",
) -> None:
    """Instead of plotting the centroids, plot the intervals defined by the min and max of each centroids at each time stamp.

    Args:
        index (Iterable): The index of the graphic (typically, the index of the dataframe that your studying).
        tkm (TemporalKmeans): The fitted Temporal KMeans.
        smoothing_coef (int, optional): The smoothing coefficient. The more it is, the more it smooth. Smoothing is done by convoluting a gaussian filter. This coefficient is the standard deviation of the gaussian filter to convolve on. Defaults to 5.

        # Figure parameters
        figsize (tuple, optional): The figsize. Defaults to (15, 6).
        fontsize_title (int, optional): The fontsize of the title. Defaults to 13.
        title (str, optional): The title. Defaults to "Courbes prototypiques".
    """
    if tkm.is_fitted:
        plt.figure(figsize=figsize)
        plt.title(title, fontsize=fontsize_title, fontweight="bold")

        for i in range(tkm.centroids.shape[1]):
            plt.fill_between(
                index,
                y1=gaussian_filter1d(tkm.mins[:, i], sigma=5),
                y2=gaussian_filter1d(tkm.maxs[:, i], sigma=5),
                label=[
                    f"Cluster {cluster_nb}"
                    for cluster_nb in np.arange(
                        start=1, stop=tkm.centroids.shape[1] + 1, step=1
                    )
                ][i],
                alpha=0.8,
            )
        plt.legend()
        plt.show()
    else:
        raise ValueError(
            "The Temporal KMeans must be fitted before plotting prototypes curves."
        )

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
