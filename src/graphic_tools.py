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
sns.set_style(style="darkgrid")
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
