import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import ClusterMixin


class TemporalKmeans(ClusterMixin):
    """Temporal variation of the KMeans algorithm."""

    def __init__(self, estimator: KMeans) -> None:
        """Class constructor.

        Args:
            estimator (KMeans): The (unfitted) sklearn KMeans estimator that will be fitted on each timestep.
        """
        self.is_fitted = False
        self.estimator = estimator
        self.n_clusters = estimator.n_clusters

    def fit(self, X: np.ndarray, y=None) -> None:
        """Compute centroid curves based on th dataset.

        Args:
            X (np.ndarray): The array of the time series to be clustered.
                The first dimension must be the timesteps and the second one the different series.
            y (_type_, optional): None, for API consistence. Defaults to None.
        """
        if not (isinstance(X, np.ndarray)):
            raise ValueError(
                f"The input variable must be an array. The actual type is {type(X)}"
            )
        if len(X.shape) != 2:
            raise ValueError(
                f"The shape of the input array must be dimensional (dim 0 for the timesteps and dim 1 for the values)\n\
                The actual shape is {X.shape}"
            )
        self.labels = np.zeros(shape=X.shape)
        self.mins = np.zeros(shape=(X.shape[0], self.n_clusters))
        self.maxs = np.zeros(shape=(X.shape[0], self.n_clusters))
        self.centroids = np.zeros(shape=(X.shape[0], self.n_clusters))

        # Fit
        for moment_t in np.arange(X.shape[0]):
            # Clustering
            clusters_moment_t = self.estimator.fit(
                X[moment_t, :].reshape(-1, 1)
            )  # reshape to avoid error due to 1 column clustering

            # Mapping dictionnary
            mapper = {
                label: position
                for (label, position) in zip(
                    np.arange(self.n_clusters),
                    np.argsort(clusters_moment_t.cluster_centers_.ravel()),
                )
            }

            self.labels[moment_t, :] = self.__vectorized_mapping(
                clusters_moment_t.labels_.ravel(), mapper
            )  # Mapping step for coherence
            self.centroids[moment_t, :] = np.sort(
                clusters_moment_t.cluster_centers_.ravel()
            )  # sort for coherence
            self.mins[moment_t, :] = [
                np.min(X[moment_t, :][self.labels[moment_t, :] == numero_label])
                for numero_label in np.arange(self.n_clusters)
            ]
            self.maxs[moment_t, :] = [
                np.max(X[moment_t, :][self.labels[moment_t, :] == numero_label])
                for numero_label in np.arange(self.n_clusters)
            ]

        self.is_fitted = True

    def predict(self, X: np.ndarray, return_softmax=True):
        """Predict the cluster of a given time series matrix.

        Args:
            X (np.ndarray): 1d array, the time serie to classify.
            return_softmax (bool, optional): If set to true, results will be the estimated probabilities.
                    The shape will therefore be : (number of variables, number of clusters).
                If set to False, results will only be the predicted cluster.
                    The shape will therefore be : (number of variables, 1).
                Defaults to True.

        Returns:
            The results described in the return_softmax parameter description
        """
        if not (self.is_fitted):
            raise ValueError(
                "The model must be fitted to compute the prediction. You might wanna use the 'fit()' method ;)"
            )
        if X.shape[0] != self.centroids.shape[0]:
            raise ValueError(
                f"The number of observations must be the same as the array used in the fit method (must have the same number of timesteps). \
                The shape of the given input is : {X.shape}"
            )
        if not (isinstance(X, np.ndarray)):
            raise ValueError(
                f"The input variable must be an array. The actual type is {type(X)}"
            )

        if X.ndim == 1 or X.shape[1] == 1:
            return self.__predict_one_column(X, return_softmax)
        else:
            return self.__predict_several_columns(X, return_softmax)

    def is_fitted(self):
        """Check if the object is fitted.

        Returns:
            boolean: True if the estimator has been fitted, else otherwise.
        """
        return self.is_fitted

    def __vectorized_mapping(self, array: np.ndarray, mapper_dict: dict) -> np.ndarray:
        """Map a numpy array (using np.vectorize) given a dictionary.

        Args:
            array (np.ndarray): The array you want to map.
            mapper_dict (dict): The dictionary that will be used for the mapping.

        Returns:
            np.ndarray: The mapped array.
        """
        return np.vectorize(mapper_dict.__getitem__)(array)

    def __reverted_softmax(self, x: np.ndarray) -> float:
        """Compute a reverted function of the softmax, so that the min values gets the max probability associated.

        Args:
            x (np.ndarray): Input array.

        Returns:
            float: The softmax result.
        """
        return np.exp(1 - x) / sum(np.exp(1 - x))

    def __predict_one_column(self, vector: np.ndarray, return_softmax=True):
        """Predict the cluster of given time series.

        Args:
            X (np.ndarray): 1d array, the time serie to classify.
            return_softmax (bool, optional): If set to true, results will be the estimated probabilities.
                    The shape will therefore be : (1, number of clusters).
                If set to False, results will only be the predicted cluster.
                    The shape will therefore be : (1, 1).
                Defaults to True.

        Returns:
            The results described in the return_softmax parameter description.
        """
        sae = np.abs((vector.reshape((vector.shape[0], 1)) - self.centroids)).sum(
            axis=0
        )
        if return_softmax:
            normalized_distance = np.log(sae / np.sum(sae))
            estimated_probas = self.__reverted_softmax(normalized_distance)
            return np.array(estimated_probas)
        else:
            return np.argmin(sae)

    def __predict_several_columns(self, X: np.ndarray, return_softmax=True):
        """Predict the cluster of given multiples time series.

        Args:
            X (np.ndarray): 1d array, the time serie to classify.
            return_softmax (bool, optional): If set to true, results will be the estimated probabilities.
                    The shape will therefore be : (number of variables, number of clusters).
                If set to False, results will only be the predicted cluster.
                    The shape will therefore be : (number of variables, 1).
                Defaults to True.

        Returns:
            The results described in the return_softmax parameter description.
        """
        predictions = list()
        for column_index in np.arange(X.shape[1]):

            if return_softmax:
                predictions.append(
                    self.__predict_one_column(
                        X[:, column_index], return_softmax
                    ).ravel()
                )

            else:
                predictions.append(
                    self.__predict_one_column(X[:, column_index], return_softmax)
                )

        return np.array(predictions)