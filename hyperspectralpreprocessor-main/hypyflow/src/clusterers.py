import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from sklearn.cluster import KMeans, MiniBatchKMeans
from kmedoids import KMedoids


class SpatialSpectralCluster:
    def __init__(
        self,
        slic_n_segments,
        slic_compactness,
        k_n_clusters: int,
        slic_sigma=0.2,
        slic_reduction_method: str = "median",
        kmed_metric = "euclidean",
        kmed_method: str = "fasterpam",
        kmed_init: str = "k-medoids++",
        kmed_max_iter: int = 300,
        k_method: str = "kmedoids",
        kmeans_mini_batch_size: int = 1024,
    ):
        """
        Initializes the clustering class with the specified parameters. The clustering method can be K-Medoids, K-Means, or Mini-Batch K-Means.
        The superpixels are obtained using SLIC.

        Parameters:
            slic_n_segments : int
                The number of superpixels to obtain with SLIC. Setting to 0 will disable SLIC.
            slic_compactness : float
                This parameter balances color (spectral) proximity and space proximity. Higher values mean that the superpixels will be more square.
            k_n_clusters : int.
                The number of clusters for the clustering algorithm. This will be the final number of "pixels" in the resulting image.
            slic_sigma : float, optional, default=0.2
                Width of Gaussian smoothing kernel for pre-processing for each dimension of the image. Leave to 0 if no smoothing is required.
            slic_reduction_method : str, optional, default="median"
                Once superpixels are obtained, this method is used to reduce the superpixel to a single value. Supported methods are "mean", "median", and "random".
            kmed_metric : str|callable, optional, default="euclidean"
                The metric for k-medoids clustering. Could be a callable or a string. Supported metrics when str are the ones in sklearn.metrics.pairwise_distances. If callable, it should take two arrays and return a distance.
            kmed_method : str, optional, default="fasterpam"
                The method for k-medoids clustering. Supported methods are "pam", "fasterpam".
            kmed_init : str, optional, default="k-medoids++"
                The initialization method for k-medoids clustering.
            kmed_max_iter : int, optional, default=300
                The maximum number of iterations for k-medoids clustering.
            k_method : str, optional, default="kmedoids"
                The clustering method to use. Supported methods are "kmedoids", "kmeans", and "kmeans_mini_batch".
            kmeans_mini_batch_size : int, optional, default=1024
                The mini-batch size for mini-batch k-means clustering.

        """

        self.slic_n_segments = slic_n_segments
        self.slic_compactness = slic_compactness
        self.slic_sigma = slic_sigma
        self.slic_reduction_method = slic_reduction_method

        self.k_n_clusters = k_n_clusters
        self.kmed_metric = kmed_metric
        self.kmed_method = kmed_method
        self.kmed_init = kmed_init
        self.kmed_max_iter = kmed_max_iter
        self.kmeans_mini_batch_size = kmeans_mini_batch_size

        self.k_method = k_method

        supported_methods = ["kmedoids", "kmeans", "kmeans_mini_batch"]
        if self.k_method not in supported_methods:
            raise ValueError(f"Invalid method. Choose one of {supported_methods}")

        if self.k_method == "kmeans_mini_batch":
            self.clustering = MiniBatchKMeans(
                n_clusters=self.k_n_clusters,
                random_state=42,
                max_iter=self.kmed_max_iter,
                batch_size=self.kmeans_mini_batch_size,
            )
        elif self.k_method == "kmeans":
            self.clustering = KMeans(
                n_clusters=self.k_n_clusters,
                random_state=42,
                max_iter=self.kmed_max_iter,
            )
        elif self.k_method == "kmedoids":
            self.clustering = KMedoids(
                n_clusters=self.k_n_clusters,
                random_state=42,
                max_iter=self.kmed_max_iter,
                metric=self.kmed_metric,
                method=self.kmed_method,
            )

        self.first_mask_is_bg = False

    def _slic(self, cube, mask=None, plot=True):
        """
        Internal method to perform SLIC (Simple Linear Iterative Clustering) segmentation on a hyperspectral image cube.
        Parameters:
            cube : numpy.ndarray
                The hyperspectral image cube to be segmented. It should be a 3D array where the third dimension represents the spectral bands.
            mask : numpy.ndarray, optional
                An optional mask to apply during segmentation. If provided, only the masked regions will be segmented.
            plot : bool, default=True
                If True, the function will plot the segmentation results.
        Returns:
            superpixel_signatures_cube : numpy.ndarray
                A 2D array where each row represents the spectral signature of a superpixel.
            segments : numpy.ndarray
                A 2D array where each element represents the segment label of the corresponding pixel in the input cube.
        """

        segments = slic(
            cube,
            n_segments=self.slic_n_segments,
            compactness=self.slic_compactness,
            channel_axis=2,
            convert2lab=False,
            enforce_connectivity=False,
            mask=mask,
            sigma=self.slic_sigma,
        )
        if plot:
            # Normalize the cube for display purposes
            image_to_plot = np.max(cube, axis=2)
            plt.figure(figsize=(10, 8))
            plt.imshow(image_to_plot, alpha=1)
            image_with_borders = mark_boundaries(image_to_plot, segments)
            plt.imshow(image_with_borders, alpha=0.3)
            plt.title(f"SLIC Segmentation in {self.slic_n_segments} superpixels")
            plt.show()
        self.first_mask_is_bg = False
        if mask is not None:
            # This implies that the 0 is the background, so we need to be aware of this
            self.first_mask_is_bg = True

        n_masks = len(np.unique(segments))

        superpixel_signatures_cube = np.zeros((n_masks, self.B))

        for i, s in enumerate(np.unique(segments)):
            mask_ = segments == s
            if self.first_mask_is_bg and i == 0:
                continue
            if self.slic_reduction_method == "mean":
                superpixel_signatures_cube[i] = np.mean(cube[mask_], axis=0)
            elif self.slic_reduction_method == "median":
                superpixel_signatures_cube[i] = np.median(cube[mask_], axis=0)
            elif self.slic_reduction_method == "random":
                idx = np.random.choice(np.sum(mask_), 1)
                superpixel_signatures_cube[i] = cube[mask_][idx]
            else:
                raise ValueError("Invalid reduction method. Choose 'mean', 'median', or 'random'.")

        self.superpixel_signatures_cube = superpixel_signatures_cube

        return superpixel_signatures_cube, segments

    def run(self, cube, mask=None, plot=True):
        """
        Perform SLIC (Simple Linear Iterative Clustering) segmentation on a hyperspectral image cube and
        compute superpixel signatures.
        On the superpixel data, perform K-Medoids clustering.

        Parameters:
            cube : numpy.ndarray
                The hyperspectral image cube with shape (H, W, B) where H is the height, W is the width, and B is the number of bands.
            mask : numpy.ndarray
                A binary mask to specify the region of interest for segmentation.
            plot : bool, optional
                If True, plots the segmentation results. Default is True.

        Returns:
            superpixel_signatures_cube : numpy.ndarray
                An array of shape (N, B) where N is the number of clusters and B is the number of bands. Each row represents the signature of a cluster.
            slic_cluster_image : numpy.ndarray
                An image where each pixel is assigned the superpixel index.
            kmed_cluster_image : numpy.ndarray
                An image where each pixel is assigned the cluster index.
        """

        self.H, self.W, self.B = cube.shape
        if self.slic_n_segments == 0:
            # Skip the slic part
            superpixel_signatures_cube = cube.reshape(-1, self.B)
            # Create the slic_cluster_image with 1 unique value per pixel
            slic_custer_image = np.arange(self.H * self.W).reshape(self.H, self.W)
        else:
            # first run the slic
            superpixel_signatures_cube, slic_custer_image = self._slic(cube, mask, plot=plot)
        # then run the kmedoids
        clusters = self.clustering.fit(superpixel_signatures_cube)
        labels = clusters.labels_

        if self.k_method == "kmeans_mini_batch" or self.k_method == "kmeans":
            medoids_indices = clusters.cluster_centers_
        else:
            medoids_indices = clusters.medoid_indices_

        kmed_cluster_image = np.zeros((self.H, self.W))
        for i, s in enumerate(np.unique(slic_custer_image)):
            mask_ = slic_custer_image == s
            if self.first_mask_is_bg and i == 0:
                continue
            kmed_cluster_image[mask_] = labels[i]
        if plot:
            cmap = plt.cm.get_cmap("tab20", self.k_n_clusters + 1)
            plt.figure(figsize=(10, 8))
            plt.imshow(kmed_cluster_image, cmap=cmap)
            plt.colorbar()
            plt.title(f"Clustered Image with {self.k_n_clusters} Clusters (0 is mask) using {self.k_method}")
            plt.show()
        self.kmed_cluster_image = kmed_cluster_image
        self.slic_cluster_image = slic_custer_image

        # Final data cube that should be used for next purposes
        final_data = np.zeros((self.k_n_clusters, self.B))
        for i in range(self.k_n_clusters):
            if self.k_method == "kmeans_mini_batch" or self.k_method == "kmeans":
                final_data[i] = medoids_indices[i]
            else:
                final_data[i] = superpixel_signatures_cube[medoids_indices[i]]

        return final_data, slic_custer_image, kmed_cluster_image

    def reconstruct(self, data):
        """
        Reconstructs the cube shape by assigning the data to the kmedoids clusters.
        Returns:
            ndarray: The reconstructed cube with the cluster spectras.
        """
        reconstructed_cube = np.zeros((self.H, self.W, self.B))

        for i, s in enumerate(np.unique(self.kmed_cluster_image)):
            mask_ = self.kmed_cluster_image == s
            if self.first_mask_is_bg and i == 0:
                continue
            reconstructed_cube[mask_] = data[i]
        return reconstructed_cube
