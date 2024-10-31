import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Any
from sklearn.decomposition import PCA, NMF
from .utils import *


class GenericPipelineBlock(ABC):
    """Base class for all the preprocessing blocks"""

    def __init__(self, type, priority=-1) -> None:
        """Constructor
        Args:
            type (str): Type of the block, can be "Mask" or "Processor"
                If type is "Mask" the block must return a mask of the same shape of the input data.
                If type is "Processor" the block must return a processed version of the input data.
            priority (int, optional): The priority of the block. Defaults to -1.
        """
        self.type = type
        self.priority = priority

    @abstractmethod
    def __call__(self, data) -> np.ndarray:
        pass

    def __str__(self) -> str:
        name = self.__class__.__name__
        # remove the "Mask" or "Process" prefix
        name = name.removeprefix("Mask").removeprefix("Process")
        # remove the "Values"/"Signature"/"Detection" if it is present
        name = name.removesuffix("Values")
        name = name.removesuffix("Signature")
        name = name.removesuffix("Detection")

        return name

    def _apply_operation_to_masked(self, data, mask, operation):
        """
        Apply a specified operation to the masked elements in the data array.

        Args:
            data (np.ndarray): The input data array.
            mask (np.ndarray, dtype=bool): The boolean mask array.
            operation (function): The operation to apply to the masked elements.

        Returns:
        - np.ndarray: The result array with the same shape as the input. Except for the reduced dimensionality operations,
            in which case the shape is (data.shape[0], data.shape[1], n_components)
        """
        result = np.zeros_like(data)
        # For the ReduceDimensionality operations, we need to create a new array with the new third dimension
        if operation.__class__.__name__ == "ProcessReduceDimensionality":
            aux = operation(data[mask])
            result = np.zeros((data.shape[0], data.shape[1], aux.shape[1]))
            result[mask] = aux
        else:
            result[mask] = operation(data[mask])
        return result


class MaskNegativeValues(GenericPipelineBlock):
    def __init__(self) -> None:
        """This block creates a mask for the negative values in the data"""
        super().__init__(type="Mask")

    def __call__(self, data) -> np.ndarray:
        return np.invert(np.any(data < 0, axis=2))


class MaskZeroValues(GenericPipelineBlock):
    def __init__(self) -> None:
        """This block creates a mask for the zero values in the data"""
        super().__init__(type="Mask")

    def __call__(self, data) -> np.ndarray:
        return np.invert(np.any(data == 0, axis=2))


class MaskSaturatedValues(GenericPipelineBlock):
    def __init__(self, percentil: int = 99.5) -> None:
        """This block creates a mask for the saturated values in the data.
        It uses a percentil to determine the threshold for saturation detection in the data.

        Args:
            percentil (int, optional): Percentil from which data is marked as saturated. Defaults to 99.5.

        Raises:
            ValueError: If the percentil is not between 0 and 100
        """
        super().__init__(type="Mask")
        if percentil > 100 or percentil < 0:
            raise ValueError("Percentil must be between 0 and 100")
        self.percentil = percentil

    def __call__(self, data) -> np.ndarray:
        p_limit_value = np.percentile(data, self.percentil)
        return np.invert(np.any(data > p_limit_value, axis=2))


class MaskRxAnomalyDetection(GenericPipelineBlock):
    """
    Local Sparsity Divergence for Hyperspectral Anomaly Detection.
    The basic idea is that we asume that the normal target and the anomalous targets are located in different subspaces.
    We can compute an index to measure the divergence of the pixel from the normal subspace. The RX algorithm is based on
    the Mahalanobis distance, which is a measure of the distance between a point and the center of a distribution.

    """

    def __init__(self, threshold: float = 0.1) -> None:
        """Anomaly detection using the RX algorithm, partially based on the PySptools implementation

        Args:
            threshold (float, optional): Threshold to create the mask. Defaults to 0.1.
        """
        super().__init__(type="Mask")
        self.threshold = threshold

    def _produce_rx_map(self, data: np.ndarray) -> np.ndarray:
        orig_data = data.copy()
        h, w, b = data.shape

        data = data.reshape(-1, data.shape[-1]).T
        mean = np.average(data, axis=1)
        cov = np.cov(data)
        cov_inv = np.linalg.inv(cov)

        orig_data = orig_data - mean
        data = orig_data.reshape(-1, orig_data.shape[-1])

        A = data.dot(cov_inv)
        r = np.einsum("ij,ij->i", A, data)
        out = r.reshape(h, w)
        return out

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        This method implements the RX anomaly detection algorithm
        Partially based on: https://github.com/spectralpython/spectral/blob/0.23.1/spectral/algorithms/detectors.py
        """
        orig_data = data.copy()
        h, w, b = data.shape

        data = data.reshape(-1, data.shape[-1]).T
        mean = np.average(data, axis=1)
        cov = np.cov(data)
        cov_inv = np.linalg.inv(cov)

        orig_data = orig_data - mean
        data = orig_data.reshape(-1, orig_data.shape[-1])

        A = data.dot(cov_inv)
        r = np.einsum("ij,ij->i", A, data)
        out = r.reshape(h, w)
        out /= np.max(out)
        return np.invert(out > self.threshold)


# BLOQUES DE MÁSCARA
class MaskTargetSignature(GenericPipelineBlock):
    def __init__(
        self,
        method: str = "SAM",
        signature: np.ndarray = None,
        threshold: float = 0.1,
        select_area=True,
        interactive: bool = True,
    ) -> None:
        """
        This block creates a mask for a specific spectral signature in the data

        Args:
            method (str): The method to use to create the mask. Can be "SAM"(Spectral Angle Mapper) or "SC"(Spectral Correlation)
            signature (np.ndarray): The spectral signature to mask. If None, a window to select the signature will be shown.
            threshold (float, optional): The threshold to create the mask. Defaults to 0.1.
            select_area (bool, optional): Whether to select the area. Defaults to True.
        """
        super().__init__(type="Mask")
        self.method = method
        self.signature = signature
        self.threshold = threshold
        self.select_area = select_area
        self.interactive = interactive
        if method not in ["SAM", "SC"]:
            raise ValueError("Method must be SAM or SC")

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        This method creates a mask for the specific spectral signature in the data
        """
        self.data = data
        method = SAM if self.method == "SAM" else SC
        if self.signature is None:
            if self.interactive:
                ihyp = InteractiveHyperspectralImageProcessor(data, method, select_area=self.select_area)
                mask = ihyp.run()
            else:
                if self.select_area:
                    selector = HyperspectralAreaSelector()
                    self.signature = selector(data)
                    self.signature = np.mean(self.signature, axis=(0, 1))
                else:
                    selector = HyperspectralPixelSelector()
                    self.signature = selector(data)

                sam_angles, mask = method(data, self.signature, self.threshold)
        return np.array(mask, dtype=bool)



# BLOQUES DE PROCESAMIENTO: NORMALIZACIÓN
class ProcessNormalize(GenericPipelineBlock):
    def __init__(
        self,
        priority=-1,
        type="MinMax",
    ) -> None:
        """
        This block normalizes the data in the range [0, 1]
        Args:
            type (str): The type of normalization to use. Can be "MinMax", "MeanStd", "SNV" or "Helicoid"
        """
        super().__init__(type="Processor", priority=priority)
        self.norm_type = type

    def _min_max_normalize(self, data) -> np.ndarray:
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def _mean_std_normalize(self, data) -> np.ndarray:
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std

    def _snv_normalize(self, data) -> np.ndarray:
        # Calculate the mean and standard deviation along the last axis (bands) for each pixel
        mean_values = np.mean(data, axis=1, keepdims=True)
        std_values = np.std(data, axis=1, keepdims=True)

        # Handle zero standard deviation by setting those values to 1 to avoid division by zero
        std_values[std_values == 0] = 1

        # Apply SNV normalization
        snv_cube = (data - mean_values) / std_values

        return snv_cube

    def _helicoid_normalize(self, data) -> np.ndarray:
        # From Slimbrain data manager -> https://gitlab.citsem.upm.es/public-projects/hyperspectral/slim_brain_data_manager/-/blob/main/slimbrain_manager/processing/ximea_processing_chain.py
        pixBrightness = np.sqrt(np.sum(np.power(data, 2), axis=1) / np.shape(data)[2])

        imageHelNorm = data / pixBrightness[:, np.newaxis, np.newaxis]

        return imageHelNorm

    def __call__(self, data) -> np.ndarray:
        if self.norm_type.casefold() == "MinMax".casefold():
            return self._min_max_normalize(data)
        elif self.norm_type.casefold() == "MeanStd".casefold():
            return self._mean_std_normalize(data)
        elif self.norm_type.casefold() == "SNV".casefold():
            return self._snv_normalize(data)
        elif self.norm_type.casefold() == "Helicoid".casefold():
            return self._helicoid_normalize(data)
        else:
            print(f"Normalization type {self.norm_type} not recognized. Using MinMax normalization")
            return self._min_max_normalize(data)


# BLOQUES DE PROCESAMIENTO: DENOISING
class ProcessDenoise(GenericPipelineBlock):
    def __init__(self, priority=10) -> None:
        """
        This block uses the first steps of the HySIME algorithm to denoise the input image
        """
        super().__init__(type="Processor", priority=priority)

    def __call__(self, data) -> np.ndarray:

        def _est_additive_noise(r):
            """
            This function infers the noise in a hyperspectral data set, by assuming that the
            reflectance at a given band is well modelled by a linear regression on the remaining bands.
            Parameters:
                y: `numpy array`
                    a HSI cube ((m*n) x p)
            noise_type: `string [optional 'additive'|'poisson']`
            Returns: `tuple numpy array, numpy array`
                * the noise estimates for every pixel (N x p)
                * the noise correlation matrix estimates (p x p)
            Copyright:
                Jose Nascimento (zen@isel.pt) and Jose Bioucas-Dias (bioucas@lx.it.pt)
                For any comments contact the authors
            """
            small = 1e-6
            L, N = r.shape
            w = np.zeros((L, N), dtype=float)
            RR = np.dot(r, r.T)
            RRi = np.linalg.pinv(RR + small * np.eye(L))
            RRi = np.matrix(RRi)
            for i in range(L):
                XX = RRi - (RRi[:, i] * RRi[i, :]) / RRi[i, i]
                RRa = RR[:, i]
                RRa[i] = 0
                beta = np.dot(XX, RRa)
                beta[0, i] = 0
                w[i, :] = r[i, :] - np.dot(beta, r)
            Rw = np.diag(np.diag(np.dot(w, w.T) / N))
            return w, Rw

        y_cube = data
        y_cube = y_cube.reshape(-1, y_cube.shape[-1])
        L, N = y_cube.shape
        noise_estimate, noise_corr_value = _est_additive_noise(y_cube.T)

        noiseless_cube = y_cube - noise_estimate.T
        return noiseless_cube.reshape(data.shape), noise_corr_value


# BLOQUES DE PROCESAMIENTO: REDUCCIÓN DE DIMENSIONALIDAD (PCA, NMF)
class ProcessReduceDimensionality(GenericPipelineBlock):
    def __init__(
        self, method: str = "PCA", n_components: int = None, explained_variance_threshold: int = 85, transform: bool = False, plot: bool = False
    ) -> None:
        """
        This block reduces the dimensionality of the data using PCA or NMF.
        Args:
            method (str): The method to use to reduce the dimensionality. Can be "PCA" or "NMF"
            n_components (int): The number of components to keep. If n_components is None, an analysis
                of the explained variance will be performed. Selecting the number of components to keep
                based on the explained variance until the threshold is reached.
            explained_variance_threshold (int): The threshold of explained variance to keep the components
            transform (bool): If True, the method will return the transformed data. If False, the method will return the cube with the selected bands.
            plot (bool): If True, the explained variance plot will be shown. Only in the case that n_components is None.
        """
        super().__init__(type="Processor")
        self.method = method
        self.n_components = n_components
        self.explained_variance_threshold = explained_variance_threshold
        self.transform = transform
        self.plot = plot
        if method == "NMF":
            raise ValueError("NMF disabled for now")

    def __call__(self, data) -> np.ndarray:
        if self.n_components is None:
            if self.method.casefold() == "PCA".casefold():
                pca = PCA()
                pca.fit(data)
                cumsum = np.cumsum(pca.explained_variance_ratio_) * 100
                most_important_bands = np.argsort(np.abs(pca.components_), axis=1)[:, -1]
                n_components = np.argmax(cumsum >= self.explained_variance_threshold) + 1
                self.n_components = n_components
            elif self.method.casefold() == "NMF".casefold():
                nmf = NMF()
                nmf.fit(data)
                cumsum = np.cumsum(nmf.explained_variance_ratio_) * 100
                most_important_bands = np.argsort(np.abs(nmf.components_), axis=1)[:, -1]
                n_components = np.argmax(cumsum >= self.explained_variance_threshold) + 1
                self.n_components = n_components
            else:
                raise ValueError("Method not recognized. Use PCA or NMF")
            print(f"            + Number of components selected: {self.n_components}")
            if self.plot:
                plt.plot(cumsum, marker="o")
                plt.grid()
                plt.xlabel("Number of components")
                plt.ylabel("Explained variance")
                plt.title(f"Explained variance for {self.method}")
                plt.ylim(0, 100)
                plt.show()
        if self.method.casefold() == "PCA".casefold():
            pca = PCA(n_components=self.n_components)
            pca.fit(data)
            # Get the selected bands
            selected_bands = np.argsort(np.abs(pca.components_), axis=1)[:, -1]
            print(f"            + Most important bands index: {selected_bands}")
            if self.transform:
                return pca.transform(data)
            else:
                # Return the cube with the selected bands
                return data[:, selected_bands]
        elif self.method.casefold() == "NMF".casefold():
            nmf = NMF(n_components=self.n_components)
            nmf.fit(data)
            selected_bands = np.argsort(np.abs(nmf.components_), axis=1)[:, -1]
            if self.transform:
                return nmf.transform(data)
            else:
                # Return the cube with the selected bands
                return data[:, selected_bands]
        else:
            raise ValueError("Method not recognized. Use PCA or NMF")


class ProcessSLIC(GenericPipelineBlock):
    def __init__(self, n_segments:int=100, compactness=10, max_iter:int=10, sigma=0, reduction_method:str="median") -> None:
        """
        This block uses the SLIC algorithm to segment the hyperspectral image into superpixels
        Args:
            n_segments (int): The number of segments to create
            compactness (int): Balances the color (spectral) proximity and space proximity. Higher values give more weight to space proximity.
            max_iter (int): The maximum number of iterations for the algorithm
            sigma (int): Width of Gaussian smoothing kernel for preprocessing the image before segmenting.
            reduction_method (str): The method to reduce the superpixel to a single value. Can be "median" or "mean" or "random".
        """
        super().__init__(type="Processor")
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.sigma = sigma
        self.reduction_method = reduction_method
        
        valid_reduction_methods = ["median", "mean", "random"]
        if reduction_method not in valid_reduction_methods:
            raise ValueError(f"Reduction method must be one of {valid_reduction_methods}")
        
        super().__init__(type="Processor")
