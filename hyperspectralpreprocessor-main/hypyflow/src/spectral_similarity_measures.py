import numpy as np
import math


def SAM(data: np.ndarray, signature: np.ndarray, threshold: float):
    """
    Spectral Angle Mapper.
    Computes the angle between two spectra, the result is a value between 0 and pi/2m the smaller the value, the more similar the spectra are.
    Args:
        data (np.ndarray): The hyperspectral data cube. The shape of the data must be (n_rows, n_cols, n_bands)
        signature (np.ndarray): The spectral signature to compare. The shape of the signature must be (n_bands,)
        threshold (float): The threshold to create the mask
    Returns:
        np.ndarray: The image with the spectral angle values
        np.ndarray: The mask with the pixels that are similar to the signature
    """
    return_as_value = False
    if len(data.shape) == 1:
        data = data[np.newaxis, np.newaxis, :]
        return_as_value = True
    if len(data.shape) != 3:
        raise ValueError("Data must be a 3D array")
    spectral_angle = np.arccos(np.sum(signature * data, axis=2) / (np.linalg.norm(signature) * np.linalg.norm(data, axis=2)))
    threshold = threshold * (np.max(spectral_angle) - np.min(spectral_angle)) + np.min(spectral_angle)
    if return_as_value:
        return spectral_angle[0, 0], np.invert(spectral_angle[0, 0] < threshold)
    return spectral_angle, np.invert(spectral_angle < threshold)


def SID(data: np.ndarray, signature: np.ndarray, threshold: float):
    """
    Computes the Spectral Information Divergence (SID) between a target vector and each pixel in a hyperspectral cube (vectorized).

    Args:
        data: numpy array of shape (H, W, B), where H is height, W is width, and B is the number of spectral bands
        signature: numpy array of shape (B,), the target spectral signature

    Returns:
        sid_map: numpy array of shape (H, W), the SID for each pixel compared to the target
        mask: numpy array of shape (H, W), a boolean mask where values below the threshold are inverted
    """
    return_as_value = False
    if len(data.shape) == 1:
        data = data[np.newaxis, np.newaxis, :]
        return_as_value = True

    if len(data.shape) != 3:
        raise ValueError("Data must be a 3D array")

    # Reshape the cube to (H*W, B) for easier manipulation
    H, W, B = data.shape
    reshaped_cube = data.reshape(-1, B)

    # Normalize the spectral vectors (pixel spectra) and the target vector
    epsilon = 1e-10
    reshaped_cube = reshaped_cube / np.sum(reshaped_cube, axis=1, keepdims=True)
    signature = signature / np.sum(signature)

    # Clip values to avoid log(0) or division by zero
    reshaped_cube = np.clip(reshaped_cube, epsilon, None)
    signature = np.clip(signature, epsilon, None)

    # Compute KL divergence for both directions (cube vs target and target vs cube)
    kl_pq = np.sum(reshaped_cube * np.log(reshaped_cube / signature), axis=1)
    kl_qp = np.sum(signature * np.log(signature / reshaped_cube), axis=1)

    # Compute SID as the sum of both divergences
    sid = kl_pq + kl_qp

    # Reshape back to (H, W) to match the image size
    sid_map = sid.reshape(H, W)

    mask = np.invert(sid_map < threshold)

    if return_as_value:
        return sid_map[0, 0], mask[0, 0]
    return sid_map, mask


def SID_TAN_SAM(data: np.ndarray, signature: np.ndarray, threshold: float):
    """
    Computes the Spectral Information Divergence (SID) and Spectral Angle Mapper (SAM)
    for the given hyperspectral data and signature, and combines them using the tangent function.

    This variation comes from the paper: "New Hyperspectral Discrimination Measure for Spectral Characterization"
    Args:
        data (np.ndarray): The hyperspectral data array.
        signature (np.ndarray): The reference spectral signature.
        threshold (float): The threshold value for masking.
    Returns:
        sid_tan_sam (np.ndarray): The combined SID and SAM values using the tangent function.
        mask_sid_tan_sam (np.ndarray): A boolean mask where values below the threshold are inverted.
    """
    sid_map, sid_mask = SID(data, signature, threshold)
    sam_map, sam_mask = SAM(data, signature, threshold)

    sid_tan_sam = sid_map * np.tan(sam_map)

    mask_sid_tan_sam = np.invert(sid_tan_sam < threshold)

    return sid_tan_sam, mask_sid_tan_sam


def SID_SIN_SAM(data: np.ndarray, signature: np.ndarray, threshold: float):
    """
    Computes the Spectral Information Divergence (SID) and Spectral Angle Mapper (SAM)
    for the given hyperspectral data and signature, and combines them using the sin function.

    This variation comes from the paper: "New Hyperspectral Discrimination Measure for Spectral Characterization" (http://dx.doi.org/10.1117/1.1766301)
    Args:
        data (np.ndarray): The hyperspectral data array.
        signature (np.ndarray): The reference spectral signature.
        threshold (float): The threshold value for masking.
    Returns:
        sid_tan_sam (np.ndarray): The combined SID and SAM values using the sin function.
        mask_sid_tan_sam (np.ndarray): A boolean mask where values below the threshold are inverted.
    """

    sid_map, sid_mask = SID(data, signature, threshold)
    sam_map, sam_mask = SAM(data, signature, threshold)

    sid_sin_sam = sid_map * np.sin(sam_map)

    mask_sid_sin_sam = np.invert(sid_sin_sam < threshold)

    return sid_sin_sam, mask_sid_sin_sam


def JM_BC(data: np.ndarray, signature: np.ndarray, threshold: float):
    """
    Computes the Jeffries-Matusita distance (JM) using the Bhattacharyya Coefficient (BC), which measures the overlap between two probability distributions.

    The final distance scales the between 0 and 2. The higher the value, the more separability between the two distributions.

    This variation comes from the paper: "Jeffries-Matusita distance as a tool for feature selection" (https://doi.org/10.1109/ICDSE47409.2019.8971800)
    Args:
        data (np.ndarray): The hyperspectral data array.
        signature (np.ndarray): The reference spectral signature.
        threshold (float): The threshold value for masking.
    Returns:
        sid_tan_sam (np.ndarray): The combined SID and SAM values using the sin function.
        mask_sid_tan_sam (np.ndarray): A boolean mask where values below the threshold are inverted.
    """

    def BC(spectrum1: np.ndarray, spectrum2: np.ndarray):
        return np.sum(np.sqrt(spectrum1 * spectrum2), axis=-1)

    return_as_value = False
    # If data is 1D, reshape it to (1, 1, B) for consistency with 3D case
    if data.ndim == 1:
        data = data[np.newaxis, np.newaxis, :]
        return_as_value = True

    # Ensure data has 3 dimensions (H, W, B)
    if data.ndim != 3:
        raise ValueError("Data must be either 1D or 3D array with spectral axis as the last dimension.")

    # Check if the signature matches the spectral dimension
    if data.shape[-1] != len(signature):
        raise ValueError("The length of the spectral signature must match the spectral dimension of the data.")


    bc = BC(data, signature)
    jm = np.sqrt(2 * (1 - np.exp(-bc)))
    mask_jm = np.invert(jm < threshold)
    if return_as_value:
        return jm[0, 0], mask_jm[0, 0]
    return jm, mask_jm


if __name__ == "__main__":
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    target = np.array([2, 3, 2.3, 2, 2, 1, 2, 3])


    # Test all the methods
    print("Test with single spectrum data:")
    print(" - SAM:", SAM(data, target, 0.5))
    print(" - SID:", SID(data, target, 0.5))
    print(" - SID_TAN_SAM:", SID_TAN_SAM(data, target, 0.5))
    print(" - SID_SIN_SAM:", SID_SIN_SAM(data, target, 0.5))
    print(" - JM_BC:", JM_BC(data, target, 0.5))

    #now with a data matrix
    data = np.array([[[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]],[[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]])
    print(f"Test with hyperspectral cube data of shape {data.shape}:")
    print(" - SAM:", SAM(data, target, 0.5))
    print(" - SID:", SID(data, target, 0.5))
    print(" - SID_TAN_SAM:", SID_TAN_SAM(data, target, 0.5))
    print(" - SID_SIN_SAM:", SID_SIN_SAM(data, target, 0.5))
    print(" - JM_BC:", JM_BC(data, target, 0.5))