# Changelog
All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com).


## 0.0.5 - 2024-10-23
### Added
- A new section called `distance_metrics` that includes several spectral metrics to compute similarity between signatures.
  - SAM: Spectral Angle Mapper
  - SID: Spectral Information Divergence
  - SID_TAN_SAM: Computes SAM and SID and combines with a tangent function.
  - SID_SIN_SAM: Computes SAM and SID and combines with a sin function.
  - JM_BC: Jeffries-Matusita distance (JM) using the Bhattacharyya Coefficient (BC)
### Changed
- Included `examples/example.ipynb` a distance_metrics part to verify the working of the new metrics.
- Improved `README.md` to include the new section and the new metrics as well as the new processing blocks from 0.0.4 version.

## 0.0.4 - 2024-10-22
### Added
- New processing blocks: ProcessSmoothSpectral and ProcessSmoothSpatial. These blocks allow to apply a smoothing filter to the spectral or spatial dimensions of the data. Check each block documentation for more details on the available filters.
- New processing block: ProcessDerivate. This block can apply the derivative of N order to the spectral axis of data. The user can select the order of the derivative.
- New processing block: ProcessInterpolate. This can be used to generate new spectral bands by interpolating the existing ones. The user can select the the interpolation ratio (eg. 2x, 3x, etc).
- New processing block: ProcessAddNoise. This block can add noise to the data. The user selects the noise level and applies gaussian noise to the data in the spectral dimension.
- Equalization trackbar in the visualization tool. The user can now adjust the equalization of the visualization using a trackbar.
### Changed
- Improved visualization of verbose messages in the processing blocks.
- Improved `examples/example.ipynb` to include the new processing blocks and guide the user on how to use them.
- Improved performance by reducing unnecessary memory allocation in the processing blocks and toolchain logic.

## 0.0.3 - 2024-10-11
### Added
- Visualization tool for the hyperspectral data. Example available in examples folder. It supports:
  - Navigation through the spectral bands
  - Pixel/Region selection and visualization
  - Spectral signature visualization
  - Zoom in/out
  - Save the image
  - Save the spectral signature
  - Do basic operations with bands (add, subtract, multiply, divide)
  - Toggle histogram equalization for the visualization

## 0.0.2 - 2024-10-10
Initial release.