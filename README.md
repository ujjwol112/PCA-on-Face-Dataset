# PCA on Face Dataset
 Insights on Face Dataset using PCA

This repository demonstrates the application of Principal Component Analysis (PCA) on a face dataset for dimensionality reduction, eigenface generation, image reconstruction, and visualization. PCA is a technique commonly used for reducing the dimensionality of data while preserving its essential structure. In this project, PCA is applied to a dataset of facial images to extract the most significant features, known as eigenfaces, and explore various aspects of facial image analysis.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Image Preprocessing](#image-preprocessing)
  - [Eigenface Generation](#eigenface-generation)
  - [Face Representation](#face-representation)
  - [Principle Components Visualization](#principle-components-visualization)
  - [Image Reconstruction](#image-reconstruction)
- [Results](#results)
- [References](#references)

## Introduction

Principal Component Analysis (PCA) is a widely used technique in data analysis and computer vision for dimensionality reduction. It helps in identifying patterns in data by transforming it into a new coordinate system such that the greatest variance lies along the first coordinate (principal component), the second greatest variance along the second coordinate, and so on. In facial recognition and image processing, PCA is particularly useful for tasks like feature extraction, face recognition, and image reconstruction.

## Dependencies

To run the code in this repository, the following dependencies are required:
- Python 3.x
- PIL (Python Imaging Library)
- NumPy
- Matplotlib
- scikit-learn (for comparison)

You can install the dependencies using `pip`:
```bash
pip install pillow numpy matplotlib scikit-learn
```

## Usage

### Image Preprocessing

The image preprocessing step involves loading facial images, converting them to grayscale, enhancing contrast, resizing, and saving the processed images to an output folder.

### Eigenface Generation

Eigenfaces are the principal components obtained from the covariance matrix of the preprocessed images. This section calculates eigenvalues and eigenvectors, sorts them in descending order, and visualizes the eigenfaces.

### Face Representation

Facial images are represented using the principal components obtained from the eigenface generation step. Visualization techniques are used to display the original images and the mean face image.

### Principle Components Visualization

The principal components obtained from PCA are visualized in 2D and 3D space to understand the distribution of facial images in the reduced feature space.

### Image Reconstruction

Facial images are reconstructed using a subset of principal components. Both the original and reconstructed images are displayed for comparison.

## Results

The results include visualizations of eigenfaces, principal components, and reconstructed images. Additionally, the cumulative explained variance is plotted to determine the optimal number of principal components for dimensionality reduction.

## References

- [PIL Documentation](https://pillow.readthedocs.io/en/stable/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
