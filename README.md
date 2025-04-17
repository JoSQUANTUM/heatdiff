# heatdiff

Image processing algorithms based on the heat semigroup.

## Installation

```bash
pip install .
```

## Description

In this repository, we aim to demonstrate the application of the heat semigroup to a variety of image processing tasks such as

- A lossy compression tool for image processing, in particular, for image corruption and restoration (and it's stochastic analogue). One can conceptually view this method as a 'learning free' denoising diffusion model.  
See the following notebooks for more details:
    - [Heat Semigroup process](notebooks/semigroup_demo.ipynb)
    - [Diffusion Process](notebooks/diffusion_demo.ipynb)

- Image compression, via its use as a kernel in a weighted K-Means algorithm. See the above notebooks

- Image Segmentation, via the heat semigroup approximation of the Perimeter functional. See [Heat Semigroup Segmentation](notebooks/segmentation_demo.ipynb)

In the future, we aim to investigate further topics such as: 
- Regularised image restoration.

- The integration of machine learning tools/integration into machine learning pipelines. 

- Lossless compression.