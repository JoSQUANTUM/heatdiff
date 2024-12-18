# Heatdiff

The aim of this initial release is to demonstrate how the heat semigroup can be used as a lossy compression tool for image processing, in particular, for image corruption and restoration.  

We also demonstrate that the heat semigroup can be used for image compression, via its use as a kernel in a weighted K-Means algorithm.

In addition, we will demonstrate that explicit diffusion equations can also be used as lossy compression for images. One can view this method as a 'learning free' denoising diffusion model. We replicate the experiments for the heat semigroup with diffusion equations, using their appropriate analogues. 

In the future, we aim to investigate further topics such as: regularised image restoration; lossless compression; and the integration of machine learning tools amongst others.

# Installation

```bash
pip install heatdiff
```

# Examples

The following notebooks demonstrate the most relevant features:

- [Heat Semigroup](notebooks/semigroup_demo.ipynb)
- [Diffusion Process](notebooks/diffusion_demo.ipynb)

