# CNN Multi Echo T2 and Spinal Cord Segmentation
Brain and Spinal Cord Segmentation using Convolutional Neural Network based on LinkNet

Datasets:
- Multi-Echo T2-weighted brain data from GRASE sequence (Native image size: 240x240x40)
- Spinal Cord data from FFE sequence

Models:
**[LinkNet Architecture](https://arxiv.org/abs/1707.03718)**; **[Full Pre-Activation ResNet18](https://arxiv.org/abs/1603.05027)** was used as Encoder

## Pre-Processing
Neural network input images were 256x256. For T2-weighted brain segmentation, all brains are extracted first. Zero padding at the beginning and crop at the end for images less than 256x256 or other way around for images greater than 256x256. 
- Resizing or scaling instead of cropping might be better but I ended up using crop because after brain extraction, a lot of blank space is left. Spinal cord data should use resizing or scaling.

Everything was done using [Jupyter Notebook](jupyter.org/). `nibabel.load()` was used to import Nifti formatted images into the notebooks and `nibabel.load().get_data` to convert to Numpy arrays. Normalization was then done on images:
- For T2 brain: each echo was treated independently and each slice was treated independently, so values were divided by the maximum value of each slice for every echo.
- For spinal cord: values divided by global maximum (**Mistake!! Should be divide by maximum of each slice**)

All images' dimensions were altered using `numpy.swapaxes()` and `numpy.expand_dims()` to conform to Tensorflow's input shape (Nx256x256xC where N is number of slices and C is number of channels (32 for brain, 1 for spinal cord)).

