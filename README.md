# CNN Multi Echo T2 and Spinal Cord Segmentation
Brain and Spinal Cord Segmentation using a Convolutional Neural Network based on LinkNet

Datasets:
- Single Echo T1-weighted brain data from 3DSAG sequence (Native image size: 160x256x256)
- Multi-Echo T2-weighted brain data from GRASE sequence (Native image size: 240x240x40x32)
- Spinal Cord data from FFE sequence (Native image size: 256x256x16)

Models:
**[LinkNet Architecture](https://arxiv.org/abs/1707.03718)**; **[Full Pre-Activation ResNet18](https://arxiv.org/abs/1603.05027)** was used as Encoder

## Pre-Processing
Neural network input images were 256x256. For brain segmentation, all brains were extracted first using FSL `bet` command. For image sizes less than 256x256 zero padding at the beginning and cropping at the end was done or the other way around for image sizes greater than 256x256. (**Resizing or scaling instead of cropping might be better but I ended up using crop because after brain extraction, a lot of blank space is left. Spinal cord data should use resizing or scaling.**)

Everything was done using [Jupyter Notebook](jupyter.org/). `nibabel.load()` was used to import Nifti formatted images into the notebooks and `nibabel.load().get_data` to convert to Numpy arrays. Normalization was then done on images:
- For T1 brain: values divided by gloval maximum (**Maybe dividing by maximum of each slice would be better**)
- For T2 brain: each echo was treated independently and each slice was treated independently, so values were divided by the maximum value of each slice for every echo.
- For spinal cord: values divided by global maximum (**Maybe dividing by maximum of each slice would be better**)

All images' dimensions were altered using `numpy.swapaxes()` and `numpy.expand_dims()` to conform to Tensorflow's input shape (Nx256x256xC where N is number of slices and C is number of channels (1 for T1 brain, 32 for T2 brain, 1 for spinal cord)).

For spinal cord segmentation only, cropping was necessary to obtain a decent segmentation result:
1. The centerline of the spinal cord was determined using [Spinal Cord Toolbox](https://sourceforge.net/projects/spinalcordtoolbox/)'s command `sct_get_centerline`
2. A 128x128 crop having the centerpoint from the centerline in the middle by Python indexing was made.
3. The cropped image was upscaled using `skimage.transform.rescale()` to 512x512.
4. The neural network crops it back to 256x256 (**This step is unecessary and information may be lost in cropping the already small image. Need to change upscale to just 256x256 instead**)

Resizing is done after feeding through neural network to get the segmentation mask back to original size; more on that later.

## Ground Truth Labels
### T1-Weighted Brain

### T2-Weighted Brain

### Spinal Cord
