# CNN Multi Echo T2 and Spinal Cord Segmentation
Brain and Spinal Cord Segmentation using a Convolutional Neural Network based on LinkNet

Datasets:
- Single Echo T1-weighted brain data from 3DSAG sequence (Native image size: 160x256x256)
- Multi-Echo T2-weighted brain data from GRASE sequence (Native image size: 240x240x40x32)
- Spinal Cord data from FFE sequence (Native image size: 256x256x16)

Models:
**[LinkNet Architecture](https://arxiv.org/abs/1707.03718)**; **[Full Pre-Activation ResNet18](https://arxiv.org/abs/1603.05027)** was used as Encoder

## Pre-Processing
Neural network input images were 2D 256x256. For brain segmentation, all brains were extracted first using FSL `bet` command. For image sizes less than 256x256 zero padding at the beginning and cropping at the end was done or the other way around for image sizes greater than 256x256. (**Resizing or scaling instead of cropping might be better but I ended up using crop because after brain extraction, a lot of blank space is left. Spinal cord data should use resizing or scaling.**)

Everything was done in [Jupyter Notebook](jupyter.org/). Handling of MRI images was done using [Nibabel](http://nipy.org/nibabel/): `nibabel.load()` was used to import Nifti formatted images into the notebooks and `nibabel.load().get_data` to convert to Numpy arrays. Normalization was then done on images:
- For T1 brain: values divided by global maximum (**Maybe dividing by maximum of each slice would be better**)
- For T2 brain: each echo was treated independently and each slice was treated independently, so values were divided by the maximum value of each slice for every echo.
- For spinal cord: values divided by global maximum (**Maybe dividing by maximum of each slice would be better**)

All images' dimensions were altered using `numpy.swapaxes()` and `numpy.expand_dims()` to conform to [Tensorflow](https://www.tensorflow.org/)'s input shape (Nx256x256xC where N is number of slices and C is number of channels (1 for T1 brain, 32 for T2 brain, 1 for spinal cord)).

For T1 and T2 models, binarizing ground truths is necessary because ground truths will have 3 classes.

For spinal cord segmentation only, cropping was necessary to obtain a decent segmentation result:
1. The centerline of the spinal cord was determined using [Spinal Cord Toolbox](https://sourceforge.net/projects/spinalcordtoolbox/)'s command `sct_get_centerline`
2. A 128x128 crop having the centerpoint from the centerline in the middle was done by Python indexing.
3. The cropped image was upscaled using `skimage.transform.rescale()` to 512x512.
4. The neural network crops it back to 256x256 (**This step is unecessary and information may be lost in cropping the already small image. Need to change upscale to just 256x256 instead**)

Resizing is done after feeding through neural network to get the segmentation mask back to original size; more on that later.

## Ground Truth Labels
### T1-Weighted Brain
Ground truths made from FSL FAST segmentation using command `fast -t 1 -n 3 -H 0.15 filename` in terminal
### T2-Weighted Brain
Registered ground truths from T1-weighted images onto T2 space:
```
flirt -in GRASE_bet.nii.gz -ref T1_bet.nii.gz -omat T2toT1.mat
convert_xfm -omat T1toT2.mat -inverse T2toT1.mat
flirt -in T1_bet_seg.nii.gz -ref GRASE_bet.nii.gz -applyxfm -init T1toT2.mat -out T2_bet_regseg.nii.gz
```
1. Create linear transform mask from T2 to T1 because T1 has higher resolution
2. Inverse the transform to get T1 to T2
3. Apply transform on T1 ground truth segmentation using inverse transform mask and T2 as reference
### Spinal Cord
Manual segmentation by hand using FSLeyes

## Training
[Keras](https://keras.io/) was used for everything (making the model, training, testing, etc.). To compile:
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Metrics: Accuracy

### T1-Weighted Model
Simple training (No data augmentation). Trained on 7 brains (1120 training examples). 7 epochs, quick and simple
### T2-Weighted Model
Transfered model from T1 but weights for the first convolution were mismatched because of the channels (1 vs 32). So before training for the first time, the weights for this first convolution layer was repeated to a size of 32.

Used data augmentation with settings:
```
data_gen_args = dict(rotation_range=180.,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True)
```
Trained on 38 brains (1520 training examples). Trained for about 15 epochs with 500 iterations each.
### Spinal Cord Model
Trained spinal cord model the exact same way as T2 model. Inputs were the cropped images, network predictions were also cropped. Data augmentation parameters/arguments were exactly the same as T2 model. Trained on 7 samples (112 training examples). Trained for about 15 epochs with 200 iterations each.

## Testing
keras `model.evaluate()` used:
- T1: tested on 1 brain
- T2: tested on 11 brains
- Spinal Cord: tested on 1 spinal cord dataset

## Results
### T1-Weighted Model

| Training Loss | Training Accuracy | Testing Loss | Testing Accuracy | DSC |
| :---: | :---: | :---: | :---: | :---: | 
| 0.0126 | 0.9948| 0.0239 | 0.9903 | 0.8873 |

![testc11_slice150](https://user-images.githubusercontent.com/28941980/44606791-307ff480-a7a3-11e8-882f-de40703ef163.png)

### T2-Weighted Model

| Training Loss | Training Accuracy | Testing Loss | Testing Accuracy | DSC |
| :---: | :---: | :---: | :---: | :---: | 
| 0.0386 | 0.9660| 0.0405 | 0.9837 | 0.8495 |

![t2_2](https://user-images.githubusercontent.com/28941980/44606895-86ed3300-a7a3-11e8-89c6-9f5eb57b2e14.png)

### Spinal Cord Model

| Training Loss | Training Accuracy | Testing Loss | Testing Accuracy | DSC |
| :---: | :---: | :---: | :---: | :---: | 
| 0.0553 | 0.9936| 0.0128 | 0.9942 | 0.6332 |

![sc3](https://user-images.githubusercontent.com/28941980/44606857-67eea100-a7a3-11e8-80ad-9a4fde332f9a.png)

A comparison of Dice Coefficient was conducted between the sets of the ground truths with the neural network and the ground truths with Spinal Cord Toolbox segmentation (first run `sct_propseg` to get spinal cord segmentation then run `sct_segment_graymattter` for gray matter segmentation):

| DSC | Spinal Cord Toolbox | Neural Network Prediction |
| :---: | :---: | :---: |
| **Ground Truths** | 0.3831 | 0.6332 |

![sctvsnn2](https://user-images.githubusercontent.com/28941980/44606852-62915680-a7a3-11e8-9e5b-66405b3904d1.png)

## Post-Processing
- For T1, no post-processing needed because native image is already 256x256
- For T2, since zero-padding is done at beginning to get from 240x240 to 256x256, cropping is done after to go back to 240x240
- For spinal cord, resizing was done by first creating a new array of zeros with the original image size then resizing the predictions using `skimage.transform.rescale()`, and finally, using numpy indexing to replace zeros with resized predictions using crops from pre-processing. 

## Summary (Workflow)
| T1 / T2 Model | Spinal Cord Model |
| :---: | :---: |
| ![screen shot 2018-08-24 at 3 00 18 pm](https://user-images.githubusercontent.com/28941980/44609894-7e4e2a00-a7ae-11e8-89b0-56246255e915.png) | ![screen shot 2018-08-24 at 2 55 55 pm](https://user-images.githubusercontent.com/28941980/44609750-e51f1380-a7ad-11e8-81b4-877f8475be78.png) |

## Future Direction
- [x] Change crop and zoom and resizing from keras (Changed to using `skimage.transform.rescale` for zoom and resize and numpy indexing for crop)
- [ ] Reduce layers/operations and pick out only the important ones so no repeats
- [ ] Compare and test with other networks/algorithms
- [ ] Multi-class segmentation (white matter, gray matter, cerebrospinal fluid)
- [ ] Lesion detection and segmentation
