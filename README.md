# Tygron AI Suite
This is a simple package for training Mask R-CNN networks using datasets exported from the [Tygron Platform](www.tygron.com).


# Contents
This repository contains a few simple classes and functions for training a simple Mask R-CNN for object detection on satellite images. Additionally it contains functions to export the model to [ONNX](https://onnx.ai/) with parameters and metadata such that it can be imported easily into the Tygron Platform.

The Configuration class stores the settings for:
* The train and test dataset location
* Mask RCNN model parameters:
 * Channels
* Settings for exporting the PyTorch model to ONNX
* Metadata that is interpreted by the Tygron Platform, such as:
  * Attributes
  * Prefered pixel sizes
  * Description, producer and version
  * Legend entries

It also has a dataset class that simplifies training based on the data generated from projects based in the Tygron Platform.

# Installation
## Python
[Download](https://www.python.org/downloads/) and Install Python 3.11 or higher.

## Anaconda
[Download](https://www.anaconda.com/download/) anaconda and install it to access conda and the anaconda-navigator.

## Conda

Once both are installed, create a new conda environment, for example '''tygronai'''.

### Create conda enviroment
```
conda create --name tygronai python=3.11
```
Once create, activate itActivate it
```
conda activate tygronai
````
### Install pytorch and torchvision with cuda:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
For alternative versions of pytorch, such as the cpu only version, please refer to the [official instructions](https://pytorch.org/get-started/locally/)

### Install onnx and onnxruntime
```
conda install onnx onnxruntime
```
### Install onnxscript
```
pip install onnxscript
```
### Install anaconda-navigator
conda install anaconda-navigator

To activate the environment and start anaconda-navigator using a simple shell script file, create one, for example named '''anaconda.sh''', with the following commands:
```
conda init 
conda activate tygronai
anaconda-navigator
```
Call the shell script using ```bash -i anaconda.sh```

