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
[Download](https://www.python.org/downloads/) and Install Python 3.9 or higher.

## Pytorch
Please refer to the [officual instructions](https://pytorch.org/get-started/locally/) to install the stable versions of torch and torchvision on your system.
If possible, install it through conda or pip3 with Cuda 12.1

## ONNX
Install the ONNX libraries using pip or conda:
