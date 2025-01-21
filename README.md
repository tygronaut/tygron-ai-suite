# Tygron AI Suite
This is a simple package for training Mask-RCNN networks using datasets exported from the Tygron Platform.

It has a configuration class that stores the settings for:
* The train and test dataset location
* Mask RCNN model parameters:
  * Channels, 
* Settings for exporting the PyTorch model to ONNX
* Metadata that is interpreted by the Tygron Platform, such as:
  * Attributes
  * Prefered pixel sizes
  * Description, producer and version
  * Legend entries

It also has a dataset class that simplifies training based on the data generated from projects based in the Tygron Platform.
