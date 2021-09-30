# DCT-SNN ICCV 2021

#### Code for the paper titled DCT-SNN: Using DCT to Distribute Spatial Information over Time for Learning Low-Latency Spiking Neural Networks ###

https://arxiv.org/abs/2010.01795

The paper has been accepted to ICCV 2021.

In this projet, a new encoding scheme for SNNs is proposed, where the analog pixel values are represented over time through DCT based modulated by the correspoding coefficients.

We first train an ANN, if ANN training is intended, that can be done using
vgg_ann_submit.py file which loads the models from vgg_ann_models_submit

After training an ANN, subsequent SNN training can be done using main_cifar10_submit 
(for cifar10), main_cifar100_submit (for cifar100) and main_ti_submit (for tinyImagenet)
files which load their corresponding model files. The snn model files include the encoding
part.

SNN training loads a pretrained ANN, we include a sample ANN for cifar10,
we also include DCT-SNN trained models for cifar10, cifar100 and tinyImagenet.
These models are available at-

https://www.dropbox.com/sh/aroe6p16gcb2iwj/AACJkMZtwF0w6s9hZ6XyKQ5Wa?dl=0

Before SNN training, we compute the layerwise thresholds using find_threshold function,
but once computed, we can save them and use for later/testing purposes. If the user wants
to compute the thresholds, the pre-computed ones must be commented and the following needs
to uncommented-
if pretrained and find_thesholds:
    find_threshold(ann_thresholds, train_loader1)

Note, to train/test, the corresponding directories of datasets/pre-trained models need to
be changed.
