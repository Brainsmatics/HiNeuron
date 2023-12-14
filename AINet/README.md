# AINet
This program is used for axial interpolation of neuron images to obtain isotropic images. The voxel resolution in the three directions of the xyz is different, and the voxel resolution in the z-axis is lower. Therefore, a high voxel resolution image in the z-axis is obtained through neural network prediction.

When using this program to train a network, you need to place two folders in the image folder, one is a low resolution image and the other is a ground truth image for learning. The training results will be saved in the checkpoint folder.

Requisites

Python 3.6 or newer

· numpy

· opencv

· keras

·	os

·	skimage

·	time

This program is trained using train.py and then predicted using predict.py. 

If using the program, it needs to create three new folders: image, checkpoint, and log. The training images are placed in the image folder, and the training results are saved in the checkpoint and log folders.

