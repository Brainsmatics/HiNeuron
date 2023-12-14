# HiNeuron
Reconstructing neuronal morphology is vital for classifying neurons and mapping brain connectivity. However, it remains a significant challenge due to their complex structure, dense distribution, and low image contrast. 
This program is used for axial interpolation of neuron images to obtain isotropic images. 
The voxel resolution in the three directions of the xyz is different, and the voxel resolution in the z-axis is lower. 
Therefore, a high voxel resolution image in the z-axis is obtained through neural network prediction.

Data and code availability are publicly available as of the date of publication.
If you use this dataset for your research, please cite our paper.
Any additional information required to reanalyze the data reported in this paper is available from the lead contact upon request.

The HiNeuron event has been deposited in Zenodo under http://doi.org/10.5281/zenodo.10260457 and is publicly available as of the date of publication. 

Requisites

Python 3.6 or newer

· numpy

· opencv

· keras

·	os

·	skimage

·	time

This program is trained using train.py and then predicted using predict.py. 

The training set is placed in the image folder.
