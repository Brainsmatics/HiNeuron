import tensorflow as tf
import numpy as np
import cv2
import os
import string
import time
import matplotlib.pyplot as plt
from pylab import *
import sys
from data import generate_traindata_data_1
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Input, UpSampling2D, merge, PReLU, Add
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad
from keras.layers.advanced_activations import PReLU
from keras.layers.merge import Concatenate
from keras.callbacks import EarlyStopping
from keras.callbacks import *
from keras.layers import *
from keras.regularizers import l2
# from keras.applications.imagenet_utils import _obtain_input_shape
# from keras.utils.multi_gpu_utils import multi_gpu_model

from keras.models import Model
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
# from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
# from sklearn.cross_validation import train_test_split
# from lsuv_init import LSUVinit

import smtplib
from math import *
import time
import skimage.io as io
from keras.callbacks import ModelCheckpoint, TensorBoard

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
K.set_session(sess)


class CNNTrain:
    def __init__(self):

        self.CONV_NUM = 7  # Number of convolutional layers
        self.KERNEL = 3
        self.FILTER = 28
        self.SAMPLE = ''  #
        self.MOSAIC_LAYER = range(1500, 1500 + 1)
        self.MOSAIC_X = range(12, 24 + 1)
        self.MOSAIC_Y = range(20, 33 + 1)
        self.MOSAIC_Namelast = '.tif'
        self.MOSAIC_channel = 'CH1'
        self.pic_num = []

        self.TRAIN_LAYER = 1
        self.TRAIN_BATCH = []
        self.TRAIN_TIMES = 100
        self.TRAIN_REPEAT = 1
        self.MAX_SAMPLE = 100000
        self.PREDICT_REPEAT = 1
        self.BatchSize = 1  # batchsize
        self.Num_Data = 2000
        self.EARLY_STOP = 50
        self.SEQ_TRAIN = False
        self.predict_stride = 100
        self.output_length = 128
        self.xPATH = r'./Train_Data/'
        self.yPATH = r'./Train_Data/'
        self.BIT = 8
        self.RESULT_TYPE = 'same'
        self.input_folder = r'H:/test/'
        self.input_folder_Y = r'H:/test1/'
        self.output_folder = r'H:/test_predict/'
        self.FOLDER = []
        self.Save_Address = r'./' + '1/'
        self.IMAGE_LENGTH = 128  #
        self.cut = int((self.IMAGE_LENGTH - self.predict_stride) / 2)
        self.MOSAIC_LENGTH_X = 500  # image x_length
        self.MOSAIC_LENGTH_Y = 500  # image y_length
        self.MOSAIC_FILL_X = self.cut * 2
        self.MOSAIC_FILL_Y = self.cut * 2
        self.diff_input_output = 0
        self.in_shape = (1, self.IMAGE_LENGTH, self.IMAGE_LENGTH)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=self.EARLY_STOP, verbose=1)
        self.loss_list = []
        # ---------------------optimizer, don't change----------------------------
        self.o_adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.o_adagrad = Adagrad(lr=0.01, epsilon=1e-06)
        self.o_adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
        self.o_sgd = SGD(lr=0.01, momentum=0.9, decay=0.9, nesterov=True)
        self.DR_min = 0
        # self.DR_max = 65535
        self.DR_max = 255
        # self.mean = 4875.9
        self.mean = 100
        self.DR_Temp = 10000

        self.learning_rate = []
        self.load_model_path = []

        self.train_num = []
        self.test_num = []

        self.test_select = []






    def init_AINet_model_data(self):
        # 64*192
        inputs = Input(shape=(128, 128, 1))
        conv1 = Conv2D(64, 3, padding='same', kernel_initializer='glorot_normal')(inputs)
        conv1 = Conv2D(64, 3, padding='same', kernel_initializer='glorot_normal')(conv1)
        prelu1 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(
            conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(prelu1)
        # 64*80

        # 128*96
        conv2 = Conv2D(128, 3, padding='same', kernel_initializer='glorot_normal')(pool1)
        conv2 = Conv2D(128, 3, padding='same', kernel_initializer='glorot_normal')(conv2)
        prelu2 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(
            conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(prelu2)
        # 128*40

        # 256*48
        conv3 = Conv2D(256, 3, padding='same', kernel_initializer='glorot_normal')(pool2)
        conv3 = Conv2D(256, 3, padding='same', kernel_initializer='glorot_normal')(conv3)
        prelu3 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(
            conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(prelu3)
        # 256*24

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(conv4)
        prelu4 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(
            conv4)


        up7 = Conv2D(512, 2, padding='same', kernel_initializer='glorot_normal')(UpSampling2D(size=(2, 2))(prelu4))
        # 256*48
        merge7 = Concatenate(axis=3)([prelu3, up7])
        conv7 = Conv2D(256, 3, padding='same', kernel_initializer='glorot_normal')(merge7)
        conv7 = Conv2D(256, 3, padding='same', kernel_initializer='glorot_normal')(conv7)
        prelu7 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(
            conv7)
        # 256*40

        up8 = Conv2D(256, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(prelu7))
        # 128*96
        merge8 = Concatenate(axis=3)([prelu2, up8])
        conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv8)
        prelu8 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(
            conv8)
        # 128*96

        up9 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(prelu8))
        # 64*192
        merge9 = Concatenate(axis=3)([prelu1, up9])
        conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv9)
        prelu9 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(
            conv9)
        conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(prelu9)
        prelu10 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)(
            conv9)
        conv10 = Conv2D(1, 1)(prelu10)


        model = Model(input=inputs, output=conv10)


        model.compile(optimizer=Adam(lr=1e-3), loss='mean_squared_error', metrics=['accuracy'])
        return model








