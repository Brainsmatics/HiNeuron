import skimage.io as io
import time
import os
import numpy as np
import cv2
from math import *
from data import  split_input_image, generate_traindata_data_1, \
    generate_traindata_data_2



def test_data_for_1600_1600_data(train):  #
    ticks_start = time.time()
    X_path = train.input_folder
    output_filename = train.output_folder

    [X, Y] = generate_traindata_data_2(train.input_folder, train.pic_num[0], train.pic_num[1])

    x_test = X[train.test_select[0]:train.test_select[1]]
    print("==> load img Done!")
    train.model = train.init_AINet_model_data()
    train.model.load_weights(train.load_model_path)
    print("==> load model Done !")
    predict_mosaic_data(train, train.model, x_test, output_filename)
    ticks_end = time.time()
    time_cost = ticks_end - ticks_start
    print("time_cost:", time_cost)


def predict_mosaic_data(self, model, x_test, output_filename):
    if not os.path.exists(output_filename):
        print('data does not exist')
    print('Reading data-------')
    i_data = (split_input_image(self, x_test, train_flag=True))
    # print 'Predicting data--------'
    i_data = np.array(i_data)
    i_data = i_data[:, :, :, np.newaxis]
    print(i_data.shape)
    o_data = (model.predict(i_data, batch_size=8)).reshape((i_data.shape[0], self.output_length,self.output_length))

    o_data = np.uint16(o_data)
    if self.output_length == 1 and False:
        o_image = np.resize(o_data, (int(sqrt(o_data.shape[0])), int(sqrt(o_data.shape[0]))))
        cv2.imwrite(output_filename, o_image)
    else:
        o_image = np.zeros(
            (x_test.shape[0], self.MOSAIC_LENGTH_X + self.MOSAIC_FILL_X, self.MOSAIC_LENGTH_Y + self.MOSAIC_FILL_Y),
            np.float32)
        predict_count = np.zeros(
            (x_test.shape[0], self.MOSAIC_LENGTH_X + self.MOSAIC_FILL_X, self.MOSAIC_LENGTH_Y + self.MOSAIC_FILL_Y),
            np.float32)
        predict_temp = np.ones((self.output_length, self.output_length), np.float32)

        i = 0  #
        for indx in range(0, x_test.shape[0]):
            for p_x in range(0, self.MOSAIC_LENGTH_X, int(self.predict_stride / self.PREDICT_REPEAT)):
                for p_y in range(0, self.MOSAIC_LENGTH_Y, int(self.predict_stride / self.PREDICT_REPEAT)):
                    o_image[indx, p_x: p_x + self.output_length, p_y: p_y + self.output_length] += np.array(
                        o_data[i, :, :],
                        np.float32)
                    predict_count[indx, p_x: p_x + self.output_length, p_y: p_y + self.output_length] += np.array(
                        predict_temp, np.float32)
                    i = i + 1

        o_image = o_image / predict_count
        o_image_crop = o_image[:, 0 + self.cut:self.MOSAIC_LENGTH_X + self.cut,
                       0 + self.cut:self.MOSAIC_LENGTH_Y + self.cut]

        o_image_crop = np.uint16(o_image_crop)


        for i in range(o_image_crop.shape[0]):
             # disp(['==》 done ！'])
            print(i)
            io.imsave(output_filename + str(i) + '.tif',
                      o_image_crop[i, :, :])




def predict_mosaic_256(train):

    [x_test, y_test] = generate_traindata_data_2(train.input_folder, train.pic_num[0], train.pic_num[1])
    print("==> load img Done!")

    train.model = train.init_AINet_model_data()

    train.model.load_weights(train.load_model_path)
    print("==> load model Done !")
    Y = train.model.predict(x_test)

    delta_y = np.resize(Y, (Y.shape[0], Y.shape[1], Y.shape[2]))
    temp = delta_y
    img_y = np.uint16(temp)

    for i in range(x_test.shape[0]):
        io.imsave(train.output_folder + str(i) + '.tif', img_y[i])



