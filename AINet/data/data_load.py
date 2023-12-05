import os
import numpy as np
import skimage.io as io
import cv2





def generate_traindata_data_1(X_path, Y_path, start_num, end_num):  
    prefix_Image = r"004_"
    x = list()
    for i in range(start_num, end_num + 1):

        img = io.imread(os.path.join(X_path, prefix_Image + "%05d.tif" % i))
        img = np.resize(img, (img.shape[0], img.shape[1], 1))
        x.append(img)

    X = list()

    for i in range(1, len(x) + 1):

        x_t = x[i-1]  #
        x_t = np.resize(x_t, (1, x_t.shape[0], x_t.shape[1], x_t.shape[2]))
        X.append(x_t)

    X = np.concatenate(X, 0)
    # X = X / 255

    y = list()
    for i in range(start_num, end_num + 1):

        img = io.imread(os.path.join(Y_path, prefix_Image + "%05d.tif" % i))
        img = np.resize(img, (img.shape[0], img.shape[1], 1))
        y.append(img)
    Y = list()
    for i in range(1, len(y) + 1):

        img = (y[i-1])
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        Y.append(img)

    Y = np.concatenate(Y, 0)
    # Y = Y / 255

    return [X, Y]

def generate_traindata_data_2(X_path, start_num, end_num):
    prefix_Image = r"test_"
    x = list()
    for i in range(start_num, end_num + 1):

        img = io.imread(os.path.join(X_path, prefix_Image + "%05d.tif" % i))
        img = np.resize(img, (img.shape[0], img.shape[1], 1))
        x.append(img)

    X = list()

    for i in range(1, len(x) + 1):

        x_t = x[i-1]  #
        x_t = np.resize(x_t, (1, x_t.shape[0], x_t.shape[1], x_t.shape[2]))
        X.append(x_t)

    X = np.concatenate(X, 0)
    # X = X / 255

    Y = list()

    return [X, Y]




def split_input_image(self, test_x, train_flag):  #
    wf_data = []
    im_wf = list()

    print('Reading data-------')
    for i in range(0, (test_x.shape[0])):

        im_wf_single_1 = test_x[i, :, :, 0]  #
        im_wf_single_1 = np.array(im_wf_single_1)

        im_wf_single_1 = im_wf_single_1.astype(np.uint16)

        im_wf_fill_1 = cv2.copyMakeBorder(im_wf_single_1, self.cut, self.cut, self.cut, self.cut,
                                          cv2.BORDER_REFLECT)
        im_wf_fill_1 = im_wf_fill_1.reshape(im_wf_fill_1.shape[0], im_wf_fill_1.shape[1],
                                            1)
        im_connect = im_wf_fill_1  #
        im_wf.append(im_connect)

    im_wf_fill = np.float32(np.array(im_wf))  #

    im_wf_fill = im_wf_fill[:, :, :, np.newaxis]

    if type(im_wf_fill) == np.ndarray:
        if train_flag:  ## Crop Image
            for indx in range(0, im_wf_fill.shape[0]):  #
                for p_x in range(0, self.MOSAIC_LENGTH_X, self.predict_stride):
                    for p_y in range(0, self.MOSAIC_LENGTH_Y, self.predict_stride):
                        img_temp = im_wf_fill[indx, p_x: p_x + self.IMAGE_LENGTH, p_y: p_y + self.IMAGE_LENGTH,
                                   :]  #
                        img_temp = np.array(img_temp).reshape(
                            [1, self.IMAGE_LENGTH, self.IMAGE_LENGTH])  #
                        wf_data.append(img_temp)
        else:
            for p_x in range(0, self.MOSAIC_LENGTH_X, self.predict_stride / self.PREDICT_REPEAT):
                for p_y in range(0, self.MOSAIC_LENGTH_Y, self.predict_stride / self.PREDICT_REPEAT):
                    img_temp = im_wf_fill[:, p_x: p_x + self.IMAGE_LENGTH, p_y: p_y + self.IMAGE_LENGTH, :]
                    img_temp = np.array(img_temp).reshape([1, self.MOSAIC_LENGTH_X, self.MOSAIC_LENGTH_X, 2])
                    wf_data.append(img_temp)

        wf_data = np.concatenate(wf_data, axis=0)  #
        return wf_data
    else:
        print('Warning!!!', name_x, 'does not exist!')
        return []
