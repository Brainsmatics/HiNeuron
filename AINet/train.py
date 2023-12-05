from model import CNNTrain
from data import generate_traindata_data_1,generate_traindata_data_2
import os
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad
from keras.callbacks import ModelCheckpoint, TensorBoard

if __name__ == '__main__':
    train = CNNTrain()
    # file info
    train.FOLDER = r"D:\AINet\checkpoint/"  #
    train.input_folder = r'D:\AINet\image\train_image\LR/'  # LR
    train.input_folder_Y = r'D:\AINet\image\train_image\HR/'  # HR

    train.pic_num = [1, 11800]
    train.train_num = [1, 11000]  # Train image
    train.test_num = [11001, 11800]


    train.TRAIN_BATCH = 10
    train.TRAIN_TIMES = 100
    train.learning_rate = 1e-5


    # read train file route
    X_path = train.input_folder
    Y_path = train.input_folder_Y
    # train image number
    [X, Y] = generate_traindata_data_1(X_path, Y_path, train.pic_num[0], train.pic_num[1])

    x_train = X[train.train_num[0]-1:train.train_num[1]]
    y_train = Y[train.train_num[0]-1:train.train_num[1]]
    #
    x_test = X[train.test_num[0]-1:train.test_num[1]]
    y_test = Y[train.test_num[0]-1:train.test_num[1]]

    for j in range(train.TRAIN_REPEAT):
        print('Init new model------')

        model = train.init_AINet_model_data()


        model.summary()
        model.compile(optimizer=Adam(lr=train.learning_rate), loss='mean_squared_error', metrics=['accuracy'])
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('Training')


        model.fit(x_train, y_train, epochs=train.TRAIN_TIMES, batch_size=train.TRAIN_BATCH,
                  validation_split=0.1, callbacks=[train.early_stopping, TensorBoard(log_dir='./log')])

        print('Testing ------------')
        loss = model.evaluate(x_test, y_test, batch_size=train.TRAIN_BATCH)
        train.loss_list.append(loss)
        print('test loss: ', loss)
        print('Saving model as' + str(train.IMAGE_LENGTH) + '_loss' + str(loss) + '.h5')

        model.save(os.path.join(train.FOLDER, str(train.IMAGE_LENGTH) + '_loss' + str(loss) + '.h5'))



