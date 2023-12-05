from model import CNNTrain
from utils import test_data_for_1600_1600_data, predict_mosaic_256
from utils import predict_mosaic_data

if __name__ == '__main__':
    train = CNNTrain()

    func_choice = 2 # 1: 256*256 ||  2:1600*1600

    # file info，加载权重结果路径
    train.load_model_path = r"D:\AINet\checkpoint/128_loss[9.203634011745454, 0.1328608810901642].h5"  # 模型路径
    if func_choice == 1:
        train.input_folder = r'D:\AINet\image\test_image/'  # 测试集数据
    if func_choice == 2:
        train.input_folder = r'D:\AINet\image\test_image/'  # 测试集数据

    train.output_folder = r'D:\AINet\test_image/'  # 保存预测结果

    train.pic_num = [1, 500]
    train.test_select = [0, 500]

    if func_choice == 1:
        predict_mosaic_256(train)

    if func_choice == 2:
        test_data_for_1600_1600_data(train)

