# 主要功能：创建数据列表，将指定文件夹中的图像路径保存为 JSON 格式的列表文件，用于训练和测试。

from utils import create_data_lists
if __name__ == '__main__':
    create_data_lists(train_folders=[
        'C:/Users\\cbbis\\Desktop\\Data_fusion/x10/train'], test_folders=[
        'C:/Users/cbbis/Desktop/Data_fusion/x10/test'], min_size=10,
        output_folder='C:/Users/cbbis/Desktop/Data_fusion/data')
