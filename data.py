
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import scipy.io as scio
import numpy as np
import torch
import os


class Datasets(Dataset):
    def __init__(self):
        initDataDir = "./data/"
        inputDir = 'input/'
        outputDir = 'output/'
        self.all_data = []
        files = os.listdir(initDataDir + inputDir)
        

        for file in files:
            label = np.load(initDataDir + outputDir + file + '.npy')
            label = label.transpose((2, 0, 1))
            assert label.shape == (57, 480, 640)
            crop_label = np.zeros((57, 480, 480))
            crop_label = label[:, :, 80:560]
            label = torch.tensor(crop_label)
            
            sub_files = os.listdir(initDataDir + inputDir + file)
            for sub_file in sub_files:
                data = scio.loadmat(initDataDir + inputDir + file + '/' + sub_file)
                data = data['CUBE1']

                data = np.expand_dims(data.transpose((2, 0, 1)), 0)
                data = data/100.0
                data = torch.tensor(data).float()

                self.all_data.append([data, label, file])

    def __getitem__(self, index):
        return self.all_data[index][0], self.all_data[index][1], self.all_data[index][2]

    def __len__(self):  # 返回所有样本的数目
        return len(self.all_data)
    
    
if __name__ == '__main__':   
    # 测试数据集         
    train_dataset = Datasets()  # 实例化自己构建的数据集
    train_loader = DataLoader(dataset = train_dataset, batch_size = 1, shuffle = True)
    
    for step, (data, label, file_name) in enumerate(train_loader):
        break
