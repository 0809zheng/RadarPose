#!pip3 install -U -r requirements.txt
from __future__ import print_function

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Model
from data import Datasets


# Training settings
parser = argparse.ArgumentParser(description='Model Hyperparameters')

# Network  Parameters
parser.add_argument('--num_hiddens', type=int, default=16, help="the base hidden channel numbers")

# Learning Hyperparameters
parser.add_argument('--random_seed', type=int, default=101, help="set random seed")
parser.add_argument('--learning_rate', type=float, default=1e-4, help="learning rate at begin")
parser.add_argument('--min_lr', type=float, default=1e-6, help="the minimum learning rate")
parser.add_argument('--lr_decay', type=int, default=10, help="learning rate decay epochs")
parser.add_argument('--epochs', type=int, default=1000, help="training epochs")

# Datasets Hyperparameters
parser.add_argument('--batch_size', type=int, default=32, help="actual batchsize")
parser.add_argument('--shuffle', type=bool, default=True, help="shuffle train and test datasets")

# Save and Load Path
parser.add_argument('--pretrained', type=bool, default=False, help="use pre-trained model")
parser.add_argument('--pretrain_path', type=str, default='./model/pretrained.pth', help="pre-trained model saved path")
parser.add_argument('--save_path', type=str, default='./log/', help='Location to save checkpoint models')
parser.add_argument('--snapshots', type=int, default=10, help='Snapshots')


opt = parser.parse_args()



def print_network(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(model)
    print('Total number of parameters: %d' % num_params)


def checkpoint(net, epoch):
    model_out_path = opt.save_path + "model_epoch_{}.pth".format(epoch)
    torch.save(net.state_dict(), model_out_path, _use_new_zipfile_serialization=False)
    print("Checkpoint saved to {}".format(model_out_path))
    

if __name__ == '__main__':
    if opt.random_seed is not None:
        np.random.seed(opt.random_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loss = []
    
    model = Model(opt.num_hiddens)
    model = nn.DataParallel(model)
    print_network(model)

    if opt.pretrained:
        if os.path.exists(opt.pretrain_path):
            model.load_state_dict(torch.load(opt.pretrain_path, map_location=lambda storage, loc: storage))
            print('Pre-trained model is loaded.')
    model = model.to(device)
            
    train_dataset = Datasets()  # 实例化自己构建的数据集
    train_loader = DataLoader(dataset = train_dataset, shuffle = opt.shuffle, batch_size = opt.batch_size, num_workers = 0)

    optimizer = optim.Adam(model.parameters(), lr = opt.learning_rate)
    loss_func = nn.MSELoss()

    start = time.time()
    for epoch in range(opt.epochs):
        print("Training epoch {}".format(epoch))
        batch_loss = []
        model.train()
    
        for step, (x, y, _) in enumerate(train_loader):
            x = x.to(device)  
            y = y.to(device)
            y_hat = model(x)    
            loss = loss_func(y_hat, y)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            batch_loss.append(loss.item())
#            print('Batch loss = {:.6f}'.format(loss.item()))
        print('Training loss = {:.6f}'.format(np.mean(batch_loss)))
        train_loss.append(np.mean(batch_loss))

        
        if (epoch+1) % (opt.snapshots) == 0:
            checkpoint(model, epoch)
             
        if (epoch+1) % (opt.lr_decay) == 0:
            for param_group in optimizer.param_groups:
                if param_group['lr'] > opt.min_lr:
                    param_group['lr'] *= 0.9
            print('Learning rate decay: lr={}\n'.format(optimizer.param_groups[0]['lr']))
    
    end_time = time.time() - start
    print("Finished. Time elapsed: {} seconds".format(end_time))
