
import cv2
import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio

import copy
from src import util
from src.parse import parsing

from model import Model

    
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = Model(16)
    model = nn.DataParallel(model)
    
    model.load_state_dict(torch.load('./model/pretrained.pth', map_location=lambda storage, loc: storage))
    print('Pre-trained model is loaded.')
    model = model.to(device)

    
    with torch.no_grad():
        model.eval()
        data = scio.loadmat('data/test4.mat')
        data = data['CUBE1']

        data = data.transpose((2, 0, 1))
        data = data/100.0
        data = torch.tensor(data).float()
        data = data.unsqueeze(0).unsqueeze(0)
        
        data = data.to(device) 

        y_hat = model(data)    
        y_hat = y_hat.detach().to("cpu").numpy()
        y_hat = y_hat.squeeze(0).transpose((1, 2, 0))
        heatmap = y_hat[:, :, 0:19]
        paf = y_hat[:, :, 19:]

        oriImg = np.zeros((paf.shape[0], paf.shape[1], 3))


        candidate, subset = parsing(heatmap, paf, oriImg)         
        canvas = copy.deepcopy(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)
            

        cv2.imwrite('./result/result.jpg', canvas)
    
    