import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_hiddens):
        super(Model, self).__init__()
        
        encoder = []
        encoder.append(nn.Conv3d(1, num_hiddens, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)))
        encoder.append(nn.ReLU())
        encoder.append(nn.Conv3d(num_hiddens, num_hiddens, kernel_size=(6,3,3), stride=(2,1,1), padding=(2,1,1)))
        encoder.append(nn.ReLU())
        encoder.append(nn.Conv3d(num_hiddens, num_hiddens*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)))
        encoder.append(nn.ReLU())
        encoder.append(nn.Conv3d(num_hiddens*2, num_hiddens*2, kernel_size=(6,3,3), stride=(2,1,1), padding=(2,1,1)))
        encoder.append(nn.ReLU())
        encoder.append(nn.Conv3d(num_hiddens*2, num_hiddens*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)))
        encoder.append(nn.ReLU())
        encoder.append(nn.Conv3d(num_hiddens*4, num_hiddens*4, kernel_size=(6,3,3), stride=(2,1,1), padding=(2,1,1)))
        encoder.append(nn.ReLU())
        encoder.append(nn.Conv3d(num_hiddens*4, num_hiddens*8, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)))
        encoder.append(nn.ReLU())
        encoder.append(nn.Conv3d(num_hiddens*8, num_hiddens*8, kernel_size=(6,3,3), stride=(2,1,1), padding=(2,1,1)))
        encoder.append(nn.ReLU())
        encoder.append(nn.Conv3d(num_hiddens*8, num_hiddens*16, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)))
        encoder.append(nn.ReLU())
        encoder.append(nn.Conv3d(num_hiddens*16, num_hiddens*16, kernel_size=(7,3,3), stride=(1,1,1), padding=(1,1,1)))
        encoder.append(nn.ReLU())
        
        self._encoder = nn.Sequential(*encoder)


        decoder = []
        decoder.append(nn.ConvTranspose2d(num_hiddens*16, num_hiddens*8, kernel_size=4, stride=2, padding=1))
        encoder.append(nn.ReLU())
        decoder.append(nn.ConvTranspose2d(num_hiddens*8, num_hiddens*4, kernel_size=4, stride=2, padding=1))
        encoder.append(nn.ReLU())
        decoder.append(nn.ConvTranspose2d(num_hiddens*4, num_hiddens*4, kernel_size=4, stride=2, padding=1))
        encoder.append(nn.ReLU())
        decoder.append(nn.ConvTranspose2d(num_hiddens*4, 19*3, kernel_size=1, stride=1, padding=0))
        
        self._decoder = nn.Sequential(*decoder)
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv3d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()



    def forward(self, x):
        z = self._encoder(x)
        z = z.squeeze(dim = 2)
        x_recon = self._decoder(z)

        return x_recon