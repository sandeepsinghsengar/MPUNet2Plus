"""
Sandeep Singh Sengar
PostDoc, Machine Learning

University of Copenhagen
February 2021
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
                                    Concatenate, Conv2D, MaxPooling2D, \
                                    UpSampling2D, Reshape
from tensorflow.keras.models import Model


class unetConv2(Model):
    def __init__(self, filters, is_batchnorm, kernel_size, kernel_reg, activation, padding, n=2, stride=1):
        super(unetConv2, self).__init__()
        self.n = n
        #self.ks = ks
        self.stride = stride
        #self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
                conv = Conv2D(filters, kernel_size, activation,
                          padding, kernel_regularizer=kernel_reg)(inputs)
                conv = BatchNormalization()(conv)
                conv = Conv2D(filters, kernel_size, activation, padding, kernel_regularizer=kernel_reg)(conv)
                conv = BatchNormalization()(conv)
                setattr(self, 'conv%d' % i, conv)
            	

        else:
            	conv = Conv2D(filters, kernel_size, activation,
                          padding, kernel_regularizer=kernel_reg)(inputs)
            	conv = Conv2D(filters, kernel_size, activation,
                          padding, kernel_regularizer=kernel_reg)(conv)
                #setattr(self, 'conv%d' % i, conv)
                
        return conv        
	
        # initialise the blocks
        #for m in self.children():
        #    init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x
'''
class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
    
class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
'''
