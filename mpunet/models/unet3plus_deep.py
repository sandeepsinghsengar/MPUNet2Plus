"""
Sandeep Singh Sengar
PostDoc, Machine Learning

University of Copenhagen
February 2021
"""


from mpunet.logging import ScreenLogger
from mpunet.utils.conv_arithmetics import compute_receptive_fields
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import nn
from tensorflow.keras import regularizers
#from mpunet.models.layers import unetConv2
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
                                    Concatenate, Conv2D, MaxPooling2D, \
                                    UpSampling2D, Reshape
import numpy as np
'''
    UNet 3+
'''
class UNet3Plus_deep(Model):
    def __init__(self,
                 n_classes,
                 img_rows=None,
                 img_cols=None,
                 dim=None,
                 n_channels=1,
                 depth=4,
                 out_activation="softmax",
                 activation="relu",
                 kernel_size=3,
                 padding="same",
                 complexity_factor=1,
                 flatten_output=False,
                 l2_reg=None,
                 logger=None,
                 **kwargs): 
        super(UNet3Plus_deep, self).__init__()
        # Set logger or standard print wrapper
        self.logger = logger or ScreenLogger()
        #print('raw, column,dim=',img_rows, img_cols, dim) #output is raw, column,dim= None None 384
        if not ((img_rows and img_cols) or dim):
            raise ValueError("Must specify either img_rows and img_col or dim")
        if dim:
            img_rows, img_cols = dim, dim
            
        # Set various attributes
        self.img_shape = (img_rows, img_cols, n_channels)
        #self.is_deconv = is_deconv
        self.in_channels = n_channels
        self.is_batchnorm = True
        self.feature_scale = depth
        self.l2_reg=l2_reg
        self.n_classes = n_classes
        self.padding=padding
        self.kernel_size=kernel_size
        self.out_activation = out_activation
        self.activation = activation
        self.kernel_reg = regularizers.l2(l2_reg) if l2_reg else None
        self.flatten_output = flatten_output
        # Shows the number of pixels cropped of the input image to the output
        self.label_crop = np.array([[0, 0], [0, 0]])

        # Build model and init base keras Model class
        super().__init__(*self.init_model())

        # Compute receptive field
        names = [x.__class__.__name__ for x in self.layers]
        index = names.index("UpSampling2D")
        self.receptive_field = compute_receptive_fields(self.layers[:index])[-1][-1]
        # Log the model definition
        self.log() #uncomment it for log 
    def unetConv2(self, inputs, filters):
        if self.is_batchnorm:
                conv = Conv2D(filters, self.kernel_size, activation=self.activation,
                          padding=self.padding, kernel_regularizer=self.kernel_reg)(inputs)
                #conv = BatchNormalization()(conv)
                conv = Conv2D(filters, self.kernel_size, activation=self.activation,
                          padding=self.padding, kernel_regularizer=self.kernel_reg)(conv)
                conv = BatchNormalization()(conv)
        return conv
    
    def init_model(self):
    
        filters = [64, 128, 256, 512, 1024]
        inputs = Input(shape=self.img_shape)
        
        ##-----decoder paramaters------##
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks
        
        
        ## -------------Encoder-------------
        h1 = self.unetConv2(inputs, filters[0])  # h1->384*384*64 
        #print('shape of h1=',h1.shape)
        h2 = MaxPooling2D(pool_size=(2, 2))(h1)
        h2 = self.unetConv2(h2, filters[1])  # h2->192*192*128
        #print('shape of h2=',h2.shape)
	
        h3 = MaxPooling2D(pool_size=(2, 2))(h2)
        h3 = self.unetConv2(h3, filters[2])   # h3->96*96*256
        #print('shape of h3=',h3.shape)
	
        h4 = MaxPooling2D(pool_size=(2, 2))(h3)
        h4 = self.unetConv2(h4, filters[3])  # h4->48*48*512
        #print('shape of h4=',h4.shape)
	
        h5 = MaxPooling2D(pool_size=(2, 2))(h4)
        hd5 = self.unetConv2(h5, filters[4])  # h5->24*24*1024
        #print('shape of hd5=',hd5.shape)
        
        ## -------------Decoder-------------
        '''stage 4d, connections preparation for X_{De}^4 in paper'''
        # h1->384*384, hd4->48*48, Pooling 8 times
        h1_PT_hd4_max = MaxPooling2D(pool_size=(8, 8), strides=(8, 8))(h1)
        h1_PT_hd4_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(h1_PT_hd4_max)
        h1_PT_hd4_bn = BatchNormalization()(h1_PT_hd4_conv)
        h1_PT_hd4 = nn.relu(h1_PT_hd4_bn)
        #print('h1_PT_hd4_max, h1_PT_hd4_conv, h1_PT_hd4_bn, h1_PT_hd4=', h1_PT_hd4_max.shape, h1_PT_hd4_conv.shape, h1_PT_hd4_bn.shape, h1_PT_hd4.shape) #(None, 48, 48, 64) (None, 48, 48, 64) (None, 48, 48, 64) (None, 48, 48, 64)
        
        # h2->160*160, hd4->40*40, Pooling 4 times
        h2_PT_hd4_max = MaxPooling2D(pool_size=(4, 4))(h2)
        h2_PT_hd4_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(h2_PT_hd4_max)
        h2_PT_hd4_bn = BatchNormalization()(h2_PT_hd4_conv)
        h2_PT_hd4 = nn.relu(h2_PT_hd4_bn)
        #print('h2_PT_hd4_max, h2_PT_hd4_conv, h2_PT_hd4_bn, h2_PT_hd4=', h2_PT_hd4_max.shape, h2_PT_hd4_conv.shape, h2_PT_hd4_bn.shape, h2_PT_hd4.shape) #(None, 48, 48, 128) (None, 48, 48, 64) (None, 48, 48, 64) (None, 48, 48, 64)
        
        # h3->80*80, hd4->40*40, Pooling 2 times
        h3_PT_hd4_max = MaxPooling2D(pool_size=(2, 2))(h3)
        h3_PT_hd4_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(h3_PT_hd4_max)
        h3_PT_hd4_bn = BatchNormalization()(h3_PT_hd4_conv)
        h3_PT_hd4 = nn.relu(h3_PT_hd4_bn)
        #print('h3_PT_hd4_max, h3_PT_hd4_conv, h3_PT_hd4_bn, h3_PT_hd4=', h3_PT_hd4_max.shape, h3_PT_hd4_conv.shape, h3_PT_hd4_bn.shape, h3_PT_hd4.shape) #(None, 48, 48, 256) (None, 48, 48, 64) (None, 48, 48, 64) (None, 48, 48, 64)
        
        # h4->40*40, hd4->40*40, Concatenation
        h4_Cat_hd4_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(h4)
        h4_Cat_hd4_bn = BatchNormalization()(h4_Cat_hd4_conv)
        h4_Cat_hd4 = nn.relu(h4_Cat_hd4_bn)
        #print('h4_Cat_hd4_conv, h4_Cat_hd4_bn, h4_Cat_hd4=', h4_Cat_hd4_conv.shape, h4_Cat_hd4_bn.shape, h4_Cat_hd4.shape) #(None, 48, 48, 64) (None, 48, 48, 64) (None, 48, 48, 64)
        
        # hd5->20*20, hd4->40*40, Upsample 2 times
        hd5_UT_hd4_up = UpSampling2D(size=(2, 2), interpolation='bilinear')(hd5)  # 14*14
        hd5_UT_hd4_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(hd5_UT_hd4_up)
        hd5_UT_hd4_bn = BatchNormalization()(hd5_UT_hd4_conv)
        hd5_UT_hd4 = nn.relu(hd5_UT_hd4_bn)
        #print('hd5_UT_hd4_up, hd5_UT_hd4_conv, hd5_UT_hd4_bn, hd5_UT_hd4=', hd5_UT_hd4_up.shape, hd5_UT_hd4_conv.shape, hd5_UT_hd4_bn.shape, hd5_UT_hd4.shape) #(None, 48, 48, 1024) (None, 48, 48, 64) (None, 48, 48, 64) (None, 48, 48, 64)
        
        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        hd4_cat=tf.concat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 3)
        hd4_conv4d_1 = Conv2D(self.UpChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(hd4_cat)
        hd4_bn4d_1 = BatchNormalization()(hd4_conv4d_1)
        hd4 = nn.relu(hd4_bn4d_1)
        #print('shape of hd4_cat, hd4_conv4d_1, hd4_bn4d_1, hd4=',hd4_cat.shape, hd4_conv4d_1.shape, hd4_bn4d_1.shape, hd4.shape) #(None, 48, 48, 320) (None, 48, 48, 320) (None, 48, 48, 320) (None, 48, 48, 320)
        
        '''stage 3d, connections preparation for X_{De}^3 in paper'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        h1_PT_hd3_max = MaxPooling2D(pool_size=(4, 4))(h1)
        h1_PT_hd3_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(h1_PT_hd3_max)
        h1_PT_hd3_bn = BatchNormalization()(h1_PT_hd3_conv)
        h1_PT_hd3 = nn.relu(h1_PT_hd3_bn)
        
        # h2->160*160, hd3->80*80, Pooling 2 times
        h2_PT_hd3_max = MaxPooling2D(pool_size=(2, 2))(h2)
        h2_PT_hd3_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(h2_PT_hd3_max)
        h2_PT_hd3_bn = BatchNormalization()(h2_PT_hd3_conv)
        h2_PT_hd3 = nn.relu(h2_PT_hd3_bn)
        
        # h3->80*80, hd3->80*80,  Concatenation
        h3_Cat_hd3_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(h3)
        h3_Cat_hd3_bn = BatchNormalization()(h3_Cat_hd3_conv)
        h3_Cat_hd3 = nn.relu(h3_Cat_hd3_bn)
        
        # h4->40*40, hd3->80*80, Upsample 2 times
        hd4_UT_hd3_up = UpSampling2D(size=(2, 2), interpolation='bilinear')(hd4)  # 14*14
        hd4_UT_hd3_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(hd4_UT_hd3_up)
        hd4_UT_hd3_bn = BatchNormalization()(hd4_UT_hd3_conv)
        hd4_UT_hd3 = nn.relu(hd4_UT_hd3_bn)
        
        # hd5->20*20, hd3->80*80, Upsample 4 times
        hd5_UT_hd3_up = UpSampling2D(size=(4, 4), interpolation='bilinear')(hd5)  # 14*14
        hd5_UT_hd3_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(hd5_UT_hd3_up)
        hd5_UT_hd3_bn = BatchNormalization()(hd5_UT_hd3_conv)
        hd5_UT_hd3 = nn.relu(hd5_UT_hd3_bn)
        
        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        hd3_cat=tf.concat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 3)
        hd3_conv3d_1 = Conv2D(self.UpChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(hd3_cat)
        hd3_bn3d_1 = BatchNormalization()(hd3_conv3d_1)
        hd3 = nn.relu(hd3_bn3d_1)
        #print('shape of hd3=',hd3.shape) #(None, 96, 96, 320)
        
        
        '''stage 2d, connections preparation for X_{De}^2 in paper'''
        # h1->320*320, hd2->160*160, Pooling 2 times
        h1_PT_hd2_max = MaxPooling2D(pool_size=(2, 2))(h1)
        h1_PT_hd2_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(h1_PT_hd2_max)
        h1_PT_hd2_bn = BatchNormalization()(h1_PT_hd2_conv)
        h1_PT_hd2 = nn.relu(h1_PT_hd2_bn)
        
        # h2->160*160, hd2->160*160, Concatenation
        h2_Cat_hd2_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(h2)
        h2_Cat_hd2_bn = BatchNormalization()(h2_Cat_hd2_conv)
        h2_Cat_hd2 = nn.relu(h2_Cat_hd2_bn)
        
        # h3->80*80, hd2->160*160,  Upsample 2 times
        hd3_UT_hd2_up = UpSampling2D(size=(2, 2), interpolation='bilinear')(hd3)  # 14*14
        hd3_UT_hd2_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(hd3_UT_hd2_up)
        hd3_UT_hd2_bn = BatchNormalization()(hd3_UT_hd2_conv)
        hd3_UT_hd2 = nn.relu(hd3_UT_hd2_bn)
        
        # h4->40*40, hd2->160*160, Upsample 4 times
        hd4_UT_hd2_up = UpSampling2D(size=(4, 4), interpolation='bilinear')(hd4)  # 14*14
        hd4_UT_hd2_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(hd4_UT_hd2_up)
        hd4_UT_hd2_bn = BatchNormalization()(hd4_UT_hd2_conv)
        hd4_UT_hd2 = nn.relu(hd4_UT_hd2_bn)
        
        # hd5->20*20, hd2->160*160, Upsample 8 times
        hd5_UT_hd2_up = UpSampling2D(size=(8, 8), interpolation='bilinear')(hd5)  # 14*14
        hd5_UT_hd2_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(hd5_UT_hd2_up)
        hd5_UT_hd2_bn = BatchNormalization()(hd5_UT_hd2_conv)
        hd5_UT_hd2 = nn.relu(hd5_UT_hd2_bn)
        
        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        hd2_cat=tf.concat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 3)
        hd2_conv2d_1 = Conv2D(self.UpChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(hd2_cat)
        hd2_bn2d_1 = BatchNormalization()(hd2_conv2d_1)
        hd2 = nn.relu(hd2_bn2d_1)
        #print('shape of hd2=',hd2.shape) #(None, 192, 192, 320)
        
        '''stage 1d, connections preparation for X_{De}^1 in paper'''
        # h1->320*320, hd1->320*320, Concatenation
        h1_Cat_hd1_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(h1)
        h1_Cat_hd1_bn = BatchNormalization()(h1_Cat_hd1_conv)
        h1_Cat_hd1 = nn.relu(h1_Cat_hd1_bn)
        
        # h2->160*160, hd1->320*320, Upsample 2 times
        hd2_UT_hd1_up = UpSampling2D(size=(2, 2), interpolation='bilinear')(hd2)  # 14*14
        hd2_UT_hd1_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(hd2_UT_hd1_up)
        hd2_UT_hd1_bn = BatchNormalization()(hd2_UT_hd1_conv)
        hd2_UT_hd1 = nn.relu(hd2_UT_hd1_bn)
        
        # h3->80*80, hd1->320*320,  Upsample 4 times
        hd3_UT_hd1_up = UpSampling2D(size=(4, 4), interpolation='bilinear')(hd3)  # 14*14
        hd3_UT_hd1_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(hd3_UT_hd1_up)
        hd3_UT_hd1_bn = BatchNormalization()(hd3_UT_hd1_conv)
        hd3_UT_hd1 = nn.relu(hd3_UT_hd1_bn)
        
        # h4->40*40, hd1->320*320, Upsample 8 times
        hd4_UT_hd1_up = UpSampling2D(size=(8, 8), interpolation='bilinear')(hd4)  # 14*14
        hd4_UT_hd1_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(hd4_UT_hd1_up)
        hd4_UT_hd1_bn = BatchNormalization()(hd4_UT_hd1_conv)
        hd4_UT_hd1 = nn.relu(hd4_UT_hd1_bn)
        
        # hd5->20*20, hd1->320*320, Upsample 16 times
        hd5_UT_hd1_up = UpSampling2D(size=(16, 16), interpolation='bilinear')(hd5)  # 14*14
        hd5_UT_hd1_conv = Conv2D(self.CatChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(hd5_UT_hd1_up)
        hd5_UT_hd1_bn = BatchNormalization()(hd5_UT_hd1_conv)
        hd5_UT_hd1 = nn.relu(hd5_UT_hd1_bn)
        
        # fusion(h1_PT_hd1, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        hd1_cat=tf.concat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 3)
        hd1_conv1d_1 = Conv2D(self.UpChannels, self.kernel_size, activation=self.activation, padding=self.padding, kernel_regularizer=self.kernel_reg)(hd1_cat)
        hd1_bn1d_1 = BatchNormalization()(hd1_conv1d_1)
        hd1 = nn.relu(hd1_bn1d_1)
        #print('shape of hd1=',hd1.shape) #(None, 384, 384, 320)
        
        out1 = Conv2D(self.n_classes, 1, activation=self.out_activation)(hd1)
        out2 = Conv2D(self.n_classes, 1, activation=self.out_activation)(hd2_UT_hd1_up)
        out3 = Conv2D(self.n_classes, 1, activation=self.out_activation)(hd3_UT_hd1_up)
        out4 = Conv2D(self.n_classes, 1, activation=self.out_activation)(hd4_UT_hd1_up)
        out5 = Conv2D(self.n_classes, 1, activation=self.out_activation)(hd5_UT_hd1_up)
        out=(out1+out2+out3+out4+out5)/5
    	
    	
    	
    	
    	
        
        #print('input and output shapes before reshape are',inputs.shape, out.shape) #(None, 384, 384, 1) (None, 384, 384, 8)
        if self.flatten_output:
            out = Reshape([self.img_shape[0]*self.img_shape[1],
                           self.n_classes], name='flatten_output')(out)
        #print('input and output shapes after reshape are',inputs.shape, out.shape) #(None, 384, 384, 1) (None, 384, 384, 8)
        return [inputs], [out]
    def log(self):
        self.logger("UNet3+ Model Summary\n------------------")
        self.logger("Image rows:        %i" % self.img_shape[0])
        self.logger("Image cols:        %i" % self.img_shape[1])
        self.logger("Image channels:    %i" % self.img_shape[2])
        self.logger("N classes:         %i" % self.n_classes)
        #self.logger("CF factor:         %.3f" % self.cf**2)
        self.logger("Depth:             %i" % self.feature_scale)
        self.logger("l2 reg:            %s" % self.l2_reg)
        self.logger("Padding:           %s" % self.padding)
        self.logger("Receptive field:   %s" % self.receptive_field)
        self.logger("Conv activation:   %s" % self.activation)
        self.logger("Out activation:    %s" % self.out_activation)
        self.logger("N params:          %i" % self.count_params())
        self.logger("Output:            %s" % self.output)	
        
