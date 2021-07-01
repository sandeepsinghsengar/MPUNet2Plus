"""
Mathias Perslev
MSc Bioinformatics

University of Copenhagen
November 2017
"""

from mpunet.logging import ScreenLogger
from mpunet.utils.conv_arithmetics import compute_receptive_fields

from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
                                    Concatenate, Conv2D, MaxPooling2D, \
                                    UpSampling2D, Reshape, ZeroPadding2D,\
                                    Dense, Conv2DTranspose, concatenate
import numpy as np
from tensorflow import keras
import tensorflow as tf

from tensorflow.keras import backend as K
#from tensorflow.keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, Lambda
#from tensorflow.keras.layers.advanced_activations import ELU, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l2
#from tensorflow.keras.layers.noise import GaussianDropout

class UNet2Plus(Model):
    """
    2D UNet implementation with batch normalization and complexity factor adj.

    See original paper at http://arxiv.org/abs/1505.04597
    """
    def __init__(self,
                 n_classes,
                 img_rows=None,
                 img_cols=None,
                 dim=None,
                 n_channels=1,
                 depth=4,
                 out_activation="softmax",
                 activation="elu",
                 kernel_size=3,
                 padding="same",
                 complexity_factor=1,
                 flatten_output=False,
                 l2_reg=None,
                 logger=None,
                 **kwargs):
        """
        n_classes (int):
            The number of classes to model, gives the number of filters in the
            final 1x1 conv layer.
        img_rows, img_cols (int, int):
            Image dimensions. Note that depending on image dims cropping may
            be necessary. To avoid this, use image dimensions DxD for which
            D * (1/2)^n is an integer, where n is the number of (2x2)
            max-pooling layers; in this implementation 4.
            For n=4, D \in {..., 192, 208, 224, 240, 256, ...} etc.
        dim (int):
            img_rows and img_cols will both be set to 'dim'
        n_channels (int):
            Number of channels in the input image.
        depth (int):
            Number of conv blocks in encoding layer (number of 2x2 max pools)
            Note: each block doubles the filter count while halving the spatial
            dimensions of the features.
        out_activation (string):
            Activation function of output 1x1 conv layer. Usually one of
            'softmax', 'sigmoid' or 'linear'.
        activation (string):
            Activation function for convolution layers
        kernel_size (int):
            Kernel size for convolution layers
        padding (string):
            Padding type ('same' or 'valid')
        complexity_factor (int/float):
            Use int(N * sqrt(complexity_factor)) number of filters in each
            2D convolution layer instead of default N.
        flatten_output (bool):
            Flatten the output to array of shape [batch_size, -1, n_classes]
        l2_reg (float in [0, 1])
            L2 regularization on Conv2D weights
        logger (mpunet.logging.Logger | ScreenLogger):
            MutliViewUNet.Logger object, logging to files or screen.
        """
        super(UNet2Plus, self).__init__()
        # Set logger or standard print wrapper
        self.logger = logger or ScreenLogger()
        if not ((img_rows and img_cols) or dim):
            raise ValueError("Must specify either img_rows and img_col or dim")
        if dim:
            img_rows, img_cols = dim, dim

        # Set various attributes
        self.img_shape = (img_rows, img_cols, n_channels)
        self.num_class = n_classes
        self.cf = np.sqrt(complexity_factor)
        self.kernel_size = kernel_size
        self.activation = activation
        self.out_activation = out_activation
        self.l2_reg = l2_reg
        self.padding = padding
        self.depth = depth
        self.flatten_output = flatten_output
        self.img_rows=img_rows
        self.img_cols=img_cols
        self.smooth = 1.
        self.dropout_rate = 0.5
        self.kernel_reg = regularizers.l2(l2_reg) if l2_reg else None
	

        # Shows the number of pixels cropped of the input image to the output
        self.label_crop = np.array([[0, 0], [0, 0]])

        # Build model and init base keras Model class
        super().__init__(*self.init_model())

        # Compute receptive field
        #names = [x.__class__.__name__ for x in self.layers]
        #index = names.index("UpSampling2D")
        #self.receptive_field = compute_receptive_fields(self.layers[:index])[-1][-1]

        # Log the model definition
        self.log()

    def standard_unit(self, input_tensor, stage, nb_filter, kernel_size=3):
	
    	act = 'elu'
    	x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=self.kernel_reg)(input_tensor)
    	#x = Dropout(self.dropout_rate, name='dp'+stage+'_1')(x)
    	x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=self.kernel_reg)(x)
    	#x = Dropout(self.dropout_rate, name='dp'+stage+'_2')(x)
    	return x
    	
    def init_model(self):
    
    	color_type=1
    	deep_supervision=False
	
    	nb_filter = [64, 128, 256, 512, 1024]
    	act = 'elu'
    	inputs = Input(shape=self.img_shape)
    	num_class=self.num_class
	
    	# Handle Dimension Ordering for different backends
    	global bn_axis
    	bn_axis = 3
    	#img_input = Input(shape=(self.img_rows, self.img_cols, color_type), name='main_input')
      	
    	conv1_1 = self.standard_unit(inputs, stage='11', nb_filter=nb_filter[0])
    	pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    	conv2_1 = self.standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    	pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    	up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    	conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    	conv1_2 = self.standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    	conv3_1 = self.standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    	pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    	up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    	conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    	conv2_2 = self.standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    	up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    	conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    	conv1_3 = self.standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    	conv4_1 = self.standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    	pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    	up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    	conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    	conv3_2 = self.standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    	up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    	conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    	conv2_3 = self.standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    	up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    	conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    	conv1_4 = self.standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    	conv5_1 = self.standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    	up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    	conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    	conv4_2 = self.standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])
    	up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    	conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    	conv3_3 = self.standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    	up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    	conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    	conv2_4 = self.standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    	up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    	conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    	conv1_5 = self.standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    	nestnet_output_1 = Conv2D(num_class, (1, 1), activation=self.out_activation, name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=self.kernel_reg)(conv1_2)
    	nestnet_output_2 = Conv2D(num_class, (1, 1), activation=self.out_activation, name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=self.kernel_reg)(conv1_3)
    	nestnet_output_3 = Conv2D(num_class, (1, 1), activation=self.out_activation, name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=self.kernel_reg)(conv1_4)
    	nestnet_output_4 = Conv2D(num_class, (1, 1), activation=self.out_activation, name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=self.kernel_reg)(conv1_5)
    	out = Reshape([self.img_shape[0]*self.img_shape[1], num_class], name='flatten_output')(nestnet_output_4)
    	return [inputs], [out]
        
        
    	

    def crop_nodes_to_match(self, node1, node2):
        """
        If necessary, applies Cropping2D layer to node1 to match shape of node2
        """
        s1 = np.array(node1.get_shape().as_list())[1:-1]
        s2 = np.array(node2.get_shape().as_list())[1:-1]

        if np.any(s1 != s2):
            c = (s1 - s2).astype(np.int)
            cr = np.array([c//2, c//2]).T
            cr[:, 1] += c % 2
            cropped_node1 = Cropping2D(cr)(node1)
            self.label_crop += cr
        else:
            cropped_node1 = node1

        return cropped_node1
    	

    def log(self):
        self.logger("UNet Model Summary\n------------------")
        self.logger("Image rows:        %i" % self.img_shape[0])
        self.logger("Image cols:        %i" % self.img_shape[1])
        self.logger("Image channels:    %i" % self.img_shape[2])
        self.logger("N classes:         %i" % self.num_class)
        self.logger("CF factor:         %.3f" % self.cf**2)
        self.logger("Depth:             %i" % self.depth)
        self.logger("l2 reg:            %s" % self.l2_reg)
        self.logger("Padding:           %s" % self.padding)
        self.logger("Conv activation:   %s" % self.activation)
        self.logger("Out activation:    %s" % self.out_activation)
        #self.logger("Receptive field:   %s" % self.receptive_field)
        self.logger("N params:          %i" % self.count_params())
        self.logger("Output:            %s" % self.output)
        #self.logger("Crop:              %s" % (self.label_crop if np.sum(self.label_crop) != 0 else "None"))
