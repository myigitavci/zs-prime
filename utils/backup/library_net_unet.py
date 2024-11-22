##########################################################
# %%
# import libraries
##########################################################

import numpy                as  np
import tensorflow           as  tf


# keras
from tensorflow.keras import Input
from tensorflow.keras.layers import MaxPooling2D, ZeroPadding2D, concatenate
from tensorflow.keras.models import Model
from library_net_function import *

##########################################################
# %%
# U-net 
##########################################################

def createOneLevel_UNet2D(x, num_out_chan, kernel_size, depth, num_chan_increase_rate, activation_type, USE_BN):
    if depth > 0:

        # Left
        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)
        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)

        x_to_lower_level = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last')(
            x)

        # Lower level
        x_from_lower_level = createOneLevel_UNet2D(x_to_lower_level, int(num_chan_increase_rate * num_out_chan),
                                                   kernel_size, depth - 1, num_chan_increase_rate, activation_type,
                                                   USE_BN)
        x_conv2Dt = conv2Dt_bn_nonlinear(x_from_lower_level, num_out_chan, kernel_size, activation_type=activation_type,
                                         USE_BN=USE_BN)

        # Right
        x = concatenate([x, x_conv2Dt], axis=3)
        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)
        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)

    else:
        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)
        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)

    return x


def create_unet(nx, ny, ns, nc_input=4, depth=4, num_filters=64, kernel_size=(3, 3), num_chan_increase_rate=2, activation_type='LeakyReLU', USE_BN=True):
    # define the inputs
    input_x = Input(shape=(nx, ny, nc_input), dtype=tf.float32)

    # zero-padding for preventing non integer U-net layer
    pnx, pny = 0, 0
    mnx, mny = nx, ny

    if np.mod(nx, (2 ** depth)) > 0:
        mnx = np.int(np.ceil(nx / (2 ** depth)) * (2 ** depth))
        pnx = np.int((mnx - nx) / 2)
    if np.mod(ny, (2 ** depth)) > 0:
        mny = np.int(np.ceil(ny / (2 ** depth)) * (2 ** depth))
        pny = np.int((mny - ny) / 2)

    # input_padded = ZeroPadding2D( padding=(pnx, pny) )(input_x)
    input_padded = ZeroPadding2D(padding=((0, 2 * pnx), (0, 2 * pny)))(input_x)

    # create Unet for segmentation
    out_padded  = createOneLevel_UNet2D(input_padded, num_out_chan = num_filters, kernel_size=kernel_size, 
                             depth=depth, num_chan_increase_rate=num_chan_increase_rate, activation_type=activation_type, USE_BN=USE_BN)
        
    # define model
    out_x       = out_padded[:, :nx, :ny, :]   
    out_x       = conv2D_bn_nonlinear(out_x, ns, kernel_size, activation_type=None, USE_BN=False)
    
    return Model(inputs     =   input_x,
                 outputs    =   out_x,  )
