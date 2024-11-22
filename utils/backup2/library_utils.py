##########################################################
# %%
# Library for tensorflow 2.2.0
##########################################################

import sys, os
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image

import h5py

# tensorflow
import tensorflow as tf

# keras
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Activation, BatchNormalization, \
    Conv2DTranspose, LeakyReLU, concatenate
from tensorflow.keras.models import Model


##########################################################
# %%
# define common functions
##########################################################

def mosaic(img, num_row, num_col, fig_num, clim, title='', use_transpose=False, use_flipud=False):
    fig = plt.figure(fig_num)
    fig.patch.set_facecolor('black')

    if img.ndim < 3:
        img_res = img
        plt.imshow(img_res)
        plt.gray()
        plt.clim(clim)
    else:
        if img.shape[2] != (num_row * num_col):
            print('sizes do not match')
        else:
            if use_transpose:
                for slc in range(0, img.shape[2]):
                    img[:, :, slc] = np.transpose(img[:, :, slc])

            if use_flipud:
                img = np.flipud(img)

            img_res = np.zeros((img.shape[0] * num_row, img.shape[1] * num_col))
            idx = 0

            for r in range(0, num_row):
                for c in range(0, num_col):
                    img_res[r * img.shape[0]: (r + 1) * img.shape[0], c * img.shape[1]: (c + 1) * img.shape[1]] = img[:,
                                                                                                                  :,
                                                                                                                  idx]
                    idx = idx + 1
        plt.imshow(img_res)
        plt.gray()
        plt.clim(clim)

    plt.suptitle(title, color='white', fontsize=48)


def msave_img(filename,data,intensity):
     
    data    =   (data - intensity[0]) * 255 / (intensity[1]-intensity[0])
    data[data>255]  =   255
    data[data<0]    =   0
    img     =   Image.fromarray(data.astype(np.uint8))
    img.save(filename)
    
    return True

def mvec(data):
    xl = data.size
    res = np.reshape(data, (xl))
    return res


def load_h5py(filename, rmod='r'):
    f = h5py.File(filename, rmod)
    arr = {}
    for k, v in f.items():
        arr[k] = np.transpose(np.array(v))
    return arr


def mfft(x, axis=0):
    # nx = x.shape[axis]
    y = np.fft.fftshift(np.fft.fft(np.fft.fftshift(x, axes=axis), axis=axis), axes=axis) # / np.sqrt(nx)
    return y


def mifft(x, axis=0):
    # nx = x.shape[axis]
    y = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis) # * np.sqrt(nx)
    return y


def mfft2(x, axes=(0, 1)):
    # nx, ny = x.shape[axes]
    y = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x, axes=axes), axes=axes), axes=axes) # / np.sqrt(nx,ny)
    return y


def mifft2(x, axes=(0, 1)):
    # nx, ny = x.shape[axes]
    y = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes) # * np.sqrt(nx,ny)
    return y


def msos(img, axis=3):
    return np.sqrt(np.sum(np.abs(img) ** 2, axis=axis))


##########################################################
# %%
# from zero-shot github
##########################################################

def uniform_selection(input_data, input_mask, rho=0.2, small_acs_block=(4, 4)):

    nrow, ncol = input_data.shape[0], input_data.shape[1]

    center_kx = int(find_center_ind(input_data, axes=(1, 2)))
    center_ky = int(find_center_ind(input_data, axes=(0, 2)))

    temp_mask = np.copy(input_mask)
    temp_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
    center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

    pr = np.ndarray.flatten(temp_mask)
    ind = np.random.choice(np.arange(nrow * ncol),
                            size=int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

    [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))

    loss_mask = np.zeros_like(input_mask)
    loss_mask[ind_x, ind_y] = 1

    trn_mask = input_mask - loss_mask

    return trn_mask, loss_mask

def getPSNR(ref, recon):
    """
    Measures PSNR between the reference and the reconstructed images
    """

    mse = np.sum(np.square(np.abs(ref - recon))) / ref.size
    psnr = 20 * np.log10(np.abs(ref.max()) / (np.sqrt(mse) + 1e-10))

    return psnr


def fft(ispace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters
    ----------
    ispace : coil images of size nrow x ncol x ncoil.
    axes :   The default is (0, 1).
    norm :   The default is None.
    unitary_opt : The default is True.

    Returns
    -------
    transform image space to k-space.

    """

    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(ispace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:

        fact = 1

        for axis in axes:
            fact = fact * kspace.shape[axis]

        kspace = kspace / np.sqrt(fact)

    return kspace


def ifft(kspace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters
    ----------
    ispace : image space of size nrow x ncol x ncoil.
    axes :   The default is (0, 1).
    norm :   The default is None.
    unitary_opt : The default is True.

    Returns
    -------
    transform k-space to image space.

    """

    ispace = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kspace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:

        fact = 1

        for axis in axes:
            fact = fact * ispace.shape[axis]

        ispace = ispace * np.sqrt(fact)

    return ispace


def norm(tensor, axes=(0, 1, 2), keepdims=True):
    """
    Parameters
    ----------
    tensor : It can be in image space or k-space.
    axes :  The default is (0, 1, 2).
    keepdims : The default is True.

    Returns
    -------
    tensor : applies l2-norm .

    """
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)

    if not keepdims: return tensor.squeeze()

    return tensor


def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Parameters
    ----------
    kspace : nrow x ncol x ncoil.
    axes :  The default is (1, 2, 3).

    Returns
    -------
    the center of the k-space

    """

    center_locs = norm(kspace, axes=axes).squeeze()

    return np.argsort(center_locs)[-1:]


def index_flatten2nd(ind, shape):
    """
    Parameters
    ----------
    ind : 1D vector containing chosen locations.
    shape : shape of the matrix/tensor for mapping ind.

    Returns
    -------
    list of >=2D indices containing non-zero locations

    """

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]


def sense1(input_kspace, sens_maps, axes=(0, 1)):
    """
    Parameters
    ----------
    input_kspace : nrow x ncol x ncoil
    sens_maps : nrow x ncol x ncoil

    axes : The default is (0,1).

    Returns
    -------
    sense1 image

    """

    image_space = ifft(input_kspace, axes=axes, norm=None, unitary_opt=True)
    Eh_op = np.conj(sens_maps) * image_space
    sense1_image = np.sum(Eh_op, axis=axes[-1] + 1)

    return sense1_image


def complex2real(input_data):
    """
    Parameters
    ----------
    input_data : row x col
    dtype :The default is np.float32.

    Returns
    -------
    output : row x col x 2

    """

    return np.stack((input_data.real, input_data.imag), axis=-1)


def real2complex(input_data):
    """
    Parameters
    ----------
    input_data : row x col x 2

    Returns
    -------
    output : row x col

    """

    return input_data[..., 0] + 1j * input_data[..., 1]

##########################################################
# %%
# tensorflow functions
##########################################################

# c2r = lambda x: tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)
# r2c = lambda x: tf.complex(x[..., 0], x[..., 1])


# def nrmse(y_true, y_pred):
#     return 100 * (K.sqrt(K.sum(K.square(y_pred - y_true)))) / (K.sqrt(K.sum(K.square(y_true))))

# def nmae(y_true, y_pred):
#     return 100 * (K.sum(K.abs(y_pred - y_true))) / (K.sum(K.abs(y_true)))

# def custom_norm_loss(y_true, y_pred):
#     return (nrmse(y_true, y_pred)+nmae(y_true, y_pred))/2.0



# class tfft3(Layer):
#     def __init__(self, **kwargs):
#         super(tfft3, self).__init__(**kwargs)

#     def build(self, input_shape):
#         super(tfft3, self).build(input_shape)

#     def call(self, x):
#         xc = r2c(x[0])

#         # fft3
#         xt = tf.signal.fftshift(xc, axes=(-3, -2, -1))
#         kt = tf.signal.fft3d(xt)
#         kt = tf.signal.fftshift(kt, axes=(-3, -2, -1))

#         return c2r(kt)


# class tifft3(Layer):
#     def __init__(self, **kwargs):
#         super(tifft3, self).__init__(**kwargs)

#     def build(self, input_shape):
#         super(tifft3, self).build(input_shape)

#     def call(self, x):
#         xc = r2c(x[0])

#         # ifft3
#         xt = tf.signal.fftshift(xc, axes=(-3, -2, -1))
#         kt = tf.signal.ifft3d(xt)
#         kt = tf.signal.fftshift(kt, axes=(-3, -2, -1))

#         return c2r(kt)


# class tfft2(Layer):
#     def __init__(self, **kwargs):
#         super(tfft2, self).__init__(**kwargs)

#     def build(self, input_shape):
#         super(tfft2, self).build(input_shape)

#     def call(self, x):
#         xc = r2c(x[0])

#         hx = int(int(xc.shape[1]) / 2)
#         hy = int(int(xc.shape[2]) / 2)

#         # fft2 over last two dimension
#         xt = tf.roll(xc, shift=(hx, hy), axis=(-2, -1))
#         kt = tf.signal.fft2d(xt)
#         kt = tf.roll(kt, shift=(hx, hy), axis=(-2, -1))

#         return c2r(kt)


# # needed to be changed to multi-coil
# class tifft2(Layer):
#     def __init__(self, **kwargs):
#         super(tifft2, self).__init__(**kwargs)

#     def build(self, input_shape):
#         super(tifft2, self).build(input_shape)

#     def call(self, x):
#         xc = r2c(x[0])

#         hx = int(int(xc.shape[1]) / 2)
#         hy = int(int(xc.shape[2]) / 2)

#         # ifft2 over last two dimension
#         it = tf.roll(xc, shift=(-hx, -hy), axis=(-2, -1))
#         it = tf.signal.ifft2d(it)
#         it = tf.roll(it, shift=(-hx, -hy), axis=(-2, -1))

#         return c2r(it)


##########################################################
# %%
# U-net functions
##########################################################

# # Conv2D -> Batch Norm -> Nonlinearity
# def conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type='relu', USE_BN=True, layer_name='', strides = (1, 1), alpha = 0.3 ):
#     with K.name_scope(layer_name):
#         x = Conv2D(num_out_chan, kernel_size, activation=None, padding='same', kernel_initializer='truncated_normal', strides=strides)(x)
#         if USE_BN:
#             x = BatchNormalization()(x)
#         if activation_type == 'LeakyReLU':
#             return LeakyReLU(alpha=alpha)(x)
#         else:
#             return Activation(activation_type)(x)




# # Conv2D -> Batch Norm -> softmax
# def conv2D_bn_softmax(x, num_out_chan, kernel_size, USE_BN=True, layer_name=''):
#     with K.name_scope(layer_name):
#         x = Conv2D(num_out_chan, kernel_size, activation=None, padding='same', kernel_initializer='truncated_normal')(x)
#         if USE_BN:
#             x = BatchNormalization()(x)
#         return Activation('softmax')(x)



# def UNet2D_softmax(nx, ny, ns, nc_input=2, kernel_size=(3, 3), num_out_chan_highest_level=64, depth=5,
#                    num_chan_increase_rate=2, activation_type='LeakyReLU', USE_BN=True):
#     # define the inputs
#     input_x = Input(shape=(nx, ny, nc_input), dtype=tf.float32)

#     x = conv2D_bn_nonlinear(input_x, num_out_chan_highest_level, kernel_size, activation_type=activation_type,
#                             USE_BN=USE_BN)

#     temp = createOneLevel_UNet2D(x, num_out_chan_highest_level, kernel_size, depth - 1, num_chan_increase_rate,
#                                  activation_type, USE_BN)

#     # output_img = conv2D_bn_nonlinear(temp, num_output_chans, kernel_size, activation_type=None, USE_BN=False)
#     output_img = conv2D_bn_softmax(temp, ns, kernel_size, USE_BN=False)

#     return Model(inputs=input_x, outputs=output_img)

