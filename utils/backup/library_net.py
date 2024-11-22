##########################################################
# %%
# import libraries
##########################################################

import numpy                as  np
import tensorflow           as  tf

# keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Concatenate, Lambda
from tensorflow.keras.models import Model
from scipy.linalg import dft

from library_net_function import *
# from library_net_sense  import create_sense
from library_net_modl1   import create_modl1
from library_net_modl2   import create_modl2
from library_net_modl3   import create_modl3
from library_net_unet   import create_unet

##########################################################
# %%
# forward model
##########################################################

class pForward(Layer):
    def __init__(self, **kwargs):
        super(pForward, self).__init__(**kwargs)

    def build(self, input_shape):
        super(pForward, self).build(input_shape)

    def call(self, x):
        o1,c1,m1,f1,fmt,tl,bool_updown   =  x
        c_p         =   tf.transpose(c1, perm=[0,2,3,1])
        time_eff    =   tf.cond(bool_updown, lambda: tl, lambda: tf.reverse(tl, axis = [0]))
        def forward1(tmp1):
            o2,c2,m2,f2= tmp1
            def forward2(tmp2):
                o3,c3,m3,f3= tmp2
                # find encoding matrix
                fm_phase    =   tf.constant(2.0 * np.pi, dtype=tf.complex64) *  tf.linalg.matmul( tf.expand_dims(time_eff,axis=1) , tf.expand_dims(f3[:,0],axis=0) )
                fm_exp      =   tf.math.exp(1j*tf.squeeze(fm_phase))
                fmt_sub     =   fmt * fm_exp * tf.expand_dims(m3, axis=-1)
                # Ax
                i_coil      =   tf.multiply(o3,c3)
                Ax          =   tf.linalg.matmul(fmt_sub,i_coil)
                return Ax

            inp2    =   (o2,c2,m2,f2)
            y1      =   tf.map_fn(forward2, inp2, dtype=tf.complex64)
            y12     =   tf.transpose(y1,[1,2,0]) # [x,y,c] -> [y,c,x]
            y13     =   tf.signal.fftshift(tf.signal.fft(tf.signal.fftshift(y12,-1)),-1)
            y14     =   tf.transpose(y13,[1,2,0]) # [y,c,x] -> [c,x,y]
            # y14     =   tf.transpose(y13,[2,0,1]) # [y,c,x] -> [x,y,c]
            return y14

        inp1 = (o1,c_p,m1,f1)
        rec = tf.map_fn(forward1, inp1, dtype=tf.complex64)

        return rec


##########################################################
# %%
# phase
##########################################################

class phase_hamming(Layer):
    def __init__(self, **kwargs):
        super(phase_hamming, self).__init__(**kwargs)

    def build(self, input_shape):
        super(phase_hamming, self).build(input_shape)

    def call(self, x):
        [img]   =   x
        nx      =   img.shape[1]
        ny      =   img.shape[2]

        hmx     =   np.hamming(nx).reshape((1,1,nx,1))
        hmy     =   np.hamming(ny).reshape((1,1,1,ny))
        hmxy    =   tf.convert_to_tensor(hmx * hmy, dtype=tf.complex64)
        # hmx     =   tf.ones((1,1,nx,1),  dtype=tf.dtypes.complex64)
        # hmy     =   tf.ones((1,1,1,ny),  dtype=tf.dtypes.complex64)
        # hmxy    =   hmx * hmy # tf.convert_to_tensor(hmx * hmy, dtype=tf.complex64)

        # kdata   =   tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(img[...,0],[-2,-1])),[-2,-1])
        # # print(img.shape)
        kdata   =   tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(tf.transpose(img,(0,3,1,2)),[-2,-1])),[-2,-1]) /  tf.math.sqrt(tf.cast(nx*ny,tf.complex64))
        kdat_h  =   hmxy * kdata
        idata   =   tf.transpose(tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(kdat_h,[-2,-1])),[-2,-1]),(0,2,3,1)) *  tf.math.sqrt(tf.cast(nx*ny,tf.complex64))

        phs     =   tf.math.angle(idata+K.epsilon())
        # phs     =   tf.math.angle(idata) # tf.expand_dims(tf.math.angle(idata),-1)
        img_r   =   img * tf.math.exp(-1j*tf.cast(phs, tf.complex64))

        return [img_r, tf.cast(phs, tf.complex64)]


class C_phase(Layer):
    def __init__(self, **kwargs):
        super(C_phase, self).__init__(**kwargs)

    def build(self, input_shape):
        super(C_phase, self).build(input_shape)

    def call(self, x):
        img, phs    =   x
        res         =   img * tf.math.exp(-1j*tf.cast(phs, tf.complex64))

        return tf.cast(res, tf.complex64)


class F_phase(Layer):
    def __init__(self, **kwargs):
        super(F_phase, self).__init__(**kwargs)

    def build(self, input_shape):
        super(F_phase, self).build(input_shape)

    def call(self, x):
        img, phs    =   x
        res         =   img * tf.math.exp(1j*tf.cast(phs, tf.complex64))

        return tf.cast(res, tf.complex64)



##########################################################
# %%
# network
##########################################################

def create_buda_net_v1(nx, ny, nc, nLayers, num_block, num_filters = 64, esp = 0, unet_depth = 4, unet_filters = 32):

    # define the inputs
    input_c         =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_c')
    input_k_trn1    =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_trn1')
    input_k_trn2    =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_trn2')
    input_k_lss1    =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_lss1')
    input_k_lss2    =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_lss2')
    input_m_trn1    =   Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_trn1')
    input_m_trn2    =   Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_trn2')
    input_m_lss1    =   Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_lss1')
    input_m_lss2    =   Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_lss2')

    # define networks
    recon_single    =   create_modl1(nx = nx, ny = ny, nc = nc, num_block = num_block,  nLayers = nLayers, num_filters=   num_filters)
    recon_joint     =   create_modl2(nx = nx, ny = ny, nc = nc, num_block = num_block,  nLayers = nLayers, num_filters=   num_filters , esp = esp)
    unet_field      =   create_unet(nx,ny,1,nc_input=2,kernel_size=(3,3), num_filters=unet_filters, depth=unet_depth,
                            num_chan_increase_rate=2, activation_type='LeakyReLU', USE_BN=True)

    # functions and variables
    oForward    =   pForward()
    rmbg        =   rm_bg()

    # define variable
    # fftmtx      =   tf.convert_to_tensor(np.fft.fftshift(dft(ny),axes=(0,1))/ny, dtype=tf.complex64)
    fftmtx      =   tf.convert_to_tensor(np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.eye(ny,dtype=np.complex64),axes=0),axis=0),axes=0),dtype=tf.complex64) # / np.sqrt(ny)
    timeline    =   tf.convert_to_tensor(np.arange(ny,dtype=np.complex64) * esp, dtype=tf.complex64)

    # single shot recon
    out1        =   recon_single([input_c, input_m_trn1, input_k_trn1])
    out2        =   recon_single([input_c, input_m_trn2, input_k_trn2])
    out3        =   K.expand_dims(r2c(out1),axis=-1)
    out4        =   K.expand_dims(r2c(out2),axis=-1)

    # field estimate
    field_est   =   unet_field(Concatenate(axis=-1)([K.abs(out3),K.abs(out4)])) # input_fm #
    field_est   =   rmbg([field_est,input_c])

    # BUDA
    [out5,out6] =   recon_joint([input_c, input_m_trn1, input_m_trn2, input_k_trn1, input_k_trn2, field_est])
    out7        =   K.expand_dims(r2c(out5),axis=-1)
    out8        =   K.expand_dims(r2c(out6),axis=-1)

    # loss points in k-space
    lss_1st     =   oForward([out7,     input_c,    input_m_lss1+input_m_trn1,  K.cast(field_est,tf.complex64), fftmtx, timeline, True])
    lss_2nd     =   oForward([out8,     input_c,    input_m_lss2+input_m_trn2,  K.cast(field_est,tf.complex64), fftmtx, timeline, False])

    # loss points in k-space
    lss_l1      =   K.sum(K.abs(lss_1st - input_k_lss1 - input_k_trn1),axis=1) / (K.sum(K.abs(input_k_lss1+input_k_trn1))+K.sum(K.abs(input_k_lss2+input_k_trn2)))   \
                    + K.sum(K.abs(lss_2nd - input_k_lss2 - input_k_trn2),axis=1) / (K.sum(K.abs(input_k_lss1+input_k_trn1))+K.sum(K.abs(input_k_lss2+input_k_trn2)))
    lss_l2      =   K.sum(K.square(K.abs(lss_1st - input_k_lss1 - input_k_trn1)),axis=1) / K.sqrt(K.sum(K.square(K.abs(input_k_lss1+input_k_trn1))+K.square(K.abs(input_k_lss2+input_k_trn2))))   \
                    + K.sum(K.square(K.abs(lss_2nd - input_k_lss2 - input_k_trn2)),axis=1) / K.sqrt(K.sum(K.square(K.abs(input_k_lss1+input_k_trn1))+K.square(K.abs(input_k_lss2+input_k_trn2))))

    # outputs
    out_final   =   Concatenate(axis=-1)([  out3,   out4,   \
                                            K.cast(field_est,tf.complex64),         \
                                            out7,   out8,   \
                                            K.cast(K.expand_dims(lss_l1,axis=-1),tf.complex64),          \
                                            K.cast(K.expand_dims(lss_l2,axis=-1),tf.complex64),          \
                                          ])

    return Model(inputs     =   [ input_c,  input_k_trn1,   input_k_trn2,   input_k_lss1,   input_k_lss2,
                                            input_m_trn1,   input_m_trn2,   input_m_lss1,   input_m_lss2    ],
                 outputs    =   [ out_final  ],
                 name       =   'buda-net-v01' )


def create_buda_net_v2(nx, ny, nc, nLayers, num_block, num_filters = 64, esp = 0, unet_depth = 4, unet_filters = 32):

    # define the inputs
    input_c         =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_c')
    input_k_trn1    =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_trn1')
    input_k_trn2    =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_trn2')
    input_k_lss1    =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_lss1')
    input_k_lss2    =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_lss2')
    input_m_trn1    =   Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_trn1')
    input_m_trn2    =   Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_trn2')
    input_m_lss1    =   Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_lss1')
    input_m_lss2    =   Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_lss2')

    # define networks
    recon_single    =   create_modl1(nx = nx, ny = ny, nc = nc, num_block = num_block,  nLayers = nLayers, num_filters=   num_filters)
    recon_joint     =   create_modl2(nx = nx, ny = ny, nc = nc, num_block = num_block,  nLayers = nLayers, num_filters=   num_filters , esp = esp)
    unet_field      =   create_unet(nx,ny,1,nc_input=2,kernel_size=(3,3), num_filters=unet_filters, depth=unet_depth,
                            num_chan_increase_rate=2, activation_type='LeakyReLU', USE_BN=True)

    # functions and variables
    oForward    =   pForward()
    rmbg        =   rm_bg()
    PHam        =   phase_hamming()
    Cphs        =   C_phase()
    Fphs        =   F_phase()

    # define variable
    # fftmtx      =   tf.convert_to_tensor(np.fft.fftshift(dft(ny),axes=(0,1))/ny, dtype=tf.complex64)
    fftmtx      =   tf.convert_to_tensor(np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.eye(ny,dtype=np.complex64),axes=0),axis=0),axes=0),dtype=tf.complex64) # / np.sqrt(ny)
    timeline    =   tf.convert_to_tensor(np.arange(ny,dtype=np.complex64) * esp, dtype=tf.complex64)

    # single shot recon
    out1        =   recon_single([input_c, input_m_trn1, input_k_trn1])
    out2        =   recon_single([input_c, input_m_trn2, input_k_trn2])
    out3        =   K.expand_dims(r2c(out1),axis=-1)
    out4        =   K.expand_dims(r2c(out2),axis=-1)

    # field estimate
    field_est   =   unet_field(Concatenate(axis=-1)([K.abs(out3),K.abs(out4)])) # input_fm #
    field_est   =   rmbg([field_est,input_c])

    # BUDA
    [out5,out6] =   recon_joint([input_c, input_m_trn1, input_m_trn2, input_k_trn1, input_k_trn2, field_est])
    out7        =   K.expand_dims(r2c(out5),axis=-1)
    out8        =   K.expand_dims(r2c(out6),axis=-1)

    # phase estimatation using hamming filter
    [rimg1,pimg1]  =   PHam([out7])
    [rimg2,pimg2]  =   PHam([out8])

    # combine imgs
    comb_img    =   (rimg1 + rimg2) / 2.0

    # loss points in k-space
    lss_1st     =   oForward([Fphs([comb_img,pimg1]),     input_c,    input_m_lss1+input_m_trn1,  K.cast(field_est,tf.complex64), fftmtx, timeline, True])
    lss_2nd     =   oForward([Fphs([comb_img,pimg2]),     input_c,    input_m_lss2+input_m_trn2,  K.cast(field_est,tf.complex64), fftmtx, timeline, False])

    # loss points in k-space
    lss_l1      =   K.sum(K.abs(lss_1st - input_k_lss1 - input_k_trn1),axis=1) / (K.sum(K.abs(input_k_lss1+input_k_trn1))+K.sum(K.abs(input_k_lss2+input_k_trn2)))   \
                    + K.sum(K.abs(lss_2nd - input_k_lss2 - input_k_trn2),axis=1) / (K.sum(K.abs(input_k_lss1+input_k_trn1))+K.sum(K.abs(input_k_lss2+input_k_trn2)))
    lss_l2      =   K.sum(K.square(K.abs(lss_1st - input_k_lss1 - input_k_trn1)),axis=1) / K.sqrt(K.sum(K.square(K.abs(input_k_lss1+input_k_trn1))+K.square(K.abs(input_k_lss2+input_k_trn2))))   \
                    + K.sum(K.square(K.abs(lss_2nd - input_k_lss2 - input_k_trn2)),axis=1) / K.sqrt(K.sum(K.square(K.abs(input_k_lss1+input_k_trn1))+K.square(K.abs(input_k_lss2+input_k_trn2))))

    # outputs
    out_final   =   Concatenate(axis=-1)([  out3,   out4,   \
                                            K.cast(field_est,tf.complex64),         \
                                            out7,   out8,   \
                                            comb_img,                               \
                                            K.cast(K.expand_dims(lss_l1,axis=-1),tf.complex64),          \
                                            K.cast(K.expand_dims(lss_l2,axis=-1),tf.complex64),          \
                                          ])

    return Model(inputs     =   [ input_c,  input_k_trn1,   input_k_trn2,   input_k_lss1,   input_k_lss2,
                                            input_m_trn1,   input_m_trn2,   input_m_lss1,   input_m_lss2    ],
                 outputs    =   [ out_final  ],
                 name       =   'buda-net-v02' )



# using TOPUP Field Map Input
def create_buda_net_v3(nx, ny, nc, nLayers, num_block, num_filters = 64, esp = 0):

    # define the inputs
    input_c         =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_c')
    input_k_trn1    =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_trn1')
    input_k_trn2    =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_trn2')
    input_k_lss1    =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_lss1')
    input_k_lss2    =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_lss2')
    input_m_trn1    =   Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_trn1')
    input_m_trn2    =   Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_trn2')
    input_m_lss1    =   Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_lss1')
    input_m_lss2    =   Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_lss2')
    input_fm        =   Input(shape=(nx,ny,1),    dtype = tf.float32,         name = 'input_fm')

    # define networks
    recon_joint     =   create_modl2(nx = nx, ny = ny, nc = nc, num_block = num_block,  nLayers = nLayers, num_filters=   num_filters , esp = esp)

    # BUDA
    [out5,out6] =   recon_joint([input_c, input_m_trn1, input_m_trn2, input_k_trn1, input_k_trn2, input_fm])
    out7        =   K.expand_dims(r2c(out5),axis=-1)
    out8        =   K.expand_dims(r2c(out6),axis=-1)

    print(out7.shape)

    # outputs
    out_final   =   Concatenate(axis=-1)([  out7,   out8,                           \
                                            K.cast(input_fm,tf.complex64),          \
                                          ])
    print(out_final.shape)
    return Model(inputs     =   [ input_c,  input_k_trn1,   input_k_trn2,   input_k_lss1,   input_k_lss2,
                                            input_m_trn1,   input_m_trn2,   input_m_lss1,   input_m_lss2,   input_fm    ],
                 outputs    =   out_final,
                 name       =   'buda-net-v03' )
'''

    # functions and variables
    oForward    =   pForward()
    rmbg        =   rm_bg()
    PHam        =   phase_hamming()
    Cphs        =   C_phase()
    Fphs        =   F_phase()

    # define variable
    # fftmtx      =   tf.convert_to_tensor(np.fft.fftshift(dft(ny),axes=(0,1))/ny, dtype=tf.complex64)
    fftmtx      =   tf.convert_to_tensor(np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.eye(ny,dtype=np.complex64),axes=0),axis=0),axes=0),dtype=tf.complex64) # / np.sqrt(ny)
    timeline    =   tf.convert_to_tensor(np.arange(ny,dtype=np.complex64) * esp, dtype=tf.complex64)

    # BUDA
    [out5,out6] =   recon_joint([input_c, input_m_trn1, input_m_trn2, input_k_trn1, input_k_trn2, input_fm])
    out7        =   K.expand_dims(r2c(out5),axis=-1)
    out8        =   K.expand_dims(r2c(out6),axis=-1)

    # phase estimatation using hamming filter
    [rimg1,pimg1]  =   PHam([out7])
    [rimg2,pimg2]  =   PHam([out8])

    # combine imgs
    comb_img    =   (rimg1 + rimg2) / 2.0

    # loss points in k-space
    lss_1st     =   oForward([Fphs([comb_img,pimg1]),     input_c,    input_m_lss1+input_m_trn1,  K.cast(input_fm,tf.complex64), fftmtx, timeline, True])
    lss_2nd     =   oForward([Fphs([comb_img,pimg2]),     input_c,    input_m_lss2+input_m_trn2,  K.cast(input_fm,tf.complex64), fftmtx, timeline, False])

    # loss points in k-space
    lss_l1      =   K.sum(K.abs(lss_1st - input_k_lss1 - input_k_trn1),axis=1) / (K.sum(K.abs(input_k_lss1+input_k_trn1))+K.sum(K.abs(input_k_lss2+input_k_trn2)))   \
                    + K.sum(K.abs(lss_2nd - input_k_lss2 - input_k_trn2),axis=1) / (K.sum(K.abs(input_k_lss1+input_k_trn1))+K.sum(K.abs(input_k_lss2+input_k_trn2)))
    lss_l2      =   K.sum(K.square(K.abs(lss_1st - input_k_lss1 - input_k_trn1)),axis=1) / K.sqrt(K.sum(K.square(K.abs(input_k_lss1+input_k_trn1))+K.square(K.abs(input_k_lss2+input_k_trn2))))   \
                    + K.sum(K.square(K.abs(lss_2nd - input_k_lss2 - input_k_trn2)),axis=1) / K.sqrt(K.sum(K.square(K.abs(input_k_lss1+input_k_trn1))+K.square(K.abs(input_k_lss2+input_k_trn2))))

    # outputs
    out_final   =   Concatenate(axis=-1)([  out7,   out8,                           \
                                            K.cast(input_fm,tf.complex64),          \
                                            comb_img,                               \
                                            K.cast(K.expand_dims(lss_l1,axis=-1),tf.complex64),          \
                                            K.cast(K.expand_dims(lss_l2,axis=-1),tf.complex64),          \
                                          ])

    return Model(inputs     =   [ input_c,  input_k_trn1,   input_k_trn2,   input_k_lss1,   input_k_lss2,
                                            input_m_trn1,   input_m_trn2,   input_m_lss1,   input_m_lss2,   input_fm    ],
                 outputs    =   [ out_final  ],
                 name       =   'buda-net-v03' )
'''

# field map udate iteratively
def create_buda_net_v4(nx, ny, nc, nLayers, num_block, num_filters = 64, esp = 0, unet_depth = 4, unet_filters = 32):

    # define the inputs
    input_c         =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_c')
    input_k_trn1    =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_trn1')
    input_k_trn2    =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_trn2')
    input_k_lss1    =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_lss1')
    input_k_lss2    =   Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_lss2')
    input_m_trn1    =   Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_trn1')
    input_m_trn2    =   Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_trn2')
    input_m_lss1    =   Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_lss1')
    input_m_lss2    =   Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_lss2')

    # define networks
    recon_joint     =   create_modl3(nx = nx, ny = ny, nc = nc, num_block = num_block,  nLayers = nLayers, num_filters=   num_filters , esp = esp)

    # functions and variables
    oForward    =   pForward()
    rmbg        =   rm_bg()
    PHam        =   phase_hamming()
    Cphs        =   C_phase()
    Fphs        =   F_phase()

    # define variable
    # fftmtx      =   tf.convert_to_tensor(np.fft.fftshift(dft(ny),axes=(0,1))/ny, dtype=tf.complex64)
    fftmtx      =   tf.convert_to_tensor(np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.eye(ny,dtype=np.complex64),axes=0),axis=0),axes=0),dtype=tf.complex64) # / np.sqrt(ny)
    timeline    =   tf.convert_to_tensor(np.arange(ny,dtype=np.complex64) * esp, dtype=tf.complex64)

    # BUDA
    [out5,out6, fm_est]     =   recon_joint([input_c, input_m_trn1, input_m_trn2, input_k_trn1, input_k_trn2])
    out7        =   K.expand_dims(r2c(out5),axis=-1)
    out8        =   K.expand_dims(r2c(out6),axis=-1)

    # phase estimatation using hamming filter
    [rimg1,pimg1]  =   PHam([out7])
    [rimg2,pimg2]  =   PHam([out8])

    # combine imgs
    comb_img    =   (rimg1 + rimg2) / 2.0

    # loss points in k-space
    lss_1st     =   oForward([Fphs([comb_img,pimg1]),     input_c,    input_m_lss1+input_m_trn1,  K.cast(fm_est,tf.complex64), fftmtx, timeline, True])
    lss_2nd     =   oForward([Fphs([comb_img,pimg2]),     input_c,    input_m_lss2+input_m_trn2,  K.cast(fm_est,tf.complex64), fftmtx, timeline, False])

    # loss points in k-space
    lss_l1      =   K.sum(K.abs(lss_1st - input_k_lss1 - input_k_trn1),axis=1) / (K.sum(K.abs(input_k_lss1+input_k_trn1))+K.sum(K.abs(input_k_lss2+input_k_trn2)))   \
                    + K.sum(K.abs(lss_2nd - input_k_lss2 - input_k_trn2),axis=1) / (K.sum(K.abs(input_k_lss1+input_k_trn1))+K.sum(K.abs(input_k_lss2+input_k_trn2)))
    lss_l2      =   K.sum(K.square(K.abs(lss_1st - input_k_lss1 - input_k_trn1)),axis=1) / K.sqrt(K.sum(K.square(K.abs(input_k_lss1+input_k_trn1))+K.square(K.abs(input_k_lss2+input_k_trn2))))   \
                    + K.sum(K.square(K.abs(lss_2nd - input_k_lss2 - input_k_trn2)),axis=1) / K.sqrt(K.sum(K.square(K.abs(input_k_lss1+input_k_trn1))+K.square(K.abs(input_k_lss2+input_k_trn2))))

    # # loss points in image domain
    # syn_1st     =   tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(lss_1st,[-2,-1])),[-2,-1]) * tf.math.sqrt(tf.cast(nx*ny,tf.complex64))
    # syn_2nd     =   tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(lss_2nd,[-2,-1])),[-2,-1]) * tf.math.sqrt(tf.cast(nx*ny,tf.complex64))
    # ref_1st     =   tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(input_k_trn1+input_k_lss1,[-2,-1])),[-2,-1]) * tf.math.sqrt(tf.cast(nx*ny,tf.complex64))
    # ref_2nd     =   tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(input_k_trn2+input_k_lss2,[-2,-1])),[-2,-1]) * tf.math.sqrt(tf.cast(nx*ny,tf.complex64))

    # # normalized loss (in image domain)
    # lss_l1      =   K.sum(K.abs(syn_1st - ref_1st),axis=1) / (K.sum(K.abs(ref_1st))+K.sum(K.abs(ref_2nd)))   \
    #               + K.sum(K.abs(syn_2nd - ref_2nd),axis=1) / (K.sum(K.abs(ref_1st))+K.sum(K.abs(ref_2nd)))
    # lss_l2      =   K.sum(K.square(K.abs(syn_1st - ref_1st)),axis=1) / K.sqrt(K.sum(K.square(K.abs(ref_1st)))+K.sum(K.square(K.abs(ref_2nd))))  \
    #               + K.sum(K.square(K.abs(syn_2nd - ref_2nd)),axis=1) / K.sqrt(K.sum(K.square(K.abs(ref_1st)))+K.sum(K.square(K.abs(ref_2nd))))


    # outputs
    out_final   =   Concatenate(axis=-1)([  out7,   out8,                           \
                                            K.cast(fm_est,tf.complex64),          \
                                            comb_img,                               \
                                            K.cast(K.expand_dims(lss_l1,axis=-1),tf.complex64),          \
                                            K.cast(K.expand_dims(lss_l2,axis=-1),tf.complex64),          \
                                          ])

    return Model(inputs     =   [ input_c,  input_k_trn1,   input_k_trn2,   input_k_lss1,   input_k_lss2,
                                            input_m_trn1,   input_m_trn2,   input_m_lss1,   input_m_lss2       ],
                 outputs    =   [ out_final  ],
                 name       =   'buda-net-v04' )


##########################################################
# %%
# custom loss
##########################################################


def loss_custom_v01(y_true, y_pred):

    # l1 norm
    l1      =   50 * K.sum(K.abs(y_pred[...,-2]))
    # l2 norm
    l2      =   50 * K.sqrt(K.sum(K.abs(y_pred[...,-1])))

    return ( l1 + l2 )
