##########################################################
# %%
# import libraries
##########################################################

import numpy                as  np
import tensorflow           as  tf

# keras
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Concatenate
from tensorflow.keras.models import Model

from utils.library_net_function  import *
from utils.library_net_recon_last     import create_recon

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
# network
##########################################################
    
def create_zero_MIRID_model(nx, ny, nc, nLayers, num_block, virtual_coil,num_filters = 64, esp = 0):

    # define the inputs    
    input_c         = Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_c')
    input_k_trn1    = Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_trn1') 
    input_k_trn2    = Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_trn2') 
    input_k_lss1    = Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_lss1') 
    input_k_lss2    = Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_lss2') 
    input_m_trn1    = Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_trn1') 
    input_m_trn2    = Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_trn2') 
    input_m_lss1    = Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_lss1') 
    input_m_lss2    = Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_lss2') 
    input_fm        = Input(shape=(nx,ny,1),    dtype = tf.float32,         name = 'input_fm')

    recon_joint     = create_recon(nx = nx, ny = ny, nc = nc, num_block = num_block,virtual_coil=virtual_coil, nLayers = nLayers, num_filters=   num_filters,  esp = esp)
                                           
    # functions and variables

    oForward    =   pForward()

    # define variable
    # fftmtx      =   tf.convert_to_tensor(np.fft.fftshift(dft(ny),axes=(0,1))/ny, dtype=tf.complex64)
    #----ask----
    fftmtx      =   tf.convert_to_tensor(np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.eye(ny,dtype=np.complex64),axes=0),axis=0),axes=0),dtype=tf.complex64) # / np.sqrt(ny)
    timeline    =   tf.convert_to_tensor(np.arange(ny,dtype=np.complex64) * esp, dtype=tf.complex64)
    
    # Joint Recon
    [out1,out2,krg1,krg2,dc1,dc2,rg1,rg2,Atb1,intermediate_results] =   recon_joint([input_c, input_m_trn1, input_m_trn2, input_k_trn1, input_k_trn2, input_fm]) 
    out3        =   K.expand_dims(r2c(out1),axis=-1)
    out4        =   K.expand_dims(r2c(out2),axis=-1)    

    # loss points in k-space
    #----ask---- what is this function exactly doing?
    lss_1st     =   oForward([out3, input_c,    input_m_lss1+input_m_trn1,  K.cast(input_fm,tf.complex64), fftmtx, timeline, True])
    lss_2nd     =   oForward([out4, input_c,    input_m_lss2+input_m_trn2,  K.cast(input_fm,tf.complex64), fftmtx, timeline, False])

    # loss \ in k-space
    lss_l1      =   K.sum(K.abs(lss_1st - input_k_lss1 - input_k_trn1),axis=1) / (K.sum(K.abs(input_k_lss1+input_k_trn1))+K.sum(K.abs(input_k_lss2+input_k_trn2)))   \
                    + K.sum(K.abs(lss_2nd - input_k_lss2 - input_k_trn2),axis=1) / (K.sum(K.abs(input_k_lss1+input_k_trn1))+K.sum(K.abs(input_k_lss2+input_k_trn2)))  
    lss_l2      =   K.sum(K.square(K.abs(lss_1st - input_k_lss1 - input_k_trn1)),axis=1) / K.sqrt(K.sum(K.square(K.abs(input_k_lss1+input_k_trn1))+K.square(K.abs(input_k_lss2+input_k_trn2))))   \
                    + K.sum(K.square(K.abs(lss_2nd - input_k_lss2 - input_k_trn2)),axis=1) / K.sqrt(K.sum(K.square(K.abs(input_k_lss1+input_k_trn1))+K.square(K.abs(input_k_lss2+input_k_trn2))))   
    # lss_l1      =   K.sum(K.abs(lss_1st - input_k_lss1 - input_k_trn1),axis=1) / (K.sum(K.abs(input_k_lss1+input_k_trn1))+K.sum(K.abs(input_k_lss2+input_k_trn2)))   \
    #                 + K.sum(K.abs(lss_2nd - input_k_lss2 - input_k_trn2),axis=1) / (K.sum(K.abs(input_k_lss1+input_k_trn1))+K.sum(K.abs(input_k_lss2+input_k_trn2)))  
    # lss_l2      =   K.sum(K.square(K.abs(lss_1st - input_k_lss1 - input_k_trn1)),axis=1) / (K.sum(K.square(K.abs(input_k_lss1+input_k_trn1))+K.square(K.abs(input_k_lss2+input_k_trn2))))   \
    #                 + K.sum(K.square(K.abs(lss_2nd - input_k_lss2 - input_k_trn2)),axis=1) / (K.sum(K.square(K.abs(input_k_lss1+input_k_trn1))+K.square(K.abs(input_k_lss2+input_k_trn2))))     
    #outputs
    out_final   =   Concatenate(axis=-1)([  out3,   out4,   \
                                            K.cast(K.expand_dims(lss_l1,axis=-1),tf.complex64),          \
                                            K.cast(K.expand_dims(lss_l2,axis=-1),tf.complex64),          \

                                            K.expand_dims(krg1,axis=-1),K.expand_dims(krg2,axis=-1),K.expand_dims(dc1,axis=-1),K.expand_dims(dc2,axis=-1),K.expand_dims(rg1,axis=-1),K.expand_dims(rg2,axis=-1),K.expand_dims(Atb1,axis=-1),intermediate_results
                                           # K.cast(K.expand_dims(tf.reduce_max(K.abs(input_k_lss1),axis=0),axis=-1),tf.complex64),          \
                                           # K.cast(K.expand_dims(tf.reduce_max(K.abs(input_k_trn1),axis=0),axis=-1),tf.complex64), \
                                           # K.cast(K.expand_dims(tf.reduce_max(K.abs(lss_1st),axis=0),axis=-1),tf.complex64)


                                          ])   
    
    return Model(inputs     =   [ input_c,  input_k_trn1,   input_k_trn2,   input_k_lss1,   input_k_lss2,    
                                            input_m_trn1,   input_m_trn2,   input_m_lss1,   input_m_lss2,   input_fm  ],
                 outputs    =   [ out_final  ],
                 name       =   'zero-MIRID' )


##########################################################
# %%
# custom loss
##########################################################

def loss_custom(y_true, y_pred):
    
    # l1 norm
    l1      =   K.sum(K.abs(y_pred[...,2]))  
    # l2 norm
    l2      =   K.sqrt(K.sum(K.abs(y_pred[...,3]))) 

    return ( l1 + l2 )  

def loss_with_mc(y_true, y_pred):
    weight_mc=2e-7
   # weight_mc = tf.cast(weight_mc, tf.complex64)

    # l1 norm
    l1      =   K.sum(K.abs(y_pred[...,2]))  
    # l2 norm
    l2      =   K.sqrt(K.sum(K.abs(y_pred[...,3]))) 
        # magnitude constraint
    #y_pred_complex1 = tf.cast(y_pred[..., 0], tf.complex64)
    #y_pred_complex2 = tf.cast(y_pred[..., 1], tf.complex64)

    mc = weight_mc * K.sum(K.abs(K.abs(y_pred[..., 0]) - K.abs(y_pred[..., 1])))
    loss    =  l1+l2+mc
    return ( loss )  