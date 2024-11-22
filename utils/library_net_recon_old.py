##########################################################
# %%
# import libraries
##########################################################

import numpy                as  np
import tensorflow           as  tf
from    matplotlib          import  pyplot   as plt

# keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Add, Concatenate
from tensorflow.keras.models import Model
from scipy.linalg import dft
import tensorflow.keras
from utils.library_net_function import *


##########################################################
# %%
# SENSE
##########################################################
class Aclass:
    def __init__(self, csm,mask,field,fmt,timeline,bool_pol,lam):
        with tf.name_scope('Ainit2'):
            self.mask       =       mask
            self.csm        =       csm
            self.lam        =       lam
            self.timeline   =       tf.cast(timeline, tf.float32)
            time_eff        =       tf.cond(bool_pol, lambda: self.timeline, lambda: tf.reverse(self.timeline, axis = [0]))
            fm_phase        =       tf.constant(2 * np.pi, dtype=tf.float32) * tf.linalg.matmul(tf.expand_dims(time_eff,axis=1),tf.expand_dims(tf.cast(tf.squeeze(field), dtype=tf.float32) ,axis=0))
            fm_exp          =       tf.complex(tf.math.cos(fm_phase),tf.math.sin(fm_phase))
            self.fmt        =       tf.math.multiply(fmt * tf.expand_dims(self.mask, axis=-1), fm_exp)
    @tf.function 
    def myAtA(self,img):
        with tf.name_scope('AtA2'):
            # csm : [y,c]
            # img : [y,]
            coilImages      =       self.csm * tf.expand_dims(img,axis=-1)
            # kspace : [y,c]
            # kspace          =       tf.transpose(tf.signal.fftshift(tf.signal.fft(tf.signal.fftshift(tf.transpose(coilImages,perm=[1,0]),axes=1)),axes=1),perm=[1,0])
            #harcoded
            kspace          =       tf.linalg.matmul( self.fmt, coilImages)
            # mask : [y,]
            # temp : [y,c]
            temp            =       kspace*tf.expand_dims(self.mask, axis=-1)
            # coilImgs : [y,c]
            coilImgs        =       tf.linalg.matmul( self.fmt, temp, adjoint_a=True )

            # coilImgs        =       tf.transpose(tf.signal.ifftshift(tf.signal.ifft(tf.signal.ifftshift(tf.transpose(temp,perm=[1,0]),axes=1)),axes=1),perm=[1,0])
            # coilComb : [y,c]
            coilComb        =       tf.reduce_sum(coilImgs*tf.math.conj(self.csm),axis=-1,name="aclass_red_sum")
            # coilComb : [y,]
            coilComb        =       coilComb+self.lam*img
        return coilComb
    

def myCG(A,rhs):
    rhs=r2c(rhs)
    cond=lambda i,rTr,*_: tf.logical_and( tf.less(i,10), rTr>1e-8)
    def body(i,rTr,x,r,p):
        with tf.name_scope('cgBody'):
            Ap      =   A.myAtA(p)
            # alpha   =   rTr / tf.cast(tf.reduce_sum(tf.math.conj(p)*Ap),dtype=tf.float32)
            alpha   =   rTr / tf.math.real(tf.reduce_sum(tf.math.conj(p)*Ap,name="CG_red_sum3"))
            alpha   =   tf.complex(alpha,0.)
            x       =   x + alpha * p
            r       =   r - alpha * Ap
            # rTrNew  =   tf.cast( tf.reduce_sum(tf.math.conj(r)*r),dtype=tf.float32)
            rTrNew  =   tf.math.real( tf.reduce_sum(tf.math.conj(r)*r,name="CG_red_sum2"))
            beta    =   rTrNew / rTr
            beta    =   tf.complex(beta,0.)
            p       =   r + beta * p
        return i+1,rTrNew,x,r,p

    x       =   tf.zeros_like(rhs)
    i,r,p   =   0,rhs,rhs
    # rTr     =   tf.cast( tf.reduce_sum(tf.math.conj(r)*r),dtype=tf.float32)
    rTr     =   tf.math.real( tf.reduce_sum(tf.math.conj(r)*r,name="CG_red_sum") )
    loopVar =   i,rTr,x,r,p
    out     =   tf.while_loop(cond,body,loopVar,name='CGwhile',parallel_iterations=1)[2]
    return c2r(out)


class myDC(Layer):

    def __init__(self, **kwargs):
        super(myDC, self).__init__(**kwargs)
        self.lam1 = self.add_weight(name='lam1', shape=(1,), initializer=tf.constant_initializer(value=0.05),
                                     dtype='float32', trainable=False)
        self.lam2 = self.add_weight(name='lam2', shape=(1,), initializer=tf.constant_initializer(value=0.05),
                                     dtype='float32', trainable=False)

    def build(self, input_shape):
        super(myDC, self).build(input_shape)
    @tf.function 
    def call(self, x):
        rhs, csm, mask, field, fmt, timeline, bool_pol = x
        lam3 = tf.complex(self.lam1+ self.lam2, 0.)

        def fn1(tmp1):
            c1, m1, f1, r1 = tmp1
            # [c,x,y] -> [x,y,c]
            in_tmp = (tf.transpose(c1,perm=[1,2,0]),m1,f1,r1)

            def fn2(tmp2):
                c2,m2,f2,r2 = tmp2
                Aobj = Aclass(c2, m2, f2, fmt, timeline, bool_pol, lam3)
                y2 = myCG(Aobj, r2)
                return y2

            y1 = tf.map_fn(fn2, in_tmp, dtype=tf.float32, name='mapFn_sub1')
            return y1

        inp = (csm, mask, field, rhs )
        # Mapping functions with multi-arity inputs and outputs
        rec = tf.map_fn(fn1, inp, dtype=tf.float32, name='mapFn2')
        return rec

    def lam_weight(self, x):
        in0, in1 = x
        res = self.lam1 * in0 + self.lam2 * in1
        return res


class Aty(Layer):
    def __init__(self, **kwargs):
        super(Aty, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Aty, self).build(input_shape)
    @tf.function 
    def call(self, x):
        kdata, csm, mask, field, fmt, tl, bool_updown   =       x
        time_eff    =   tf.cond(bool_updown, lambda: tl, lambda: tf.reverse(tl, axis = [0]))

        def backward1(tmp1):
            k1, c1, m1, f1 = tmp1
            # [c,x,y] -> [c,y,x]
            k11 = tf.transpose(k1,perm=[0,2,1])
            # ifft shift along kx
            k12 = tf.signal.ifftshift(tf.signal.ifft(tf.signal.ifftshift(k11,axes=-1)),axes=-1)
            # [c,y,x] -> [x,y,c]
            k13 = tf.transpose(k12,perm=[2,1,0])
            # cast field
            f11     =   tf.cast(tf.squeeze(f1), dtype=tf.complex64)
            # [c,x,y] -> [x,y,c]
            c11 = tf.transpose(c1,perm=[1,2,0])
            # input
            in_tmp  =   (k13,c11,m1,f11)

            # processes just one line ky direction    
            def backward2(tmp2):
                k2, c2, m2, f2 = tmp2
                # find encoding matrix
                # refer to docs to understand how field map works
                fm_phase    =   tf.constant(2.0 * np.pi, dtype=tf.complex64) *  tf.linalg.matmul( tf.expand_dims(time_eff,axis=1) , tf.expand_dims(f2,axis=0) )
                fm_exp      =   tf.math.exp(1j*tf.squeeze(fm_phase))
                fmt_sub     =   fmt * tf.expand_dims(m2, axis=-1) * fm_exp
                # At
                ks          =   k2 * tf.expand_dims(m2,axis=-1)
                ci          =   tf.linalg.matmul(fmt_sub,ks,adjoint_a=True)
                # ci  =   tf.transpose(tf.signal.ifftshift(tf.signal.ifft(tf.signal.ifftshift(tf.transpose(ks,perm=[1,0]),axes=1)),axes=1),perm=[1,0])
                y2  =   tf.reduce_sum(ci*tf.math.conj(c2),axis=-1,name="aty_red_sum")
                return tf.squeeze(y2)

            y1 = tf.map_fn(backward2, in_tmp, dtype=tf.complex64, name='mapBack3')
            return y1

        inp = (kdata, csm, mask, field)
        rec = tf.map_fn(backward1, inp, dtype=tf.complex64, name='mapBack2')

        return rec
    
def create_recon(nx, ny, nc, nLayers, num_block, virtual_coil, num_filters = 64,  esp = 0):
    # define the inputs
    input_c     =   Input(shape=(nc,nx,ny), dtype = tf.complex64,   name = 'input_c')
    input_m1    =   Input(shape=(nx,ny),    dtype = tf.complex64,   name = 'input_m1')
    input_m2    =   Input(shape=(nx,ny),    dtype = tf.complex64,   name = 'input_m2')
    input_k1    =   Input(shape=(nc,nx,ny), dtype = tf.complex64,   name = 'input_k1')
    input_k2    =   Input(shape=(nc,nx,ny), dtype = tf.complex64,   name = 'input_k2')
    input_field =   Input(shape=(nx,ny),    dtype = tf.float32,     name = 'field_estimate')

    # define variable
    # fftmtx -> fft for y direction
    # timeline 
    fftmtx      =   tf.convert_to_tensor(np.fft.fftshift(dft(ny),axes=(0,1)), dtype=tf.complex64)
    timeline    =   tf.convert_to_tensor(np.arange(ny,dtype=np.complex64) * esp, dtype=tf.complex64)

    # define functions
    UpdateDC    =   myDC()
    rmbg        =   rm_bg()
    calc_Aty    =   Aty()
    myFFT       =   tfft2()
    myIFFT      =   tifft2()

    # calc Atb
    Atb1        =   calc_Aty([input_k1,input_c,input_m1,input_field,fftmtx,timeline,tf.constant(True)])
    Atb2        =   calc_Aty([input_k2,input_c,input_m2,input_field,fftmtx,timeline,tf.constant(False)])

    # calc init
    dc1         =   c2r(Atb1)
    dc2         =   c2r(Atb2)
    intermediate_results = None  # Initialize as None
    # define networks
    # for vc 3rd channel is 4, without vc is 2
    if virtual_coil is True:
        RegConv_k   =   RegConvLayers(nx,ny,4,nLayers,num_filters)
        RegConv_i   =   RegConvLayers(nx,ny,4,nLayers,num_filters)
    else:
        RegConv_k   =   RegConvLayers(nx,ny,2,nLayers,num_filters)
        RegConv_i   =   RegConvLayers(nx,ny,2,nLayers,num_filters)

    # # loop
    for blk in range(0,num_block):
    #     # VC coils use complex-conjugate information to enhance the reconstruction:
    #     # https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.21652
         if virtual_coil is True:
    #     # concat shots with VC
        #     dc_cat_i    = Concatenate(axis=-1)([dc1,dc2,tconj(dc1),tconj(dc2)]) # tconj later
    
             dc_cat_k    = Concatenate(axis=-1)([myFFT([dc1]),myFFT([dc2]),myFFT([tconj(dc1)]),myFFT([tconj(dc2)])])
         else:
    #     # no vc
            dc_cat_i    = Concatenate(axis=-1)([dc1,dc2]) # tconj later
            dc_cat_k    = Concatenate(axis=-1)([myFFT([dc1]),myFFT([dc2])]) 

    #     # CNN Regularization
         rg_term_i   = RegConv_i(dc_cat_i)
         rg_term_k   = RegConv_k(dc_cat_k)

    #     #separate shots    
         if virtual_coil is True:
    #         # get rid of vc
             irg1        = (rg_term_i[:,:,:,0:2] + tconj(rg_term_i[:,:,:,4:6]))/2
             irg2        = (rg_term_i[:,:,:,2:4] + tconj(rg_term_i[:,:,:,6:8]))/2
             krg1        = (myIFFT([rg_term_k[:,:,:,0:2]]) + tconj(myIFFT([rg_term_k[:,:,:,4:6]])))/2
             krg2        = (myIFFT([rg_term_k[:,:,:,2:4]]) + tconj(myIFFT([rg_term_k[:,:,:,6:8]])))/2
         else:
      
             irg1        = (rg_term_i[:,:,:,0:2] )
             irg2        = (rg_term_i[:,:,:,2:4] )
             krg1        = (myIFFT([rg_term_k[:,:,:,0:2]]) )
             krg2        = (myIFFT([rg_term_k[:,:,:,2:4]]) ) 

         rg1         = UpdateDC.lam_weight([irg1,krg1])
         rg2         = UpdateDC.lam_weight([irg2,krg2])
        
        #  rg1         = UpdateDC.lam_weight([0,irg1])
        #  rg2         = UpdateDC.lam_weight([0,irg2])
        # ----ask----
        # Atb + lam1*[denoised kspace] +lam2*[denoised img]
        # adding image space and kspace terms does not make full sense to me
        # AtA update
         rg1         = Add()([c2r(Atb1), rg1])
         rg2         = Add()([c2r(Atb2), rg2])

        # Update DC
         dc1         = UpdateDC([rg1, input_c, input_m1, input_field, fftmtx, timeline, tf.constant(True)])
         dc2         = UpdateDC([rg2, input_c, input_m2, input_field, fftmtx, timeline, tf.constant(False)])

         new_tensors = Concatenate(axis=-1)([
                K.expand_dims(r2c(dc1), axis=-1), 
                K.expand_dims(r2c(rg1), axis=-1), 
                K.expand_dims(r2c(irg1), axis=-1), 
                K.expand_dims(r2c(irg2), axis=-1),
                K.expand_dims(r2c(dc2), axis=-1), 
                K.expand_dims(r2c(rg2), axis=-1) 

            ])
            
         if intermediate_results is None:
            # Initialize on the first iteration
            intermediate_results = new_tensors
         else:
            # Concatenate with existing results
            intermediate_results = Concatenate(axis=-1)([intermediate_results, new_tensors])    
    # remove background
    out1 = rmbg([dc1,input_c])
    out2 = rmbg([dc2,input_c])


    return Model(inputs     =   [ input_c, input_m1, input_m2, input_k1, input_k2, input_field],
                 outputs    =   [ out1, out2,r2c(irg1),r2c(irg2),r2c(dc1),r2c(dc2),r2c(rg1),r2c(rg2),Atb1,intermediate_results],
                 name       =   'RECON' )
