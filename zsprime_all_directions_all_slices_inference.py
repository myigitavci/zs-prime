# -*- coding: utf-8 -*-

##########################################################
# %%
# import libraries
##########################################################
import  numpy               as      np
import  glob                as      gl
import  nibabel             as      nib
import  utils.library_utils       as      mf
import  utils.library_net         as      mn
from    utils.library_dat     import  data_generator
from    scipy.io            import  savemat
import  time
from    scipy.io            import  savemat
from    scipy.io            import  loadmat
from    matplotlib          import  pyplot   as plt
from tensorflow.keras.optimizers import  Adam
from tensorflow.keras.callbacks  import  ModelCheckpoint, EarlyStopping
import tensorflow as tf

##########################################################
# %%
# setting paths
##########################################################
# change the path to correspond to the folder in your Google Drive
path_drive = '/u/home/avm/ssl_buda_net_new/'
path_save='/u/home/avm/ssl_buda_net-master/'
experiment='experiments/new_data_nb_6_oldest_net_layer_16_filter_32_lam005_cg10_1e7_split5_1d_sample_jaejin_recon_field_from_echo2_all_directions_all_slices'
#nb_1_zssl_net_layer_8_filter_16_lam_008
virtual_coil_bool=0
batch_size  =   1
num_block   =   6 #8
slc_select=19
#mixed_precision.set_global_policy('mixed_float16')
import os, sys
os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)
sys.path.append(path_drive+'utils')
sys.path.append(path_drive+'utils')
path_raw    = path_drive+'data/compressed/'
path_ref    = path_drive+'data/ref/'
path_field  = path_drive+'data/fields/'
path_coil   = path_drive+'data/coil/'
path_net    = path_save+'network/'
path_fig_save    = path_save+experiment+'/'

if not os.path.exists(path_fig_save):
    os.makedirs(path_fig_save)
if not os.path.exists(path_net+experiment):
    os.makedirs(path_net+experiment)    

##########################################################
# %%
# imaging parameters
##########################################################
num_slc, nx, ny, nc,nd     =   32, 220, 220, 12,32
esp                     =   1.90e-1 / 1000  #   msec / 10
#esp                     =   1.88e-3 / 10  #   msec / 10

model_name              =   path_net + experiment+'/ssl_buda_net_v02.h5'
hist_name               =   path_net + experiment+'/ssl_buda_net_v02.npy'

num_epoch   =   int(50)
num_layer   =   16  #16
num_filter  =   32 #46
loss_c      =   mn.loss_custom
shot_select =   [0,5]
num_split   =   1
num_val     =   1
rho_val     =   0.2
rho_trn     =   0.4

##########################################################
# %%
# load data
##########################################################
model = mn.create_zero_MIRID_model(     nx =   nx,
                                        ny =   ny,
                                        nc =   nc,
                                        num_block   =   num_block,
                                        nLayers     =   num_layer,
                                        virtual_coil=   virtual_coil_bool,
                                        num_filters =   num_filter,
                                        esp         =   esp
                                        )
# Define an optimizer
adam_opt = Adam(learning_rate=1e-3-9e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-09,decay=1e-5)

# Compile the model
model.compile(optimizer=adam_opt, loss=loss_c)

try:
    print(model_name)
    model.load_weights(model_name)
    print('succeeded to load the model')
except:
    print('failed to load the model')

# file names
list_truth = sorted(gl.glob(path_ref + '/loraks_*.mat'))
list_kdata = sorted(gl.glob(path_raw + 'set*.mat'))
list_field = sorted(gl.glob(path_field  + 'field_*.nii.gz'))

# load csm
import scipy.io
print('loading coil sensitivity map')
csm = np.transpose(scipy.io.loadmat('/u/home/avm/ssl_buda_net_new/data/csm_compressed_eigen_th_08.mat')['sens'], axes=(2, 3, 0, 1))
csm=csm[2:34,]
#csm     =   np.transpose(mf.load_h5py('/u/home/avm/ssl_buda_net/data/csm_compressed_eigen_th_08.mat')['sens'], axes=(2, 3, 0, 1))
#csm     =   csm['real'] + 1j*csm['imag']

for direction in range(len(list_kdata)):

    print('loading data in the direction # ' + str(direction+1)+'data: '+str(list_kdata[direction]))
    kdata       =   np.transpose(scipy.io.loadmat(list_kdata[direction])['kdata_gcc'],  axes=(4, 2, 0, 1, 3,5))
    kdata=np.squeeze(np.stack([np.squeeze(kdata[:,:,:,:,0,0,np.newaxis]),np.squeeze(kdata[:,:,:,:,2,1,np.newaxis])],axis=-1))
    kdata=kdata[2:34,]
      
    """ generate mask with all nonzeros through slices and all channels"""
    print('finding k-space mask')
    mask_all            =   np.zeros([1,nx,ny,len(shot_select)],   dtype=np.complex64)
    ind_non_z           =   np.nonzero(np.reshape(np.sum(np.sum(np.abs(kdata)>5e-8,1),0),mask_all.shape))
    mask_all[ind_non_z] =   1
    del ind_non_z
   # truth       =   np.transpose(scipy.io.loadmat(list_truth[direction])['img'],    axes=(2, 0, 1, 3))[:,:,:,shot_select]
   # truth_all[direction*num_slc:(direction+1)*num_slc,:,:,:]    =   truth
    field       =   np.transpose(nib.load(list_field[direction]).get_fdata(), axes=(2,0,1))
    field=field[4:36]
    field=field[:,:,:,np.newaxis]

    print('loading data - done')

    kdata   =    4e5  * kdata


    ##########################################################
    # %%
    # interfere
    ##########################################################
    ## here nd???
    print("mask_all_tiled:"+str(np.ndim(np.tile(mask_all[...,0], (nd,1,1)))))
    print("nd:"+str(nd))
    tst_par = { 'kdata_all'     : kdata,
                'csm'           : csm,
                'mask_trn1'     : np.tile(mask_all[...,0], (nd,1,1)),
                'mask_trn2'     : np.tile(mask_all[...,1], (nd,1,1)),
                'mask_lss1'     : np.tile(mask_all[...,0], (nd,1,1)),
                'mask_lss2'     : np.tile(mask_all[...,1], (nd,1,1)),
                'b0_map'        : field,                                          }

    tst_dat =   data_generator(**tst_par)
    pred    =   model.predict(tst_dat,verbose=1)
    print("Pred Data Shape:")
    print(pred.shape)
    recon_all = pred[...,0:2]
    recon_dif = mf.msos(np.transpose(recon_all,(1,2,0,3)),axis=-1)
    #recon_all=recon_all[:,:,13,:]

    #recon_dwi = mf.msos(recon_dif,axis=-1)
    savemat(path_fig_save+str(direction)+'example_results.mat', {"msEPI": recon_all, 'dif': recon_dif  })

