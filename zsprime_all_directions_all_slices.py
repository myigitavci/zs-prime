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

# load kdata
kdata_all   =   np.zeros([num_slc*len(list_kdata),nc,nx,ny,len(shot_select)],   dtype=np.complex64)
truth_all   =   np.zeros([num_slc*len(list_kdata),nx,ny,len(shot_select)],      dtype=np.complex64)
field_all   =   np.zeros([num_slc*len(list_kdata),nx,ny,1],                     dtype=np.complex64)
for direction in range(len(list_kdata)):

    print('loading data in the direction # ' + str(direction+1)+'data: '+str(list_kdata[direction]))
    kdata       =   np.transpose(scipy.io.loadmat(list_kdata[direction])['kdata_gcc'],  axes=(4, 2, 0, 1, 3,5))
    kdata=np.squeeze(np.stack([np.squeeze(kdata[:,:,:,:,0,0,np.newaxis]),np.squeeze(kdata[:,:,:,:,2,1,np.newaxis])],axis=-1))
    kdata=kdata[2:34,]
    kdata_all[direction*num_slc:(direction+1)*num_slc,:,:,:,:]  =   kdata
   # truth       =   np.transpose(scipy.io.loadmat(list_truth[direction])['img'],    axes=(2, 0, 1, 3))[:,:,:,shot_select]
   # truth_all[direction*num_slc:(direction+1)*num_slc,:,:,:]    =   truth
    field       =   np.transpose(nib.load(list_field[direction]).get_fdata(), axes=(2,0,1))
    field=field[4:36]
    field_all[direction*num_slc:(direction+1)*num_slc,:,:,:]    =   field[:,:,:,np.newaxis]

print('loading data - done')

# scaling
# hard coding
kdata_all   =    4e5  * kdata_all
#truth_all   =    4e5  * truth_all / 220
#field_all   =    1e-2 * field_all
del kdata

##########################################################
# %%
# select one slice
##########################################################
#kdata_all   =   kdata_all[slc_select::num_slc,]
#truth_all   =   truth_all[slc_select::num_slc,]
#field_all   =   field_all[slc_select::num_slc,]
#csm         =   csm[slc_select::num_slc,]
# ##########################################################
# # %%
# # load data
# ##########################################################

# # file names
# list_truth = sorted(gl.glob(path_ref + '/loraks_*.mat'))
# list_kdata = sorted(gl.glob(path_raw + '/kdata_*.mat'))
# list_field = sorted(gl.glob(path_field  + '/topup_field_id01_correcte*.nii'))

# # load csm
# print('loading coil sensitivity map')
# csm     =   np.transpose(mf.load_h5py(path_coil + '/csm.mat')['sens'], axes=(2, 3, 0, 1))
# csm     =   csm['real'] + 1j*csm['imag']

# # load kdata
# kdata_all   =   np.zeros([num_slc*len(list_kdata),nc,nx,ny,len(shot_select)],   dtype=np.complex64)
# truth_all   =   np.zeros([num_slc*len(list_kdata),nx,ny,len(shot_select)],      dtype=np.complex64)
# field_all   =   np.zeros([num_slc*len(list_kdata),nx,ny,1],                     dtype=np.complex64)
# for direction in range(len(list_kdata)):
#     print('loading data in the direction # ' + str(direction+1))
#     kdata       =   np.transpose(mf.load_h5py(list_kdata[direction])['kdata'],  axes=(2, 3, 0, 1, 4))[:,:,:,:,shot_select]
#     kdata_all[direction*num_slc:(direction+1)*num_slc,:,:,:,:]  =   kdata['real'] + 1j*kdata['imag']
#     truth       =   np.transpose(mf.load_h5py(list_truth[direction])['img'],    axes=(2, 0, 1, 3))[:,:,:,shot_select]
#     truth_all[direction*num_slc:(direction+1)*num_slc,:,:,:]    =   truth['real'] + 1j*truth['imag']
#     field       =   np.transpose(nib.load(list_field[direction]).get_fdata(), axes=(2,0,1))
#     field_all[direction*num_slc:(direction+1)*num_slc,:,:,:]    =   field[:,:,:,np.newaxis]

# print('loading data - done')

# # scaling
# # hard coding
# kdata_all   =    4e5  * kdata_all
# truth_all   =    4e5  * truth_all / 220
# field_all   =    1e-2 * field_all
# del kdata, truth

# ##########################################################
# # %%
# # select one slice
# ##########################################################
# kdata_all   =   kdata_all[slc_select::num_slc,]
# truth_all   =   truth_all[slc_select::num_slc,]
# field_all   =   field_all[slc_select::num_slc,]
# csm         =   csm[slc_select::num_slc,]

##########################################################
# %%
# find EPI mask
##########################################################

""" generate mask with all nonzeros through slices and all channels"""
print('finding k-space mask')
mask_all            =   np.zeros([1,nx,ny,len(shot_select)],   dtype=np.complex64)
ind_non_z           =   np.nonzero(np.reshape(np.sum(np.sum(np.abs(kdata_all/(4e5))>5e-8,1),0),mask_all.shape))
mask_all[ind_non_z] =   1
del ind_non_z

##########################################################
# %%
# gen validation mask
##########################################################

print('generating validating mask')



mask_trn1 = np.empty((nd*num_slc, nx, ny), dtype=np.float32)
mask_trn2 = np.empty((nd*num_slc, nx, ny), dtype=np.float32)
mask_val1 = np.empty((nd*num_slc, nx, ny), dtype=np.float32)
mask_val2 = np.empty((nd*num_slc, nx, ny), dtype=np.float32)
for ii in range(nd*num_slc):
    kdata1 = np.transpose(kdata_all[ii,:,:,:,0],axes=(1,2,0))
    kdata2 = np.transpose(kdata_all[ii,:,:,:,1],axes=(1,2,0))

    
    mask_trn1[ii,...], mask_val1[ii,...] = mf.uniform_selection1d(kdata1, mask_all[0,:,:,0].real, rho_val)
    mask_trn2[ii,...], mask_val2[ii,...] = mf.uniform_selection1d(kdata2, mask_all[0,:,:,1].real, rho_val)

    ##########################################################
    # %%
    # gen training mask
    ##########################################################
    """ after generating 32 masks, we further divide them each mask to 50 random masks """
print('generating training mask')

mask_trn_split1 = np.empty((num_split*nd*num_slc, nx, ny), dtype=np.complex64)
mask_trn_split2 = np.empty((num_split*nd*num_slc, nx, ny), dtype=np.complex64)
mask_lss_split1 = np.empty((num_split*nd*num_slc, nx, ny), dtype=np.complex64)
mask_lss_split2 = np.empty((num_split*nd*num_slc, nx, ny), dtype=np.complex64)

for jj in range(nd*num_slc):
    kdata1 = np.transpose(kdata_all[jj,:,:,:,0],axes=(1,2,0))
    kdata2 = np.transpose(kdata_all[jj,:,:,:,1],axes=(1,2,0))
    mask1=mask_trn1[jj, ...]
    mask2=mask_trn2[jj, ...]
    for mm in range(num_split):
        mask_trn_split1[jj*num_split+mm, ...], mask_lss_split1[jj*num_split+mm, ...] = mf.uniform_selection1d(kdata1,np.copy(mask1),rho=rho_trn)
        mask_trn_split2[jj*num_split+mm, ...], mask_lss_split2[jj*num_split+mm, ...] = mf.uniform_selection1d(kdata2,np.copy(mask2),rho=rho_trn)

del ii, jj, kdata1, kdata2

print("mask_trn1_in_ain:"+str(mask_trn_split1.shape))


##########################################################
# %%
# define generator
##########################################################
trn_par = { 'kdata_all'     : kdata_all,
            'csm'           : csm,
            'mask_trn1'     : mask_trn_split1,
            'mask_trn2'     : mask_trn_split2,
            'mask_lss1'     : mask_lss_split1,
            'mask_lss2'     : mask_lss_split2,
            'b0_map'        : field_all,

            }
trn_dat =   data_generator(**trn_par)

val_par = { 'kdata_all'     : kdata_all,
            'csm'           : csm,
            'mask_trn1'     : mask_trn1,
            'mask_trn2'     : mask_trn2,
            'mask_lss1'     : mask_val1,
            'mask_lss2'     : mask_val2,
            'b0_map'        : field_all,
            }
val_dat =   data_generator(**val_par)
##########################################################
# %%
# load the network
##########################################################
print("K Data Shape:")
print(kdata_all.shape)
print("csm Shape:")
print(csm.shape)
print("field_all Shape:")
print(field_all.shape)
print("field_all Shape:")
print(mask_val1.shape)


##########################################################
# %%
# train the network
##########################################################
# tst_par = { 'kdata_all'     : kdata_all,
#             'csm'           : csm,
#             'mask_trn1'     : mask_trn_split1[0:1,...],
#             'mask_trn2'     : mask_trn_split2[0:1,...],
#             'mask_lss1'     : mask_lss_split1[0:1,...],
#             'mask_lss2'     : mask_lss_split2[0:1,...],
#             'b0_map'        : field_all,
#             'batch_size'    : batch_size,
#             'num_split'     : 0                                  }
kdata_test   =   kdata_all[slc_select::num_slc,]
csm_test         =   csm[slc_select::num_slc,]
field_test   =   field_all[slc_select::num_slc,]

tst_par = { 'kdata_all'     : kdata_test,
            'csm'           : csm_test,
            'mask_trn1'     : np.tile(mask_all[...,0], (1,1,1)),
            'mask_trn2'     : np.tile(mask_all[...,1], (1,1,1)),
            'mask_lss1'     : np.tile(mask_all[...,0], (1,1,1)),
            'mask_lss2'     : np.tile(mask_all[...,1], (1,1,1)),
            'b0_map'        : field_test,                                   }
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
#model.summary()

##########################################################
# %%
# callback
##########################################################

model_callback = ModelCheckpoint(   model_name,
                                    monitor           =   'val_loss',
                                    verbose           =   1,
                                    save_best_only    =   True,
                                    save_weights_only =   True,
                                    mode              =   'auto'    )

model_EarlyStop = EarlyStopping(    monitor                 =   "val_loss",
                                    min_delta               =   0,
                                    patience                =   5,
                                    verbose                 =   1,
                                    mode                    =   "auto",
                                    baseline                =   None,
                                    restore_best_weights    =   True,   )

try:
    print(model_name)
    model.load_weights(model_name)
    print('succeeded to load the model')
except:
    print('failed to load the model')
tst_dat =   data_generator(**tst_par)
class CustomPredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, tst_dat, mf, path_fig_save):
        super().__init__()
        self.tst_dat = tst_dat
        self.mf = mf
        self.path_fig_save = path_fig_save

    def on_epoch_end(self, epoch, logs=None):
        # Make predictions
        pred = self.model.predict(self.tst_dat)
        print("Pred Data Shape:")
        print(pred.shape)
        
        # Process predictions
        recon_all = pred[..., 0:2]
        recon_dif = self.mf.msos(np.transpose(recon_all, (1, 2, 0, 3)), axis=-1)
        
        # Save results to .mat file
        current_path_fig_save1 = f"{self.path_fig_save}/epoch_{epoch+1}_Pred1"
        current_path_fig_save2 = f"{self.path_fig_save}/epoch_{epoch+1}_Pred2"
        print(f"out3 {np.shape(np.squeeze(np.abs(pred[...,0])))}")
        print(f"out4 {np.shape(np.squeeze(np.abs(pred[...,1])))}")
        print(f"lss_l1 {np.shape(np.squeeze(np.abs(pred[...,2])))}")
        print(f"lss_l2 {np.shape(np.squeeze(np.abs(pred[...,3])))}")
        print(f"krg2 {np.shape(np.squeeze(np.abs(pred[...,4])))}")
        print(f"krg1 {np.shape(np.squeeze(np.abs(pred[...,5])))}")
        print(f"dc1 {np.shape(np.squeeze(np.abs(pred[...,6])))}")
        print(f"rg1 {np.shape(np.squeeze(np.abs(pred[...,7])))}")
        print(f"atb1 {np.shape(np.squeeze(np.abs(pred[...,8])))}")

        print(f"out3 {np.linalg.norm(np.squeeze(pred[...,0]))}")
        print(f"out4 {np.linalg.norm(np.squeeze(pred[...,1]))}")
        print(f"lss_l1 {np.linalg.norm(np.squeeze(pred[...,2]))}")
        print(f"lss_l2 {np.linalg.norm(np.squeeze(pred[...,3]))}")
        print(f"krg1 {np.linalg.norm(np.squeeze(pred[...,4]))}")
        print(f"krg2 {np.linalg.norm(np.squeeze(pred[...,5]))}")
        print(f"dc1 {np.linalg.norm(np.squeeze(pred[...,6]))}")
        print(f"dc2 {np.linalg.norm(np.squeeze(pred[...,7]))}")
        print(f"rg1 {np.linalg.norm(np.squeeze(pred[...,8]))}")
        print(f"rg2 {np.linalg.norm(np.squeeze(pred[...,9]))}")
        print(f"atb1 {np.linalg.norm(np.squeeze(pred[...,10]))}")
       # print(f"input_k_lss1 {np.squeeze(np.abs(pred[...,9])).max()}")
       # print(f"input_k_trn1 {np.squeeze(np.abs(pred[...,10])).max()}")
       # print(f"lss_1st {np.squeeze(np.abs(pred[...,11])).max()}")

          #savemat(current_path_fig_save + 'example_results.mat', {"msEPI": recon_all, 'dif': recon_dif})
        plt.imsave(f"{self.path_fig_save}/epoch_{epoch+1}_krg1.png",np.squeeze(np.abs(pred[...,4])),cmap='gray')
        plt.imsave(f"{self.path_fig_save}/epoch_{epoch+1}_krg2.png",np.squeeze(np.abs(pred[...,5])),cmap='gray')
          #savemat(current_path_fig_save + 'example_results.mat', {"msEPI": recon_all, 'dif': recon_dif})
        plt.imsave(f"{self.path_fig_save}/epoch_{epoch+1}_lss_l1.png",np.squeeze(np.abs(pred[...,2])),cmap='gray')
        plt.imsave(f"{self.path_fig_save}/epoch_{epoch+1}_lss_l2.png",np.squeeze(np.abs(pred[...,3])),cmap='gray')

        plt.imsave(f"{self.path_fig_save}/epoch_{epoch+1}_dc1.png",np.squeeze(np.abs(pred[...,6])),cmap='gray')
        plt.imsave(f"{self.path_fig_save}/epoch_{epoch+1}_dc2.png",np.squeeze(np.abs(pred[...,7])),cmap='gray')
        plt.imsave(f"{self.path_fig_save}/epoch_{epoch+1}_rg1.png",np.squeeze(np.abs(pred[...,8])),cmap='gray')
        plt.imsave(f"{self.path_fig_save}/epoch_{epoch+1}_rg2.png",np.squeeze(np.abs(pred[...,9])),cmap='gray')
        plt.imsave(f"{self.path_fig_save}/epoch_{epoch+1}_atb1.png",np.squeeze(np.abs(pred[...,10])),cmap='gray')
        recon_all = pred[...,0:2]
        recon_dif = mf.msos(np.transpose(recon_all,(1,2,0,3)),axis=-1)
        #recon_all=recon_all[:,:,13,:]

        #recon_dwi = mf.msos(recon_dif,axis=-1)
        savemat(path_fig_save+'example_results.mat', {"msEPI": recon_all, 'dif': recon_dif  })
        # Visualize and save mosaic
        self.mf.mosaic(np.rot90(np.abs(np.squeeze(recon_all[:,:,:,0]))),1,1,100,[0,10],current_path_fig_save1,'Pred1')
        self.mf.mosaic(np.rot90(np.abs(np.squeeze(recon_all[:,:,:,1]))),1,1,100,[0,10],current_path_fig_save2,'Pred2')
       # print(f"Results saved for epoch {epoch+1} at {current_path_fig_save1}")
        # List to store max values for the current epoch
        max_values = []

        # Loop to save images
        for i in range(0, num_block*6, 6):
            blk_num = int(i / 6)
            
            # Save images as before
            plt.imsave(f"{self.path_fig_save}/epoch_{epoch+1}_blk_{blk_num}_dc1.png", np.squeeze(np.abs(pred[..., 11+i])), cmap='gray')
            plt.imsave(f"{self.path_fig_save}/epoch_{epoch+1}_blk_{blk_num}_rg1.png", np.squeeze(np.abs(pred[..., 12+i])), cmap='gray')
            plt.imsave(f"{self.path_fig_save}/epoch_{epoch+1}_blk_{blk_num}_krg1.png", np.squeeze(np.abs(pred[..., 13+i])), cmap='gray')
            plt.imsave(f"{self.path_fig_save}/epoch_{epoch+1}_blk_{blk_num}_krg2.png", np.squeeze(np.abs(pred[..., 14+i])), cmap='gray')
            plt.imsave(f"{self.path_fig_save}/epoch_{epoch+1}_blk_{blk_num}_dc2.png", np.squeeze(np.abs(pred[..., 15+i])), cmap='gray')
            plt.imsave(f"{self.path_fig_save}/epoch_{epoch+1}_blk_{blk_num}_rg2.png", np.squeeze(np.abs(pred[..., 16+i])), cmap='gray')

            # Save max values for the current block
            max_dc1 = np.linalg.norm(np.squeeze(pred[..., 11+i]))
            max_rg1 = np.linalg.norm(np.squeeze(pred[..., 12+i]))
            max_krg1 = np.linalg.norm(np.squeeze(pred[..., 13+i]))
            max_krg2 = np.linalg.norm(np.squeeze(pred[..., 14+i]))
            max_dc2 = np.linalg.norm(np.squeeze(pred[..., 15+i]))
            max_rg2 = np.linalg.norm(np.squeeze(pred[..., 16+i]))
            
            # Append max values to the list
            max_values.append(f"Block {blk_num} - dc1: {max_dc1}, rg1: {max_rg1}, krg1: {max_krg1}, krg2: {max_krg2}, dc2: {max_dc2}, rg2: {max_rg2}")

        # Save max values to a text file after the loop
        with open(f"{self.path_fig_save}/epoch_{epoch+1}_max_values.txt", 'w') as f:
            f.write(f"Max values for epoch {epoch+1}:\n")
            f.write("\n".join(max_values))
class CustomPredictionCallback2(tf.keras.callbacks.Callback):
    def __init__(self, tst_dat, mf, path_fig_save):
        super().__init__()
        self.tst_dat = tst_dat
        self.mf = mf
        self.path_fig_save = path_fig_save

    def on_epoch_end(self, epoch, logs=None):
        # Make predictions
        pred = self.model.predict(self.tst_dat)
        print("Pred Data Shape:")
        print(pred.shape)

        print(f"lss_1st {np.squeeze(np.abs(pred[...,0])).max()}")
        print(f"lss_2nd {np.squeeze(np.abs(pred[...,1])).max()}")
        print(f"input_k_lss1 {np.squeeze(np.abs(pred[...,2])).max()}")
        print(f"input_k_trn1 {np.squeeze(np.abs(pred[...,3])).max()}")
        print(f"input_k_lss2 {np.squeeze(np.abs(pred[...,4])).max()}")
        print(f"input_k_trn2 {np.squeeze(np.abs(pred[...,5])).max()}")

        print(f"lss_1st {np.linalg.norm(np.squeeze(np.abs(pred[...,0])))}")
        print(f"lss_2nd {np.linalg.norm(np.squeeze(np.abs(pred[...,1])))}")
        print(f"input_k_lss1 {np.linalg.norm(np.squeeze(np.abs(pred[...,2])))}")
        print(f"input_k_trn1 {np.linalg.norm(np.squeeze(np.abs(pred[...,3])))}")
        print(f"input_k_lss2 {np.linalg.norm(np.squeeze(np.abs(pred[...,4])))}")
        print(f"input_k_trn2 {np.linalg.norm(np.squeeze(np.abs(pred[...,5])))}")
class CustomPredictionCallback3(tf.keras.callbacks.Callback):
    def __init__(self, tst_dat, mf, path_fig_save):
        super().__init__()
        self.tst_dat = tst_dat
        self.mf = mf
        self.path_fig_save = path_fig_save

    def on_epoch_end(self, epoch, logs=None):
        # Make predictions
        pred = self.model.predict(self.tst_dat)
        print("Pred Data Shape:")
        print(pred.shape)
        
        # Process predictions
        recon_all = pred[..., 0:2]
        recon_dif = self.mf.msos(np.transpose(recon_all, (1, 2, 0, 3)), axis=-1)
        savemat(path_fig_save+'example_results.mat', {"msEPI": recon_all, 'dif': recon_dif  })

# Usage
# Assuming 'tst_dat', 'mf', and 'path_fig_save' are defined elsewhere
callback = CustomPredictionCallback3(tst_dat, mf, path_fig_save)
print("Training starts!")
#print(model.weights[98])
#print(model.weights[99])
#print(model.weights[196])
#print(model.weights[197])
t_start     =   time.time()


with tf.device('GPU:0'):
    hist        =   model.fit(  trn_dat,
                            validation_data =   tst_dat,
                            epochs          =   num_epoch,
                            batch_size      =   batch_size,
                            verbose         =   1,
                            steps_per_epoch =   None,
                            callbacks       =   [model_EarlyStop,callback,model_callback] ,
                            shuffle= True   )

t_end       =   time.time()
print("Training finished!")
#print(model.weights[98])
#print(model.weights[99])

# model.save_weights(model_name)
np.save(hist_name,{'hist':hist.history,'train_time':t_end-t_start})
plt.figure(1)
# Plot training and validation loss
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

#Save the plot
plt.savefig(path_fig_save+'loss_plot.png')

##########################################################
# %%
# interfere
##########################################################
## here nd???
print("mask_all_tiled:"+str(np.ndim(np.tile(mask_all[...,0], (nd,1,1)))))
print("nd:"+str(nd))
tst_par = { 'kdata_all'     : kdata_all,
            'csm'           : csm,
            'mask_trn1'     : np.tile(mask_all[...,0], (nd*num_slc,1,1)),
            'mask_trn2'     : np.tile(mask_all[...,1], (nd*num_slc,1,1)),
            'mask_lss1'     : np.tile(mask_all[...,0], (nd*num_slc,1,1)),
            'mask_lss2'     : np.tile(mask_all[...,1], (nd*num_slc,1,1)),
            'b0_map'        : field_all,                                          }

tst_dat =   data_generator(**tst_par)
pred    =   model.predict(tst_dat,verbose=1)
print("Pred Data Shape:")
print(pred.shape)
recon_all = pred[...,0:2]
recon_dif = mf.msos(np.transpose(recon_all,(1,2,0,3)),axis=-1)
#recon_all=recon_all[:,:,13,:]

#recon_dwi = mf.msos(recon_dif,axis=-1)
savemat(path_fig_save+'example_results.mat', {"msEPI": recon_all, 'dif': recon_dif  })

mf.mosaic(np.rot90(np.abs(np.squeeze(recon_all))),1,1,100,[0,0.6],path_fig_save,'Pred')
#mf.mosaic(np.rot90(np.abs(np.squeeze(buda_all))),1,2,100,[0,0.6],path_fig_save,'Buda')

# mf.mosaic(np.rot90(np.abs(np.squeeze(recon_dif))),4,8,101,[0,0.6],'Zero-MIRID')
# mf.mosaic(np.rot90(np.abs(np.squeeze(recon_dwi))),1,1,102,[0,4],'DWI')


mf.mosaic(np.rot90(np.abs(np.squeeze(recon_dif))),4,8,101,[0,0.6],path_fig_save,'Zero-MIRID')

mf.mosaic(np.rot90(np.abs(np.squeeze(1e4*pred[0,:,:,2:3]))),1,1,111,[0,1],path_fig_save,'loss_l1_kspace')
mf.mosaic(np.rot90(np.abs(np.squeeze(2e0*pred[0,:,:,3:4]))),1,1,113,[0,1],path_fig_save,'loss_l2_kspace')
