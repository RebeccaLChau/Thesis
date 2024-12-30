"""
call: run vae_ml_data_prep.py
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import random

import pandas as pd

import numpy as numpy
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from rembg import remove
from PIL import Image
import seaborn as sns

import tensorflow as tf

import pickle

import sys
import datetime

from torchvision import datasets, transforms

from astropy.io import fits
from skimage import io, img_as_float

import scipy.stats
from inspect import currentframe, getframeinfo


#pl.seed_everything()

# torch.manual_seed(1)
# numpy.random.seed(1)

## set seed to allow reproducible results
SEED = 1
numpy.random.seed(SEED)
os.environ['PYTHONHASSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)




## track time
t0 = str(datetime.datetime.today())
print('Read data: '+t0)
#pl.seed_everything()


# DIRECTORIES
img_path = '/data/leuven/369/vsc36918/Full/Fullcropped'
model_path = '/data/leuven/369/vsc36918/LD64E1000' #check if correct
#-----------------------------------------------------------------------------#
#    DATA LOADER                                                              #
#-----------------------------------------------------------------------------#

batchsz    = 64 #match main.py
trainpctg  = 0.5 #data used for training


dataset = datasets.ImageFolder(
    img_path,
    transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        transforms.Normalize(
              mean=[0., 0., 0.],
              std=[1., 1.0, 1.0])
]))

all_img = len(dataset)


trainset, testset = random_split(dataset, [int(trainpctg*len(dataset)),
                                           len(dataset)-int(trainpctg*len(dataset))])
print('Len: ', len(dataset), len(trainset), len(testset))

smallsz     = 50
subidx      = numpy.random.choice(len(trainset), smallsz, replace=False)
trainsubset = torch.utils.data.Subset(trainset, subidx)
subidx      = numpy.random.choice(len(testset),  smallsz, replace=False)
testsubset  = torch.utils.data.Subset(testset, subidx)

large = True

if large:
    dataloader = DataLoader(trainset, batch_size=batchsz, shuffle=True, drop_last=True, num_workers=4)
    testloader = DataLoader(testset , batch_size=batchsz, shuffle=False, drop_last=True, num_workers=4)
    setloader  = DataLoader(dataset , batch_size=1, shuffle=False, drop_last=True, num_workers=4)
else:
    dataloader = DataLoader(trainsubset, batch_size=1, shuffle=True, num_workers=4)
    testloader = DataLoader(testsubset , batch_size=1, shuffle=False, num_workers=4)


# sys.exit()

#----------------------------------------------------------------------------- #
# TRAINED MODELS                                                               #
#----------------------------------------------------------------------------- #
print('--------------------------------------------------------------')
import vae_model_ext as vaemodel #check which model

## large flare dataset
FNAME = 'vae_flares_Br_beta25_lt006_bz200_splt50_ep1000_lr5e-4' 
# FNAME = 'vae_flares_Br_beta25_lt006_bz300_splt50_ep1000_lr5e-4_ext'
# FNAME = 'vae_flares_Br_beta25_lt018_bz300_splt50_ep1000_lr5e-4_ext'
# FNAME = 'vae_flares_Br_beta25_lt036_bz100_splt50_ep1000_lr5e-4_ext'

# FNAME = 'VSC/vae_flares_Br_beta25_lt018_bz100_splt50_ep1000_lr5e-4'
PATH = model_path+FNAME+'.pth'
print(FNAME)
print('--------------------------------------------------------------')


# define model
model = torch.load(PATH)
model.eval()

#----------------------------------------------------------------------------- #
# Loop over all the images, get latent space and run clustering                #
#----------------------------------------------------------------------------- #

## definitions
x,_ = next(iter(setloader))
t = model(x)
z, mu, lvar = model.encode(x)
print(z.detach().numpy())

## litlle test:
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize = (8,5),
#                          constrained_layout=True)
# axes[0].imshow(x[0].permute((1,2,0))[:,:,0].detach().numpy(),cmap='gray')
# axes[1].imshow(t[0][0].permute((1,2,0))[:,:,0].detach().numpy(),cmap='gray')
# plt.show()

# frameinfo = getframeinfo(currentframe())
# ln = frameinfo.lineno
# print('Stoped at line: ', ln+3)
# sys.exit()

## definitions
latdim = len(z[0])
sNum = len(setloader)

vae_lvec = numpy.empty([sNum, latdim], dtype = float)
vae_mu = numpy.empty([sNum, latdim], dtype = float)
vae_sd = numpy.empty([sNum, latdim], dtype = float)

# dist = numpy.empty([sNum, latdim, 80], dtype = float)

# org_total = numpy.empty([sNum], dtype = float)
# gen_total = numpy.empty([sNum], dtype = float)

# vae_dist = np.empty([sNum, 1, latdim], dtype = float)

# df_vae_lvec = pd.DataFrame(columns=['L1', 'L2', 'L3',
#                                     'L4', 'L5', 'L6'])

df_vae_info = pd.DataFrame(columns=['HARPNUM', 'NOAA', 'TREC',
                                    'DOY', 'MOD'])

#sys.exit()

for i, (x,_) in enumerate(setloader):
    t = model(x)
    z, mu, lvar = model.encode(x)

    ind = i*1
    img_name = [dataset.samples[j][0] for j in range(ind, ind + 1)]
    str_name = img_name[0]
    HARPNUM = str_name[54:58]
    NOAANUM = str_name[59:64]

    date = str_name[65:73]
    time = str_name[74:80]

    hh = time[0:2]
    mm = time[2:4]
    mod = (int(hh)*60+int(mm))

    dt = (date[0:4]+'-'+date[4:6]+'-'+date[6:8]
          +' '+hh+':'+mm+':'+time[4:6])
    doy = datetime.datetime.strptime(dt,"%Y-%m-%d %H:%M:%S"
                                     ).timetuple().tm_yday

    # print(HARPNUM, NOAANUM)

    new_row = {'HARPNUM': HARPNUM, 'NOAA': NOAANUM, 'TREC':dt,
               'DOY': doy, 'MOD':mod}
    new_row_df = pd.DataFrame([new_row])
    df_vae_info = pd.concat([df_vae_info, new_row_df], ignore_index=True)



    # org_total[i] = (numpy.sum(x.detach().numpy()))/3.
    # gen_total[i] = (numpy.sum(t[0].detach().numpy()))/3.

    z0 = (z.detach().numpy())[0]
    m0 = mu.detach().numpy()
    sd0  = torch.exp(lvar*0.5).detach().numpy()

    # ## I am trying something
    # m = numpy.array([[j for j in numpy.arange(-4,4,0.1)] for i in range(len(mu))])
    # dist0 = numpy.array([[scipy.stats.norm.pdf(m[k][:], m0[k][j], sd0[k][j])
    #                   for k in range(len(mu))] for j in range(latdim)])
    
    # new_row = {'L1':z0[0] ,'L2':z0[1], 'L3':z0[2],
    #            'L4':z0[3], 'L5':z0[4], 'L6':z0[5]}
    # new_row_df = pd.DataFrame([new_row])

    # df_vae_lvec = pd.concat([df_vae_lvec, new_row_df], ignore_index=True)
      


    #dist[:,0].shape
    # dist[i] = dist0[:,0]
    # # print(z0)
    # # print(' ')
    # vae_lvec[i] = z0[0]
    # # print(vae_lvec[i])
    # vae_mu[i] = m0[0]
    # vae_sd[i] = sd0[0]
    

df_vae_info.to_csv('/work1/dineva/SHARP/sav_files/VAEinfo_withNAN.csv')
# df_vae_lvec.to_csv('/work1/dineva/SHARP/sav_files/VAEdata_withNAN.csv')

sys.exit()





## SAVE AS (3 types of) FITS FILES
path = '/data/leuven/369/vsc36918/Fits_LD64E1000'


## If path doesn't exist make one.
if os.path.exists(path):
    hdu = fits.PrimaryHDU(data=vae_lvec)
    hdu.writeto(path+'LVec_'+FNAME+'.fits',overwrite=True)

    hdu = fits.PrimaryHDU(data=vae_mu)
    hdu.writeto(path+'Mu_'+FNAME+'.fits',overwrite=True)

  #  hdu = fits.PrimaryHDU(data=vae_sd)
  # hdu.writeto(path+'Std_'+FNAME+'.fits',overwrite=True)

    hdu = fits.PrimaryHDU(data=dist)
    hdu.writeto(path+'Dist_'+FNAME+'.fits',overwrite=True)
else:
    os.makedirs(path)

    hdu = fits.PrimaryHDU(data=vae_lvec)
    hdu.writeto(path+'/LVec_'+FNAME+'.fits',overwrite=True)

    hdu = fits.PrimaryHDU(data=vae_mu)
    hdu.writeto(path+'/Mu_'+FNAME+'.fits',overwrite=True)

    hdu = fits.PrimaryHDU(data=vae_sd)
    hdu.writeto(path+'/Std_'+FNAME+'.fits',overwrite=True)

    hdu = fits.PrimaryHDU(data=dist)
    hdu.writeto(path+'/Dist_'+FNAME+'.fits',overwrite=True)

print(" ")
print("PATH: ", path)
print("FINISHED AND SAVED")

sys.exit()













