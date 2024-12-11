"""
#call: run vae_multi_infer.py

## BOTTLENECK VARIABLES:
## mu = Mean
## lv = latent variance
## z  = latent variable


"""
import os

import numpy as numpy
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from rembg import remove
from PIL import Image

import pickle

import sys
import datetime

from torchvision import datasets, transforms

from astropy.io import fits
from skimage import io, img_as_float

import scipy.stats


pl.seed_everything(42)

## track time
t0 = str(datetime.datetime.today())
print('Read data: '+t0)
#pl.seed_everything()


## DIRECTORIES
img_path = 'F:\Full test cropped' 
model_path = r'C:\Users\reb\Documents\Models'
#-----------------------------------------------------------------------------#
#    DATA LOADER                                                              #
#-----------------------------------------------------------------------------#

batchsz    = 180  #changed to match below
trainpctg  = 0.5 #data used for training

dataset = datasets.ImageFolder(
    img_path,
    transforms.Compose([
        transforms.Resize((500, 500)), #changed to match main
        transforms.ToTensor()
]))

# dataset = datasets.ImageFolder(
#     img_path,
#     transforms.Compose([
#         transforms.Resize((500, 500)), #changed to match main
#         transforms.ToTensor()
# ]))



# dataset = datasets.ImageFolder(
#     img_path,
#     transforms.Compose([
#         transforms.Resize((500, 500)), #changed to match main
#         transforms.ToTensor(),
#         transforms.Normalize(
#               mean=[0., 0., 0.],
#               std=[1., 1.0, 1.0])
# ]))

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
    setloader  = DataLoader(dataset , batch_size=all_img, shuffle=False, drop_last=True, num_workers=4)
    # setloader  = DataLoader(dataset , batch_size=1, shuffle=False, drop_last=True, num_workers=4)
else:
    dataloader = DataLoader(trainsubset, batch_size=1, shuffle=True, num_workers=4)
    testloader = DataLoader(testsubset , batch_size=1, shuffle=False, num_workers=4) #changed all num_workers to 4 to match 

# print(dataset)
# print(dataloader)
# print("Min/Max:", dataset[0][0].min(), dataset[0][0].max())


# x,_ = next(iter(dataloader))
# img = x.detach().numpy()
# print(numpy.sum(img[0,0,:,:]))
# print(img.max(), img.min(), img.mean())
#
# plt.imshow(img[0,0,:,:])
# plt.show()
# plt.imshow(img[0,1,:,:])
# plt.show()
# plt.imshow(img[0,2,:,:])
# plt.show()



# sys.exit()

#----------------------------------------------------------------------------- #
# TRAINED MODELS                                                               #
#----------------------------------------------------------------------------- #
print('--------------------------------------------------------------')
import Model1 as vaemodel #check which one - model1 - architecture

## large flare dataset
# FNAME = 'vae_flares_Br_beta25_lt006_bz200_splt50_ep1000_lr5e-4'  #change images
# FNAME = 'vae_flares_Br_beta25_lt006_bz300_splt50_ep1000_lr5e-4_ext'
# FNAME = 'vae_flares_Br_beta25_lt018_bz300_splt50_ep1000_lr5e-4_ext'
FNAME = 'LD64E1000vae_model.pth' #this is for model

# FNAME = 'VSC/vae_flares_Br_beta25_lt018_bz100_splt50_ep1000_lr5e-4'

PATH = model_path+FNAME+'.pth'
print(FNAME)
print('--------------------------------------------------------------')

# define model
model = torch.load(PATH)
model.eval()

# definitions
cols  = 6
shift = 5
comp  = 0 ## (Br=0, Bp=1, Bt=2)

x,_ = next(iter(testloader))
t = model(x)


# sys.exit()
# ---------------------------------------------------------------------------- #
# CHECK RECONSTRUCTION                                                         #
# ---------------------------------------------------------------------------- #

# fig, axes = plt.subplots(nrows=2, ncols=cols, figsize = (16,5),
#                          constrained_layout=True,
#                          gridspec_kw={'height_ratios': [1, 1]})
# for i in range(cols):
#     axes[0][i].imshow(x[i].permute((1,2,0))[:,:,comp].detach().numpy(),
#                       cmap='gray')
#     axes[0][1].set_title('Original')

# for i in range(cols):
#     axes[1][i].imshow(t[0][i].permute((1,2,0))[:,:,comp].detach().numpy(),
#                       cmap='gray')
#     axes[1][1].set_title('Generated')
# plt.show(block=False)
# plt.savefig('gen_figs/' + FNAME + '.pdf', bbox_inches='tight')
# plt.close()


# sys.exit()


# ============================================================================ #
# FIGURE FOR PAPER                                                             #
# ============================================================================ #
csfont = {'fontname':'Times New Roman'}

img1 = x[140].permute((1,2,0))[:,:,comp].detach().numpy() #change to match batch size
img2 = x[150].permute((1,2,0))[:,:,comp].detach().numpy()
img3 = x[160].permute((1,2,0))[:,:,comp].detach().numpy()

img4 = x[70].permute((1,2,0))[:,:,comp].detach().numpy()
img5 = x[80].permute((1,2,0))[:,:,comp].detach().numpy()
img6 = x[90].permute((1,2,0))[:,:,comp].detach().numpy()



rec_img1 = t[0][140].permute((1,2,0))[:,:,comp].detach().numpy()
rec_img2 = t[0][150].permute((1,2,0))[:,:,comp].detach().numpy()
rec_img3 = t[0][160].permute((1,2,0))[:,:,comp].detach().numpy()

rec_img4 = t[0][70].permute((1,2,0))[:,:,comp].detach().numpy()
rec_img5 = t[0][80].permute((1,2,0))[:,:,comp].detach().numpy()
rec_img6 = t[0][90].permute((1,2,0))[:,:,comp].detach().numpy()

nx = img1[0].size
ny = img1[1].size

dx = 4
dy = 4

sx = 2.*nx + 3.*dx
sy = 2.*ny + dy

wx = 1800./2.
wy = float(wx) / sx * sy

scale = 28.34
width = wx / scale
height = wy / scale

# fig, axes = plt.subplots(2, 6, figsize=(width, height/2.5),
#                          layout="constrained")

# fig, axes = plt.subplots(2, 6, figsize=(30, 6),
#                          layout="constrained")

fig, axes = plt.subplots(2, 3, figsize=(15, 6),
                         layout="constrained")
### --------- ORIGINAL --------------------------------------------------------
axes[0][0].imshow(img1, cmap='gray')
# axes[0][0].set_yticks(numpy.arange(0, 125, 25.))
axes[0][0].set_ylabel('Solar Y')
axes[0][0].set_xlim(0, 250)
axes[0][0].set_ylim(0, 125)

axes[0][1].imshow(img2,cmap='gray')
# axes[0][1].set_title('Original')
axes[0][1].set_xlim(0, 250)
axes[0][1].set_ylim(0, 125)

axes[0][2].imshow(img3,cmap='gray')
axes[0][2].set_xlim(0, 250)
axes[0][2].set_ylim(0, 125)

# axes[0][3].imshow(img4, cmap='gray')
# axes[0][3].set_ylabel('Solar Y')
# axes[0][3].set_xlim(0, 250)
# axes[0][3].set_ylim(0, 125)

# axes[0][4].imshow(img5,cmap='gray')
# axes[0][4].set_xlim(0, 250)
# axes[0][4].set_ylim(0, 125)

# axes[0][5].imshow(img6,cmap='gray')
# axes[0][5].set_xlim(0, 250)
# axes[0][5].set_ylim(0, 125)


### --------- RECOSTRUCTED ----------------------------------------------------
axes[1][0].imshow(rec_img1,cmap='gray')
axes[1][0].set_ylabel('Solar Y')
axes[1][0].set_xlabel('Solar X')
axes[1][0].set_xlim(0, 250)
axes[1][0].set_ylim(0, 125)

axes[1][1].imshow(rec_img2,cmap='gray')
# axes[1][1].set_title('Generated')
axes[1][1].set_xlabel('Solar X')
axes[1][1].set_xlim(0, 250)
axes[1][1].set_ylim(0, 125)

axes[1][2].imshow(rec_img3,cmap='gray')
axes[1][2].set_xlabel('Solar X')
axes[1][2].set_xlim(0, 250)
axes[1][2].set_ylim(0, 125)

# axes[1][3].imshow(rec_img4,cmap='gray')
# # axes[1][1].set_title('Generated')
# axes[1][3].set_xlabel('Solar X')
# axes[1][3].set_xlim(0, 250)
# axes[1][3].set_ylim(0, 125)

# axes[1][4].imshow(rec_img5,cmap='gray')
# axes[1][4].set_xlabel('Solar X')
# axes[1][4].set_xlim(0, 250)
# axes[1][4].set_ylim(0, 125)

# axes[1][5].imshow(rec_img6,cmap='gray')
# axes[1][5].set_xlabel('Solar X')
# axes[1][5].set_xlim(0, 250)
# axes[1][5].set_ylim(0, 125)

# fig.tight_layout(pad=1.0)
plt.rcParams.update({'font.size': 18})
# fig.tight_layout(pad=-0.2)
plt.subplots_adjust(wspace=0, hspace=0.1)
# plt.show(block=False)
plt.savefig('gen_figs/vae_fig01_'+FNAME[4:53]+'.png', bbox_inches='tight')
# plt.savefig('gen_figs/vae_fig01_'+FNAME[10:38]+'.pdf', bbox_inches='tight')

# convert PDF to (okay-ish) JPEG
#from pdf2image import convert_from_path
#images = convert_from_path('figs/model1000_bz20_beta15_lt100.pdf')
#images[0].save('figs/model1000_bz20_beta15_lt100.jpg', 'JPEG')

# sys.exit()

# ---------------------------------------------------------------------------- #
# DIFERENCE IMAGES                                                             #
# ---------------------------------------------------------------------------- #
ny,nx = x[1,0].detach().numpy().shape
ration  = nx/ny

img = x[3].permute((1,2,0))[:,:,comp].detach().numpy()
rec = t[0][3].permute((1,2,0))[:,:,comp].detach().numpy()
dif = img - rec


nbins = 20
counts, bins = numpy.histogram(img)
counts1, bins1 = numpy.histogram(rec)
counts2, bins2 = numpy.histogram(dif)
plt.hist(bins[:-1], bins=nbins, weights=counts, color='red')
plt.hist(bins1[:-1], bins=nbins, weights=counts1, color='blue')
plt.hist(bins2[:-1], bins=nbins, weights=counts2, color='green')

fig, ax = plt.subplots(1,3, figsize=(8*2, 2.56*2), constrained_layout=True)

ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original')

ax[1].imshow(rec, cmap='gray')
ax[1].set_title('Reconstructed')

ax[2].imshow(dif, cmap='gray')
ax[2].set_title('Difference')

fig.tight_layout(pad=0.5)
# plt.show(block=False)
plt.savefig('gen_figs/vae_fig03_'+FNAME[4:53]+'.png', bbox_inches='tight')
# plt.show(block=False)

# sys.exit()
# ---------------------------------------------------------------------------- #
# COUNTOR IMAGES                                                               #
# ---------------------------------------------------------------------------- #


# To Do: Take an array with the original values make a contour and apply it to the images bellow

ny,nx = x[1,0].detach().numpy().shape
x_data = numpy.linspace(0,256,nx)
y_data = numpy.linspace(0,128,ny)
lvl = 0.35 # dark patches
img = x[3].permute((1,2,0))[:,:,comp].detach().numpy()
t_img = t[0][3].permute((1,2,0))[:,:,comp].detach().numpy()

norm = 2*(img - numpy.min(img)) / (numpy.max(img) - numpy.min(img))-1
t_norm = 2*(t_img - numpy.min(t_img)) / (numpy.max(t_img) - numpy.min(t_img))-1

lv1 = -0.3
lv2 = 0.25

# fig, ax = plt.subplots(2, 1, figsize = (6,8),
fig, ax = plt.subplots(2, 1, figsize = (width/2., height/2.),
                         constrained_layout=True,
                         gridspec_kw={'height_ratios': [1, 1]})

ax[0].imshow(x[3].permute((1,2,0))[:,:,comp].detach().numpy(), cmap='gray')
ax[0].contour(x_data,y_data,norm,[lv1],colors='r',linewidths=0.65)
ax[0].contour(x_data,y_data,norm,[lv2],colors='b',linewidths=0.65)
ax[0].set_ylabel('Solar Y')
ax[0].set_xlim(0, 250)
ax[0].set_ylim(0, 125)


ax[1].imshow(t[0][3].permute((1,2,0))[:,:,comp].detach().numpy(), cmap='gray')
ax[1].contour(x_data,y_data,norm,[lv1],colors='r',linewidths=0.65)
ax[1].contour(x_data,y_data,norm,[lv2],colors='b',linewidths=0.65)
ax[1].set_ylabel('Solar Y')
ax[1].set_xlabel('Solar X')
ax[1].set_xlim(0, 250)
ax[1].set_ylim(0, 125)

plt.rcParams.update({'font.size': 45})
# plt.show(block=False)
plt.savefig('gen_figs/vae_fig02_'+FNAME[4:53]+'.png', bbox_inches='tight')
plt.close()













