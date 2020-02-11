import mrcfile as mrc
from skimage.transform import radon
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import random
import cv2
import math
import time
import os
import argparse
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates

parser = argparse.ArgumentParser()
parser.add_argument('-s','--start', required=True)
parser.add_argument('-e','--end', required=True)
args = parser.parse_args()
print(f'{args.start} - {args.end}')



def save_sin(sin, name, angle):
    img_data = np.float32(sin)
    try: 
        os.mkdir('tmp2')
    except OSError:
        pass
    try: 
        os.mkdir(f'tmp2/{angle}')
    except OSError:
        pass
    with mrc.new(f'tmp2/{angle}/{name}.mrc', overwrite=True) as f:
        f.set_data(img_data)

def sino(image):

    theta = np.linspace(0., 360., 2*max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    return sinogram

def crop(vol, dm):
    ds = vol.shape[0]
    diff = math.floor((ds - dm)/2)
    if diff > 0:
        vol = vol[diff:dm+diff,:,:]
    ds = vol.shape[1]
    diff = math.floor((ds - dm)/2)
    if diff > 0:
        vol = vol[:,diff:dm+diff,:]
    ds = vol.shape[2]
    diff = math.floor((ds - dm)/2)
    if diff > 0:
        vol = vol[:,:,diff:dm+diff]
    return vol
    

with mrc.open('6gdg.mrc') as f:
    data = np.array(f.data)

dim = (1,1,1)
resized = scipy.ndimage.zoom(data, dim)
min_dim = math.ceil(resized.shape[0])

rotated_vol = resized

for i in range(int(args.start), int(args.end)):
    if i % 10 == 0:
        rotated_vol = resized

    rand_angx = random.randint(0,360)
    rand_angy = random.randint(0,360)
    rand_angz = random.randint(0,360)

    rotated_vol = scipy.ndimage.interpolation.rotate(rotated_vol, rand_angx, axes=(1,2), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)
    rotated_vol = scipy.ndimage.interpolation.rotate(rotated_vol, rand_angy, axes=(0,2), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)
    rotated_vol = scipy.ndimage.interpolation.rotate(rotated_vol, rand_angz, axes=(0,1), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)

    projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])

    projection_1 = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])


    # plt.figure(1)
    # plt.subplot(1,2,1)
    # plt.imshow(projection_1)



    # 1 ------------------------------------------------------------------------------------
    rot_vol = rotated_vol

    rotated_vol = scipy.ndimage.interpolation.rotate(rot_vol, 25, axes=(0,1), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)
    rotated_vol = scipy.ndimage.interpolation.rotate(rot_vol, 45, axes=(1,2), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)

    projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])
    projection_2 = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])
    
    # plt.figure(2)
    # plt.subplot(1,2,1)
    # plt.imshow(projection_2)
    # plt.show()
    

    sin1 = sino(projection_1)
    sin2 = sino(projection_2)

    save_sin(sin1, f'sino_{i}', 0)
    save_sin(sin2, f'sino_{i}', 45)
