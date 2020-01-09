import mrcfile as mrc
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import random
import cv2
import math


def crop(vol, dm):
    ds = vol.shape[0]
    diff = math.floor((ds - dm)/2)
    if diff > 0:
        vol = vol[diff:ds-diff,:,:]
    ds = vol.shape[1]
    diff = math.floor((ds - dm)/2)
    if diff > 0:
        vol = vol[:,diff:ds-diff,:]
    ds = vol.shape[2]
    diff = math.floor((ds - dm)/2)
    if diff > 0:
        vol = vol[:,:,diff:ds-diff]
    return vol

with mrc.open('postprocess.mrc') as f:
    data = np.array(f.data)

dim = (0.5,0.5,0.5)
resized = scipy.ndimage.zoom(data, dim)
print(resized.shape)
min_dim = math.ceil(0.7*resized.shape[0]*np.sqrt(3))

'''
repro1 = np.array([sum(data[:,x,y]) for x in range(256) for y in range(256)])
repro2 = np.array([sum(data[x,:,y]) for x in range(256) for y in range(256)])
repro3 = np.array([sum(data[x,y,:]) for x in range(256) for y in range(256)])
data = np.array(data)
repro1 = np.array(repro1)
repro1 = repro1.reshape(256,256)
plt.imshow(repro1)
plt.show()
repro2 = np.array(repro2)
repro2 = repro2.reshape(256,256)
plt.imshow(repro2)
plt.show()
repro3 = np.array(repro3)
repro3 = repro3.reshape(256,256)
plt.imshow(repro3)
plt.show()
'''

rand_angx = random.randint(0,360)
rand_angy = random.randint(0,360)
rand_angz = random.randint(0,360)

rotated_vol = scipy.ndimage.interpolation.rotate(resized, rand_angx, axes=(1,2), reshape=True) # (0,1) (1,2) (0,2)
rotated_vol = crop(rotated_vol, min_dim)
rotated_vol = scipy.ndimage.interpolation.rotate(rotated_vol, rand_angy, axes=(0,2), reshape=True) # (0,1) (1,2) (0,2)
rotated_vol = crop(rotated_vol, min_dim)
rotated_vol = scipy.ndimage.interpolation.rotate(rotated_vol, rand_angz, axes=(0,1), reshape=True) # (0,1) (1,2) (0,2)
rotated_vol = crop(rotated_vol, min_dim)

projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])
projection = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(projection)
rotated_vol = scipy.ndimage.interpolation.rotate(rotated_vol, 40, axes=(1,2), reshape=True) # (0,1) (1,2) (0,2)
rotated_vol = crop(rotated_vol, min_dim)
projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])
projection = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])
plt.subplot(1,2,2)
plt.imshow(projection)
plt.show()
    

