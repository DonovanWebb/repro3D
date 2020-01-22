import mrcfile as mrc
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import random
import cv2
import math
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s','--start', required=True)
parser.add_argument('-e','--end', required=True)
args = parser.parse_args()
print(f'{args.start} - {args.end}')

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
    print(vol.shape)
    return vol
    
def save_part(part, name):
    img_data = np.float32(part)
    try: 
        os.mkdir('tmp')
    except OSError:
        pass
    with mrc.new(f'tmp/{name}.mrcs', overwrite=True) as f:
        f.set_data(img_data)



start = time.time()

with mrc.open('6gdg.mrc') as f:
    data = np.array(f.data)

dim = (1,1,1)
resized = scipy.ndimage.zoom(data, dim)
print(resized.shape)
#min_dim = math.ceil(0.55*resized.shape[0]*np.sqrt(3))
min_dim = math.ceil(resized.shape[0])
#min_dim = math.ceil(128)

rotated_vol = resized

for i in range(int(args.start), int(args.end)):
    if i % 10 == 0:
        rotated_vol = resized

    start_it = time.time()
    rand_angx = random.randint(0,360)
    rand_angy = random.randint(0,360)
    rand_angz = random.randint(0,360)

    if random.randint(0,3) == 0:
        rotated_vol = scipy.ndimage.interpolation.rotate(rotated_vol, rand_angx, axes=(1,2), reshape=True) # (0,1) (1,2) (0,2)
        rotated_vol = crop(rotated_vol, min_dim)
    if random.randint(0,3) == 0:
        rotated_vol = scipy.ndimage.interpolation.rotate(rotated_vol, rand_angy, axes=(0,2), reshape=True) # (0,1) (1,2) (0,2)
        rotated_vol = crop(rotated_vol, min_dim)
    if random.randint(0,3) == 0:
        rotated_vol = scipy.ndimage.interpolation.rotate(rotated_vol, rand_angz, axes=(0,1), reshape=True) # (0,1) (1,2) (0,2)
        rotated_vol = crop(rotated_vol, min_dim)

    projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])

    projection = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])

    save_part(projection,f'{i}_0')

    # plt.figure(1)
    # plt.subplot(1,2,1)
    # plt.imshow(projection)

    rot_vol = rotated_vol
    rotated_vol = scipy.ndimage.interpolation.rotate(rot_vol, 5, axes=(0,1), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)

    projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])

    projection = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])

    save_part(projection,f'{i}_1')

    rotated_vol = scipy.ndimage.interpolation.rotate(rot_vol, 10, axes=(0,1), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)

    projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])

    projection = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])

    save_part(projection,f'{i}_2')

    rotated_vol = scipy.ndimage.interpolation.rotate(rot_vol, 30, axes=(0,1), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)

    projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])

    projection = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])

    save_part(projection,f'{i}_3')

    rotated_vol = scipy.ndimage.interpolation.rotate(rot_vol, 50, axes=(0,1), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)

    projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])

    projection = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])

    save_part(projection,f'{i}_4')

    # plt.subplot(1,2,2)
    # plt.imshow(projection)
    # plt.show()
    end_it = time.time()
    print(f'Time for {i}: {end_it-start_it}')
    

end = time.time()
print(f'Total time: {end-start}')

