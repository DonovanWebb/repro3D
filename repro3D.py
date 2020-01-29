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

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

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

    v = [1, 0, 0]
    print(v)


    # 1 ------------------------------------------------------------------------------------
    rot_vol = rotated_vol

    rotated_vol = scipy.ndimage.interpolation.rotate(rot_vol, 45, axes=(1,2), reshape=True) # (0,1) (1,2) (0,2)
    axis = [1, 0, 0]
    theta = (2*np.pi/360) * 90
    v_ = np.dot(rotation_matrix(axis, theta), v)
    rotated_vol = crop(rotated_vol, min_dim)

    # rotated_vol = scipy.ndimage.interpolation.rotate(rotated_vol, 20, axes=(1,2), reshape=True) # (0,1) (1,2) (0,2)
    # axis = [1, 0, 0]
    # theta = (2*np.pi/360) * 20
    # v_ = np.dot(rotation_matrix(axis, theta), v_)
    # print(v_)
    # rotated_vol = crop(rotated_vol, min_dim)

    projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])
    projection = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])
    save_part(projection,f'{i}_1')
    


    # 2 ------------------------------------------------------------------------------------
    rotated_vol = scipy.ndimage.interpolation.rotate(rot_vol, 90, axes=(0,1), reshape=True) # (0,1) (1,2) (0,2)
    axis = [0, 1, 0]
    theta = (2*np.pi/360) * 90
    v_ = np.dot(rotation_matrix(axis, theta), v)
    rotated_vol = crop(rotated_vol, min_dim)

    rotated_vol = scipy.ndimage.interpolation.rotate(rotated_vol, 45, axes=(1,2), reshape=True) # (0,1) (1,2) (0,2)
    axis = [0, 1, 0]
    theta = (2*np.pi/360) * 20
    v_ = np.dot(rotation_matrix(axis, theta), v_)
    print(v_)
    rotated_vol = crop(rotated_vol, min_dim)

    projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])
    projection = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])
    save_part(projection,f'{i}_2')


    # 3 opt ------------------------------------------------------------------------------------
    '''
    rotated_vol = scipy.ndimage.interpolation.rotate(rot_vol, 20, axes=(0,1), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)

    projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])

    projection = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])

    save_part(projection,f'{i}_3')
    '''


    # plt.subplot(1,2,2)
    # plt.imshow(projection)
    # plt.show()
    end_it = time.time()
    print(f'Time for {i}: {end_it-start_it}')
    

end = time.time()
print(f'Total time: {end-start}')

