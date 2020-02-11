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
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates

parser = argparse.ArgumentParser()
parser.add_argument('-s','--start', required=True)
parser.add_argument('-e','--end', required=True)
args = parser.parse_args()
print(f'{args.start} - {args.end}')


# Rotates 3D image around image center
# INPUTS
#   array: 3D numpy array
#   orient: list of Euler angles (phi,psi,the)
# OUTPUT
#   arrayR: rotated 3D numpy array
# by E. Moebel, 2020
def rotate_array(array, orient):
    phi = orient[0]
    psi = orient[1]
    the = orient[2]

    # create meshgrid
    dim = array.shape
    ax = np.arange(dim[0])
    ay = np.arange(dim[1])
    az = np.arange(dim[2])
    coords = np.meshgrid(ax, ay, az)

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz = np.vstack([coords[0].reshape(-1) - float(dim[0]) / 2,  # x coordinate, centered
                     coords[1].reshape(-1) - float(dim[1]) / 2,  # y coordinate, centered
                     coords[2].reshape(-1) - float(dim[2]) / 2])  # z coordinate, centered

    # create transformation matrix
    r = R.from_euler('zxz', [phi, psi, the], degrees=True)
    mat = r.as_matrix()

    # apply transformation
    transformed_xyz = np.dot(mat, xyz)

    # extract coordinates
    x = transformed_xyz[0, :] + float(dim[0]) / 2
    y = transformed_xyz[1, :] + float(dim[1]) / 2
    z = transformed_xyz[2, :] + float(dim[2]) / 2

    x = x.reshape((dim[1],dim[0],dim[2]))
    y = y.reshape((dim[1],dim[0],dim[2]))
    z = z.reshape((dim[1],dim[0],dim[2])) # reason for strange ordering: see next line

    # the coordinate system seems to be strange, it has to be ordered like this
    new_xyz = [y, x, z]

    # sample
    arrayR = map_coordinates(array, new_xyz, order=1)
    return arrayR

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



    # 1 ------------------------------------------------------------------------------------
    rot_vol = rotated_vol

    rotated_vol = scipy.ndimage.interpolation.rotate(rot_vol, 45, axes=(1,2), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)

    projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])
    projection = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])
    save_part(projection,f'{i}_1')
    


    # 2 ------------------------------------------------------------------------------------
    rotated_vol = scipy.ndimage.interpolation.rotate(rot_vol, 90, axes=(0,1), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)

    rotated_vol = scipy.ndimage.interpolation.rotate(rotated_vol, 45, axes=(1,2), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)

    projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])
    projection = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])
    save_part(projection,f'{i}_2')

    # 3 ------------------------------------------------------------------------------------
    rotated_vol = scipy.ndimage.interpolation.rotate(rot_vol, 30, axes=(0,1), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)

    rotated_vol = scipy.ndimage.interpolation.rotate(rotated_vol, 45, axes=(1,2), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)

    projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])
    projection = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])
    save_part(projection,f'{i}_3')

    # 4 ------------------------------------------------------------------------------------
    rotated_vol = scipy.ndimage.interpolation.rotate(rot_vol, 70, axes=(0,1), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)

    rotated_vol = scipy.ndimage.interpolation.rotate(rotated_vol, 25, axes=(1,2), reshape=True) # (0,1) (1,2) (0,2)
    rotated_vol = crop(rotated_vol, min_dim)

    projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])
    projection = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])
    save_part(projection,f'{i}_4')

    # opt ------------------------------------------------------------------------------------
    '''
    
    eu_angles = [45,0,0]
    rotated_vol = rotate_array(rot_vol, eu_angles)
    projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])
    projection = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])
    save_part(projection,f'{i}_opt')

    eu_angles = [45,75,0]
    rotated_vol = rotate_array(rot_vol, eu_angles)
    projection_flat = np.array([np.sum(rotated_vol[x,y,:]) for x in range(rotated_vol.shape[0]) for y in range(rotated_vol.shape[1])])
    projection = projection_flat.reshape(rotated_vol.shape[0],rotated_vol.shape[1])
    save_part(projection,f'{i}_opt2')
    '''

    '''
    r = R.from_euler('zyx', [0,45,45], degrees=True)
    vec = (0,0,8)
    r.apply(vec)
    empty_vol = np.zeros(rot_vol.shape)
    # for x in range(rot_vol.shape[0]):
    #     for y in range(rot_vol.shape[1]):
    #         for z in range(rot_vol.shape[2]):
    for x in range(25,126):
        for y in range(25,126):
            for z in range(25,126):
                vec = x-rot_vol.shape[0]/2,y-rot_vol.shape[1]/2,z-rot_vol.shape[2]/2
                rot_vec = r.apply(vec)
                rot_vec = rot_vec+rot_vol.shape[0]/2
                if rot_vec[0] >= 0 or rot_vec[1] >= 0 or rot_vec[2] >= 0:
                    try:
                        empty_vol[int(rot_vec[0]), int(rot_vec[1]), int(rot_vec[2])] = rot_vol[x,y,z]
                    except:
                        print(int(rot_vec[0]), int(rot_vec[1]), int(rot_vec[2]))
                        pass
    rotated_vol = empty_vol
    '''


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

