import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import scipy
import time

from skimage.io import imread
#from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale

import mrcfile as mrc 


#image = shepp_logan_phantom()
#image = rescale(image, scale=0.4, mode='reflect', multichannel=False)

def sino(image):
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

    ax1.set_title("Original")
    ax1.imshow(image, cmap=plt.cm.Greys_r)

    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    ax2.set_title("Radon transform\n(Sinogram)")
    ax2.set_xlabel("Projection angle (deg)")
    ax2.set_ylabel("Projection position (pixels)")
    ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
            extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

    fig1.tight_layout()
    return sinogram, theta, fig1
    

def recon(sinogram, theta, image):
    from skimage.transform import iradon

    reconstruction_fbp = iradon(sinogram, theta=theta, circle=True)
    error = reconstruction_fbp - image
    print(f"FBP rms reconstruction error: {np.sqrt(np.mean(error**2)):.3g}")

    imkwargs = dict(vmin=-0.2, vmax=0.2)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                                sharex=True, sharey=True)
    ax1.set_title("Reconstruction\nFiltered back projection")
    ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    ax2.set_title("Reconstruction error\nFiltered back projection")
    ax2.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
    fig2.tight_layout()
    return fig2

def flatten_sin(sin):
    flat_sin = np.array([np.sum(sin[x,:]) for x in range(sin.shape[0])])
    flat_sin = np.reshape(flat_sin,(flat_sin.shape[0],1))
    return flat_sin

def compare(sin1, sin2):
    mat = np.array([])
    for x in sin1.T:
        for y in sin2.T:
            # plt.plot(x)
            # plt.plot(y)
            # correl = scipy.signal.correlate(x,y, mode='valid')
            # correl = np.correlate(x,y)
            correl = np.cbrt(((x-y)**2).mean(axis=0))
            mat = np.append(mat,correl)
            # plt.show()
    mat = np.reshape(mat, (int(np.sqrt(mat.shape[0])),int(np.sqrt(mat.shape[0]))))
    return mat

def compare3(sin1, sin2, sin3):
    matxy = np.array([])
    for x in sin1.T:
        for y in sin2.T:
            correl = np.cbrt(((x-y)**2).mean(axis=0))
            matxy = np.append(matxy,correl)
    matxy = np.reshape(matxy, (int(np.sqrt(matxy.shape[0])),int(np.sqrt(matxy.shape[0]))))

    matxz = np.array([])
    for x in sin1.T:
        for z in sin3.T:
            correl = np.cbrt(((x-z)**2).mean(axis=0))
            matxz = np.append(matxz,correl)
    matxz = np.reshape(matxz, (int(np.sqrt(matxz.shape[0])),int(np.sqrt(matxz.shape[0]))))

    matyz = np.array([])
    for y in sin2.T:
        for z in sin3.T:
            correl = np.cbrt(((y-z)**2).mean(axis=0))
            matyz = np.append(matyz,correl)
    matyz = np.reshape(matyz, (int(np.sqrt(matyz.shape[0])),int(np.sqrt(matyz.shape[0]))))

    matyy = np.array([])
    for y in sin2.T:
        for z in sin2.T:
            correl = np.cbrt(((y-y)**2).mean(axis=0))
            matyy = np.append(matyy,correl)
    matyy = np.reshape(matyy, (int(np.sqrt(matyy.shape[0])),int(np.sqrt(matyy.shape[0]))))
    
    fig, axes = plt.subplots(3, 3)
    axes[0, 0].imshow(matxz)
    axes[0, 1].imshow(sin1.T)
    axes[0, 2].imshow(matxy)
    axes[1, 0].imshow(sin3)
    axes[1, 1].imshow(matyy)
    axes[1, 2].imshow(sin2)
    axes[2, 0].imshow(matyz)
    axes[2, 1].imshow(sin2.T)
    axes[2, 2].imshow(matyy)
    axes[0,0], loc = find_min(matxz, axes[0,0])
    axes[0,2], loc = find_min(matxy, axes[0,2])
    axes[2,0], loc = find_min(matyz, axes[2,0])
    return fig

def compare3d(sin1, sin2, sin3):
    mat = np.array([])
    x_ = 0
    for x in sin1.T[:40]:
        x_start = time.time()
        mat_y = np.array([])
        x_ += 1
        for y in sin2.T[:40]:
            mat_z = np.array([])
            for z in sin3.T[:40]:
                # plt.plot(x)
                # plt.plot(y)
                # correl = scipy.signal.correlate(x,y, mode='valid')
                # correl = np.correlate(x,y)
                correl1 = np.cbrt(((x-y)**2).mean(axis=0))
                correl2 = np.cbrt(((x-z)**2).mean(axis=0))
                correl3 = np.cbrt(((z-y)**2).mean(axis=0))
                correl = (correl1 + correl2 + correl3)/3
                mat_z = np.append(mat_z,correl)
            mat_y = np.append(mat_y,mat_z)
        mat = np.append(mat,mat_y)
        x_end = time.time()
        print(f'{x_}: {x_end-x_start}')
    mat = np.reshape(mat, (int(np.cbrt(mat.shape[0])),int(np.cbrt(mat.shape[0])),int(np.cbrt(mat.shape[0]))))
    mat3d = mat
    mat3d = mat3d / mat3d.max()
    fig8 = plt.figure()

    x = np.arange(mat3d.shape[0])[:, None, None]
    y = np.arange(mat3d.shape[1])[None, :, None]
    z = np.arange(mat3d.shape[2])[None, None, :]
    x, y, z = np.broadcast_arrays(x, y, z)

    # Turn the volumetric data into an RGB array that's
    # just grayscale.  There might be better ways to make
    # ax.scatter happy.
    c = np.tile(mat3d.ravel()[:, None], [1, 4])



    ax = fig8.gca(projection='3d')
    ax.scatter(x.ravel(),
            y.ravel(),
            z.ravel(),
            c=c)
    return mat
    
def find_min(mat, plt):
    x, y = mat.shape
    min_arg = np.argmin(mat)
    row = min_arg // x
    col = min_arg % x
    loc = (row,col)
    plt.scatter(loc[1], loc[0])
    try:
        plt.set_title(f'{loc} : {np.min(mat):.2f}')
    except: pass
    print(f'minimum loc: {loc}')
    print(f'minimum: {np.min(mat)}')
    print(f'minimum: {np.min(np.array(mat)[mat != np.min(mat)])}')
    return plt, loc

with mrc.open('projections/0_0.mrcs') as f:
    img1 = f.data
sin1, th, fig1 = sino(img1)
# fig2 = recon(sin1, th, img1)

with mrc.open('projections/0_4.mrcs') as f:
    img2 = f.data
sin2, th, fig2 = sino(img2)
# fig4 = recon(sin2, th, img2)

with mrc.open('projections/0_3.mrcs') as f:
    img3 = f.data
sin3, th, fig3 = sino(img3)
# fig2 = recon(sin3, th, img3)

flat_sin1 = flatten_sin(sin1)
flat_sin2 = flatten_sin(sin2)
flat_sin3 = flatten_sin(sin3)

mat = compare(sin1, sin2)

#mat3d = compare3d(sin1, sin2, sin3)
fig3 = compare3(sin1, sin2, sin3)


#mat = scipy.signal.correlate2d(sin1,sin2, mode='same')
#mat = scipy.signal.correlate2d(img1,img2, mode='same')

plt.figure(6)
plt.imshow(mat)
plt, loc = find_min(mat, plt)
plt.figure(7)
plt.plot(sin1.T[loc[0]])
plt.plot(sin2.T[loc[1]])
plt.show()
