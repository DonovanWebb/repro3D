import time
start = time.time()
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import math
import scipy
import matplotlib.pyplot as plt
from skimage.transform import radon
import mrcfile as mrc 
end_import = time.time()
print(f'import time = {end_import-start}')

def sino(image):
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

    ax1.set_title("Original")
    ax1.imshow(image, cmap=plt.cm.Greys_r)

    theta = np.linspace(0., 360., 2*max(image.shape), endpoint=False)
    start_sin = time.time()
    sinogram = radon(image, theta=theta, circle=True)
    end_sin = time.time()
    print(f'sinogram time = {end_sin-start_sin}')
    '''
    ax2.set_title("Radon transform\n(Sinogram)")
    ax2.set_xlabel("Projection angle (deg)")
    ax2.set_ylabel("Projection position (pixels)")
    ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
            extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

    fig1.tight_layout()
    '''
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
    matxy = np.zeros(sin1.T.shape[0]**2)
    count = 0
    for x in sin1.T:
        for y in sin2.T:
            correl = np.cbrt(((x-y)**2).mean(axis=0))
            #correl = scipy.signal.correlate(x,y, mode='valid')
            matxy[count] = correl
            count += 1
    matxy = np.reshape(matxy, (int(np.sqrt(matxy.shape[0])),int(np.sqrt(matxy.shape[0]))))

    matxz = np.zeros(sin1.T.shape[0]**2)
    count = 0
    for x in sin1.T:
        for z in sin3.T:
            correl = np.cbrt(((x-z)**2).mean(axis=0))
            #correl = scipy.signal.correlate(x,y, mode='valid')
            matxz[count] = correl
            count += 1
    matxz = np.reshape(matxz, (int(np.sqrt(matxz.shape[0])),int(np.sqrt(matxz.shape[0]))))

    matyz = np.zeros(sin1.T.shape[0]**2)
    count = 0
    for y in sin2.T:
        for z in sin3.T:
            correl = np.cbrt(((y-z)**2).mean(axis=0))
            #correl = scipy.signal.correlate(x,y, mode='valid')
            matyz[count] = correl
            count += 1
    matyz = np.reshape(matyz, (int(np.sqrt(matyz.shape[0])),int(np.sqrt(matyz.shape[0]))))

    
    fig, axes = plt.subplots(3, 3)

    #optional
    axes[0, 0].imshow(matxz)
    axes[0, 2].imshow(matxy)
    axes[2, 0].imshow(matyz)
    #########

    axes[0,0], minxz = find_min(matxz, axes[0,0])
    axes[0,2], minxy = find_min(matxy, axes[0,2])
    axes[2,0], minyz = find_min(matyz, axes[2,0])
    gapx = (minxz[0]-minxy[0])*360 / matxz.shape[0]
    gapy = (minxy[1]-minyz[0])*360 / matxz.shape[0]
    gapz = (minxz[1]-minyz[1])*360 / matxz.shape[0]
    return fig, gapx,gapy,gapz

def compare3_1(sin1, sin2, sin3):
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

    matyy = np.array([])
    for y in sin2.T:
        for z in sin2.T:
            correl = np.cbrt(((y-y)**2).mean(axis=0))
            matyy = np.append(matyy,correl)
    matyy = np.reshape(matyy, (int(np.sqrt(matyy.shape[0])),int(np.sqrt(matyy.shape[0]))))
    
    fig, axes = plt.subplots(2, 3)

    axes[0, 0].imshow(matxz)
    axes[0, 1].imshow(sin1.T)
    axes[0, 2].imshow(matxy)
    axes[1, 0].imshow(sin3)
    axes[1, 1].imshow(matyy)
    axes[1, 2].imshow(sin2)

    plotxz, minxz = find_min(matxz, axes[0,0])
    plotxy, minxy = find_min(matxy, axes[0,2])
    axes[0,0] = plotxz
    axes[0,2] = plotxy
    gap = (minxz[0] - minxy[0])*360 / matxz.shape[0]
    return fig, gap

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
    return plt, loc

def r(x):
    return np.round(x,3)
    

def direct3d(A,B,C):

    start3d = time.time()
    # Only measurable data:

    A = (A /360) * 2*math.pi  # angle between common tilt axes C12 and C31
    B = (B /360) * 2*math.pi  # angle between common tilt axes C12 and C31
    C = (C /360) * 2*math.pi  # angle between common tilt axes C12 and C31
    #######################################################

    # Finding Euler angles Be and Al between projections 1,2,3

    Be31 = A
    Be23 = B
    sin_Al23 = r((math.cos(C) - math.cos(Be23)*math.cos(Be31)) / (math.sin(Be23)*math.sin(Be31)))
    Al23 = r(math.asin(sin_Al23))

    C12 = (0,0,1) # Assume
    C31 = (0, math.sin(Be31), math.cos(Be31)) # Assumme in yz plane
    C31 = C31 / np.sqrt(C31[0]**2+C31[1]**2+C31[2]**2)
    C23 = (math.sin(Be23)*math.cos(Al23), math.sin(Be23)*math.sin(Al23), math.cos(Be23))
    C23 = C23 / np.sqrt(C23[0]**2+C23[1]**2+C23[2]**2)

    D1 = np.cross(C31,C12)
    D1 = D1 / np.sqrt(D1[0]**2+D1[1]**2+D1[2]**2)
    D2 = np.cross(C12,C23)
    D2 = D2 / np.sqrt(D2[0]**2+D2[1]**2+D2[2]**2)
    D3 = np.cross(C23,C31)
    D3 = D3 / np.sqrt(D3[0]**2+D3[1]**2+D3[2]**2)

    # D1
    D1Be =  math.acos(D1[2])
    # if math.sin(D1Be) == 0:
    #     D1Al = 0
    # else:
    #     D1Al = 0.5*math.asin(2*D1[0]*D1[1]/(math.sin(D1Be)**2))
    if D1[0] != 0:
        D1Al = math.atan(D1[1]/D1[0])
    else:
        D1Al = 90/360 * 2 * np.pi

    D1check = r((math.sin(D1Be)*math.cos(D1Al), math.sin(D1Be)*math.sin(D1Al), math.cos(D1Be)))

    # D2
    D2Be =  math.acos(D2[2])
    # if math.sin(D2Be) == 0:
    #     D2Al = 0
    # else:
    #     D2Al = 0.5*math.asin(2*D2[0]*D2[1]/(math.sin(D2Be)**2))
    if D2[0] != 0:
        D2Al = math.atan(D2[1]/D2[0])
    else:
        D2Al = 90/360 * 2 * np.pi

    D2check = r((math.sin(D2Be)*math.cos(D2Al), math.sin(D2Be)*math.sin(D2Al), math.cos(D2Be)))

    # D3
    D3Be = math.acos(D3[2])
    # if math.sin(D3Be) == 0:
    #     D3Al = 0
    # else:
    #     D3Al = 0.5*math.asin(2*D3[0]*D3[1]/(math.sin(D3Be)**2))
    if D3[0] != 0:
        D3Al = math.atan(D3[1]/D3[0])
    else:
        D3Al = 90/360 * 2 * np.pi

    D3check = r((math.sin(D3Be)*math.cos(D3Al), math.sin(D3Be)*math.sin(D3Al), math.cos(D3Be)))

    print('\n######## CHECKS #######')
    print('C12: ', C12)
    print('C31: ', C31)
    print('C23: ', C23)
    print('D1: ', r(D1))
    print('D1 check: ', D1check)
    print('D2: ', r(D2))
    print('D2 check: ', D2check)
    print('D3: ', r(D3))
    print('D3 check: ', D3check)
    print('\n######## RESULTS #######')

    print("(DXAl, DXBe)")
    print("D1: ", r((rad_deg(D1Al),rad_deg(D1Be))))
    print("D2: ", r((rad_deg(D2Al),rad_deg(D2Be))))
    print("D3: ", r((rad_deg(D3Al),rad_deg(D3Be))))
    end3d = time.time()
    print(f'direct3d time = {end3d-start3d}')




def rad_deg(x):
    return x*360/(2*math.pi)


########### MAIN #################

im1 = 'projections/4_1.mrcs'
im2 = 'projections/4_3.mrcs'
im3 = 'projections/4_4.mrcs'

with mrc.open(im1) as f:
    img1 = f.data
sin1, th, fig1 = sino(img1)

with mrc.open(im2) as f:
    img2 = f.data
sin2, th, fig2 = sino(img2)

with mrc.open(im3) as f:
    img3 = f.data
sin3, th, fig3 = sino(img3)

'''
fig01, gapx = compare3_1(sin1, sin2, sin3)
fig02, gapy = compare3_1(sin2, sin1, sin3)
fig03, gapz = compare3_1(sin3, sin2, sin1)
'''

start_comp = time.time()
fig04, gapx, gapy, gapz = compare3(sin1, sin2, sin3)
end_comp = time.time()
print(gapx,gapy,gapz)
direct3d(gapx,gapy,gapz)

end = time.time()
print(f'comp time = {end_comp-start_comp}')
print(f'total time = {end-start}')



'''
plt.figure(8)
plt.imshow(mat)
plt, loc = find_min(mat, plt)
plt.figure(9)
plt.plot(sin1.T[loc[0]])
plt.plot(sin2.T[loc[1]])
plt.show()
'''
plt.show()
