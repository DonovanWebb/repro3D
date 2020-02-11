import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mrcfile as mrc


g_size = 361

def get_proj(path):
    with mrc.open(path) as f:
        im = f.data
    return im

def d3(grid, vector):

    vector = vector / np.sqrt(np.sum(vector**2))
    vector_n1 = np.array([-vector[1], vector[0], 0])
    vector_n1 = vector_n1 / np.sqrt(np.sum(vector_n1**2))
    vector_n2 = np.array([-vector[0], vector[1], (vector[0]**2 + vector[1]**2)/vector[2]])
    vector_n2 = vector_n2 / np.sqrt(np.sum(vector_n2**2))
    for i in range(-2*g_size,2*g_size):
        if i == 0:
            for nx in range(-75,76):
                for ny in range(-75,76):
                    vector_x = np.array([g_size/2,g_size/2, g_size/2]) + nx*vector_n1 + ny*vector_n2
                    vector_x = np.floor(vector_x)
                    vector_x = vector_x.astype(int)
                    if vector_x[0] < 0 or vector_x[1] < 0 or vector_x[2] < 0:
                        continue
                    try:
                        grid[vector_x[0], vector_x[1], vector_x[2]] = 1
                    except IndexError as e:
                        pass
            # for ny in range(-5,5):
            #     vector_y = np.array([g_size/2,g_size/2, g_size/2]) + n*vector_n2
            #     vector_y = np.floor(vector_y)
            #     vector_y = vector_y.astype(int)
            #     if vector_y[0] < 0 or vector_y[1] < 0 or vector_y[2] < 0:
            #         continue
            #     try:
            #         grid[vector_y[0], vector_y[1], vector_y[2]] = 1
            #     except IndexError as e:
            #         pass


        vector_ = np.array([g_size/2,g_size/2, g_size/2]) + i/2 * vector
        vector_ = np.floor(vector_)
        vector_ = vector_.astype(int)
        if vector_[0] < 0 or vector_[1] < 0 or vector_[2] < 0:
            continue
        try:
            grid[vector_[0], vector_[1], vector_[2]] = 1
        except IndexError as e:
            pass
    return grid

def d2(grid, vector, num):
    activated = grid*0

    vector = vector / np.sqrt(np.sum(vector**2))
    vector_n = np.array([-vector[1], vector[0]])
    for i in range(-2*g_size,2*g_size):

        for n in range(-27,28):
            vector_ = np.array([g_size/2,g_size/2]) + i/2 * vector + n*vector_n
            vector_ = np.floor(vector_)
            vector_ = vector_.astype(int)
            if vector_[0] < 0 or vector_[1] < 0:
                continue
            try:
                #if grid[vector_[0], vector_[1]] == num-1:
                #grid[vector_[0], vector_[1]] = abs(n)+1
                if activated[vector_[0], vector_[1]] == 0:
                    grid[vector_[0], vector_[1]] += abs(n) + 1
                    activated[vector_[0], vector_[1]] = 1
            except IndexError as e:
                pass



    #grid = grid * ((activated*0.1)+0.9)
    #grid = grid * activated
    #grid[grid == num - 1] = 0
    return grid
    
def eu2vec(eu):
    eu = eu * 2 * np.pi / 360
    return np.array([math.sin(eu[1])*math.cos(eu[0]), math.sin(eu[1])*math.sin(eu[0]), math.cos(eu[1])])
    
proj1 = 'projections/0_0.mrcs'
pro1 = get_proj(proj1)
print(pro1.shape)
eu1 = np.array([0,-90])
vec1 = eu2vec(eu1)

proj2 = 'projections/0_1.mrcs'
pro2 = get_proj(proj1)
eu2 = np.array([-44.5,-90])
vec2 = eu2vec(eu2)

proj3 = 'projections/0_2.mrcs'
pro3 = get_proj(proj1)
eu3 = np.array([-4.5,133])
vec3 = eu2vec(eu3)

grid = np.zeros((g_size,g_size, g_size))
grid = d3(grid, vec1)
# grid = d3(grid, vec2)
# grid = d3(grid, vec3)

# for x in range(0,360, 1):
#     vector = np.array([math.sin(x/360 * 2* np.pi),math.cos(x/360 * 2* np.pi)])
#     grid = d2(grid, vector, x)
ax = plt.axes(projection='3d')

x_coord = []
y_coord = []
z_coord = []
for x in range(grid.shape[0]):
    for y in range(grid.shape[1]):
        for z in range(grid.shape[2]):
            if grid[x,y,z] != 0:
                x_coord.append(x)
                y_coord.append(y)
                z_coord.append(z)

ax.scatter3D(x_coord, y_coord, z_coord, 'gray')
# plt.imshow(grid)
plt.show()


                 
