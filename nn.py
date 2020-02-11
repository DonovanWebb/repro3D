#!/usr/bin/env python
import numpy as np
from skimage.transform import radon
import datetime
import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from sklearn.model_selection import train_test_split
import mrcfile
import scipy.stats
import matplotlib.pyplot as plt
import itertools as it


def load_mrc(path):
    with mrcfile.open(path) as f:
        image = f.data.T
    return image

    
def norm_dist(l1,l2):
    return scipy.stats.norm(75,4).pdf(l1)*scipy.stats.norm(75,4).pdf(l2) / (scipy.stats.norm(75,4).pdf(75)**2)
    

def split_sins_for_dnn(sin1, sin2, angle, X, Y):
    X = np.array([])
    Y = np.array([])
    for l1 in range(50,80):
        line1 = sin1[l1]
        for l2 in range(50,80):
            line2 = sin2[l2]
            x = [line1, line2]
            y = norm_dist(l1,l2)
            X = np.append(X,np.ravel(x))
            Y = np.append(Y, y)
            if l1 == 75 and l2 ==75:
                plt.figure(1)
                plt.plot(sin1[l1])
                plt.plot(sin2[l2])
                plt.show()
    X = np.reshape(X, [-1, Y.shape[0]])
    X = X.T # For keras convention
    print(X.shape)
    print(Y.shape)
    return X, Y

    
def norm_line(line):
    min_val = np.min(line)
    line_norm = line - min_val
    max_val = np.max(line_norm)
    line_norm = line_norm / max_val
    return line_norm


def split_sins_for_conv(sin1, sin2, angle, X, Y):
    for l1 in it.chain(range(30,70,10), range(70,81)):
        line1 = sin1[l1]
        line1 = norm_line(line1)
        for l2 in it.chain(range(30,70,10), range(70,81)):
            line2 = sin2[l2]
            line2 = norm_line(line2)
            x = [line1, line2]
            x = [[a,b] for a,b in zip(line1, line2)]
            y = norm_dist(l1,l2)
            X = np.append(X,np.ravel(x))
            Y = np.append(Y, y)
            # if l1 == 75 and l2 ==75:
            #     plt.figure(1)
            #     plt.plot(line1)
            #     plt.plot(line2)
            #     plt.show()
    X = np.reshape(X, [Y.shape[0],-1,2])
    # print(Y)
    # plt.figure(1)
    # plt.plot(X[5,:,0])
    # plt.plot(X[5,:,1])
    # plt.show()
    print(X.shape)
    print(Y.shape)
    return X, Y

    
def nn(X,Y):
    in_size = X.shape[1]

    model = Sequential()
    model.add(Dense(64, input_dim=in_size, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, Y, epochs=30, batch_size=100)
    
    _, accuracy = model.evaluate(X, Y)
    print('Accuracy: %.2f' % (accuracy*100))

 
def conv1D(X,Y):
    n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], 1

    model = Sequential()
    model.add(Conv1D(filters=100, kernel_size=5, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=100, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    # model.add(Conv1D(filters=100, kernel_size=10, activation='relu'))
    # model.add(Conv1D(filters=100, kernel_size=10, activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mse', optimizer='adam')
    model.summary()

    model.fit(X, Y, epochs=10, batch_size=128, verbose=1, shuffle=True)

    # _, accuracy = model.evaluate(X, Y, batch_size=32, verbose=0)
    # print('Accuracy: %.2f' % (accuracy*100))
    return model

    
def eval_m(model, sin1_l, sin2_l):

    x_input_1 = np.reshape(sin1_l, (len(sin1_l), 1))
    x_input_2 = np.reshape(sin2_l, (len(sin1_l), 1))
    x_input = np.hstack((x_input_1, x_input_2))
    x_input = np.reshape(x_input, (1, x_input.shape[0], x_input.shape[1]))
    yhat = model.predict(x_input, verbose=0)
    return yhat


def sin_comp(model, sin1, sin2):

    l_size = sin1.shape[0]
    X_inputs = np.zeros((l_size//2)**2*(l_size//2)*2)
    for x in range(0, l_size, 2):
        for y in range(0, l_size, 2):
            x_input_1 = np.reshape(sin1[x], (len(sin1[x]),1))
            x_input_2 = np.reshape(sin2[y], (len(sin1[x]),1))
            x_input = np.hstack((x_input_1, x_input_2))
            x_input = np.reshape(x_input, (1, x_input.shape[0], x_input.shape[1]))
            X_inputs[(x*l_size//2+y)*l_size//2:(x*l_size//2+y)*l_size//2+l_size] = x_input.ravel()
    X_inputs = np.reshape(X_inputs, (-1, x_input.shape[1], 2))
    print(X_inputs.shape)
    sin_comp = model.predict(X_inputs, verbose=1, batch_size=64)
    sin_comp = np.reshape(sin_comp, (150, 150))
    return sin_comp


def sino(image):

    theta = np.linspace(0., 360., 2*max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    return sinogram.T


def add_noise(image):
    dims = tuple(image.shape)
    mean = 0
    sigma = 0.5
    noise = np.random.normal(mean, sigma, dims)
    noisy_image = image + noise
    return noisy_image


def make_data():

    X = np.array([])
    Y = np.array([])
    count = 0
    for sinogram in os.listdir('/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/tmp/0'):
        mrc1 = load_mrc(f'/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/tmp/0/{sinogram}')
        mrc2 = load_mrc(f'/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/tmp/45/{sinogram}')
        noisy_mrc1 = add_noise(mrc1)
        noisy_mrc2 = add_noise(mrc2)
        sin1 = sino(noisy_mrc1)
        sin2 = sino(noisy_mrc2)
        if count < 240:
            X, Y = split_sins_for_conv(sin1, sin2, 0, X, Y)
            sin1_flip = np.fliplr(sin1)
            sin2_flip = np.fliplr(sin2)
            X, Y = split_sins_for_conv(sin1_flip, sin2_flip, 0, X, Y)
            count += 1
    np.save('X_noise.npy', X)
    np.save('Y_noise.npy', Y)


def main():
    # MAIN
    X = np.load('X.npy')
    Y = np.load('Y.npy')

    # Add noise to X
    noisyX = add_noise(X)

    X_train, X_test, Y_train,  Y_test = train_test_split(
        noisyX, Y, test_size=0.2, random_state=42)

    model = conv1D(X_train, Y_train)

    print('######### Evaluation of model #######')
    scores = model.evaluate(X_test, Y_test, verbose=1)
    print(scores)

    count = 0
    for mrc in os.listdir('/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/0/'):
        if count < 8:
            mrc1 = load_mrc(f'/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/0/{mrc}')
            noisy1 = add_noise(mrc1)
            # plt.figure(1)
            # plt.imshow(noisy1)
            # plt.show()
            mrc2 = load_mrc(f'/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/2/{mrc}')
            noisy2 = add_noise(mrc2)
            sin1 = sino(noisy1)
            sin2 = sino(noisy2)

            print(count)
            print('75,75', eval_m(model, sin1[75], sin2[75]))

            comp_all = sin_comp(model, sin1, sin2)
            plt.figure(2)
            plt.imshow(comp_all)
            plt.show()

            count += 1


main()
