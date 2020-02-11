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


def norm_dist(l1, l2):
    norm_at_0 = scipy.stats.norm(0, 4).pdf(l1)*scipy.stats.norm(0, 4).pdf(l2) / (scipy.stats.norm(0, 4).pdf(0)**2)
    norm_at_150 = scipy.stats.norm(150, 4).pdf(l1)*scipy.stats.norm(150, 4).pdf(l2) / (scipy.stats.norm(150, 4).pdf(150)**2)
    return norm_at_0 + norm_at_150


def norm_image(image):
    ''' Make conv's life easier by normalising input '''
    min_val = np.min(image)
    image_norm = image - min_val
    max_val = np.max(image_norm)
    image_norm = image_norm / max_val
    return image_norm


def split_sins_for_conv(sin1, sin2, angle, X, Y):
    '''
    Create X and Y data for network. X are pairs of single line, Y are
    similarity between two lines. If tilt is in y axis only then peak
    will be at (0,0). Normal dist with 1 at (0,0) and decays as it
    gets further.
    '''
    for l1 in it.chain(range(0, 5), range(30, 70, 10),  range(145, 155)):
        line1 = sin1[l1]
        line1 = norm_image(line1)
        for l2 in it.chain(range(0, 5), range(30, 70, 10),  range(145, 155)):
            line2 = sin2[l2]
            line2 = norm_image(line2)
            x = [line1, line2]
            x = [[a, b] for a, b in zip(line1, line2)]
            y = norm_dist(l1, l2)
            X = np.append(X, np.ravel(x))
            Y = np.append(Y, y)
            # if l1 == 153 and l2 == 153:
            #     plt.figure(1)
            #     plt.plot(line1)
            #     plt.plot(line2)
            #     plt.show()
            #     print(y)
    X = np.reshape(X, [Y.shape[0], -1, 2])
    print(X.shape)
    print(Y.shape)
    return X, Y


def nn(X, Y):
    '''
    Direct neural network for feature extraction. Old so make
    better before use!
    '''
    in_size = X.shape[1]
    model = Sequential()
    model.add(Dense(64, input_dim=in_size, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, epochs=30, batch_size=100)


def conv1D(X,Y):
    '''
    1 dimensional conv net for feature extraction. Output is
    regression where num (between 0 and 1) shows similarity between
    two input single lines. 1 is v.similar 0 is not.
    '''
    n_timesteps, n_features = X.shape[1], X.shape[2]

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

    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mse', optimizer='adam')
    model.summary()

    model.fit(X, Y, epochs=100, batch_size=128, verbose=1, shuffle=True)
    return model


def eval_m(model, sin1_l, sin2_l):
    ''' evaluate 2 lines as input '''
    x_input_1 = np.reshape(sin1_l, (len(sin1_l), 1))
    x_input_2 = np.reshape(sin2_l, (len(sin1_l), 1))
    x_input = np.hstack((x_input_1, x_input_2))
    x_input = np.reshape(x_input, (1, x_input.shape[0], x_input.shape[1]))
    yhat = model.predict(x_input, verbose=0)
    return yhat


def sin_comp(model, sin1, sin2):
    '''
    Compare two sinograms using trained model. Output is similarity
    map. Made much faster by generating large array of all possible
    pairs and then feeding all into model at once.
    '''
    l_size = sin1.shape[0]
    X_inputs = np.zeros((l_size//2)**2*(l_size//2)*2)
    for x in range(0, l_size, 2):
        for y in range(0, l_size, 2):
            x_input_1 = np.reshape(sin1[x], (len(sin1[x]), 1))
            x_input_2 = np.reshape(sin2[y], (len(sin1[x]), 1))
            x_input = np.hstack((x_input_1, x_input_2))
            x_input = np.reshape(
                      x_input, (1, x_input.shape[0], x_input.shape[1]))
            pos = (x*l_size//2 + y)*l_size//2
            X_inputs[pos:pos+l_size] = x_input.ravel()
    X_inputs = np.reshape(X_inputs, (-1, x_input.shape[1], 2))
    sin_comp = model.predict(X_inputs, verbose=1, batch_size=64)
    sin_comp = np.reshape(sin_comp, (150, 150))
    return sin_comp


def sino(image):
    ''' Make sinogram from image '''
    theta = np.linspace(0., 360., 2*max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    return sinogram.T


def add_noise(image):
    ''' Add gaussian noise to data '''
    dims = tuple(image.shape)
    mean = 0
    sigma = 0.3
    noise = np.random.normal(mean, sigma, dims)
    noisy_image = image + noise
    return noisy_image


def make_data():
    ''' Create data set and save it as file '''
    X = np.array([])
    Y = np.array([])
    count = 0
    for sinogram in os.listdir('/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/0/'):
        mrc1 = load_mrc(f'/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/0/{sinogram}')
        mrc2 = load_mrc(f'/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/1/{sinogram}')
        norm_mrc1 = norm_image(mrc1)
        norm_mrc2 = norm_image(mrc2)
        noisy_mrc1 = add_noise(norm_mrc1)
        noisy_mrc2 = add_noise(norm_mrc2)
        sin1 = sino(noisy_mrc1)
        sin2 = sino(noisy_mrc2)
        if count < 40:
            X, Y = split_sins_for_conv(sin1, sin2, 0, X, Y)
            sin1_flip = np.fliplr(sin1)
            sin2_flip = np.fliplr(sin2)
            X, Y = split_sins_for_conv(sin1_flip, sin2_flip, 0, X, Y)
            count += 1
    np.save('X_noise_03.npy', X)
    np.save('Y_noise_03.npy', Y)


def main():
    X = np.load('X_noise_03.npy')
    Y = np.load('Y_noise_03.npy')
    noisyX = add_noise(X)
    X_train, X_test, Y_train,  Y_test = train_test_split(
        noisyX, Y, test_size=0.2, random_state=42)

    model = conv1D(X_train, Y_train)

    print('######### Evaluation of model #########')
    scores = model.evaluate(X_test, Y_test, verbose=1)
    print(scores)

    count = 0
    for mrc in os.listdir('/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/0/'):
        if count < 8:
            mrc1 = load_mrc(f'/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/0/{mrc}')
            mrc2 = load_mrc(f'/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/1/{mrc}')
            noisy1 = add_noise(mrc1)
            noisy2 = add_noise(mrc2)
            # plt.figure(1)
            # plt.imshow(noisy1)
            # plt.show()
            sin1 = sino(noisy1)
            sin2 = sino(noisy2)

            print(count)
            print('75,75', eval_m(model, sin1[75], sin2[75]))

            comp_all = sin_comp(model, sin1, sin2)
            plt.figure(2)
            plt.imshow(comp_all)
            plt.show()

            count += 1


make_data()
main()
