#!/usr/bin/env python
import numpy as np
from skimage.transform import radon
from datetime import datetime
import tensorflow as tf
import os
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.models import model_from_json
from keras import metrics
from sklearn.model_selection import train_test_split
import mrcfile
import scipy.stats
import matplotlib.pyplot as plt
import itertools as it

import multi_model


def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model
    

def load_mrc(path):
    with mrcfile.open(path) as f:
        image = f.data.T
    return image


def norm_dist(l1, l2):
    ''' Normal distribution '''
    sig = 4
    norm_at_0 = scipy.stats.norm(0, sig).pdf(l1)*scipy.stats.norm(0, sig).pdf(l2) / (scipy.stats.norm(0, sig).pdf(0)**2)
    norm_at_150 = scipy.stats.norm(150, sig).pdf(l1)*scipy.stats.norm(150, sig).pdf(l2) / (scipy.stats.norm(150, sig).pdf(150)**2)
    val = round(norm_at_0 + norm_at_150,3)
    return val


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
    for l1 in it.chain(range(0, 5), range(30, 70, 10),  range(145, 151)):
        line1 = sin1[l1]
        line1 = norm_image(line1)
        for l2 in it.chain(range(0, 5), range(30, 70, 10),  range(145, 151)):
            line2 = sin2[l2]
            line2 = norm_image(line2)
            x = [line1, line2]
            x = [[a, b] for a, b in zip(line1, line2)]
            y = norm_dist(l1, l2)
            X = np.append(X, np.ravel(x))
            Y = np.append(Y, y)
            # if l1 == 0 and l2 == 0:
            #     plt.figure(1)
            #     plt.plot(line1)
            #     plt.plot(line2)
            #     plt.show()
            #     print(y)
        # print(Y)
    X = np.reshape(X, [Y.shape[0], -1, 2])
    print(X.shape)
    print(Y.shape)
    return X, Y
    

def split_sins_for_conv_no_angle(sin1,sin2, X, Y):
    ''' provide 2 of the same sinogram with different noise '''
    for i in range(100):
        line1 = norm_image(sin1[i])
        line2 = norm_image(sin2[i])

        x = [line1, line2]
        x = [[a, b] for a, b in zip(line1, line2)]
        y = 1
        X = np.append(X, np.ravel(x))
        Y = np.append(Y, y)
    for i in range(100):
        line1 = norm_image(sin1[i])
        line2 = norm_image(sin2[i+np.random.randint(20,50)])

        x = [line1, line2]
        x = [[a, b] for a, b in zip(line1, line2)]
        y = 0
        X = np.append(X, np.ravel(x))
        Y = np.append(Y, y)
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
    model.fit(X, Y, epochs=10, batch_size=100)


def conv1D(X_train, Y_train, X_test, Y_test, n_filts=64, k_size=3, test_name='no_name'):
    '''
    1 dimensional conv net for feature extraction. Output is
    regression where num (between 0 and 1) shows similarity between
    two input single lines. 1 is v.similar 0 is not.
    '''
    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]

    if test_name == 'no_name':
        now = datetime.now()
        test_name = now.strftime("%d/%m_%H:%M")
    logdir = "logs/scalars/" + test_name
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)

    model = Sequential()
    model.add(Conv1D(filters=n_filts, kernel_size=k_size, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=n_filts, kernel_size=k_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='hard_sigmoid'))

    model.compile(loss='mse', optimizer='adam', metrics=[metrics.mae])
    # model.summary()

    batch_size = 256
    training_history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        verbose=2,
        epochs=200,
        validation_data=(X_test, Y_test),
        callbacks=[tensorboard_callback],
        shuffle=True)
    print("Average test loss: ", np.average(training_history.history['loss']))
    _, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
    return model, accuracy


def eval_m(model, sin1_l, sin2_l):
    ''' evaluate 2 lines as input '''
    x_input_1 = np.reshape(sin1_l, (-1,len(sin1_l), 1))
    x_input_2 = np.reshape(sin2_l, (-1,len(sin1_l), 1))
    x_input = np.hstack((x_input_1, x_input_2))
    x_input = np.reshape(x_input, (1, x_input.shape[0], x_input.shape[1]))
    if True:
        yhat = model.predict([x_input_1,x_input_2], verbose=2)
    else:
        yhat = model.predict(x_input, verbose=2)
    return yhat


def sin_comp(model, sin1, sin2):
    '''
    Compare two sinograms using trained model. Output is similarity
    map. Made much faster by generating large array of all possible
    pairs and then feeding all into model at once.
    '''
    l_size = sin1.shape[0]
    X_inputs = np.zeros((l_size)**2*(l_size//2)*2)
    for x in range(0, l_size):
        line1 = norm_image(sin1[x])
        for y in range(0, l_size):
            line2 = norm_image(sin2[y])
            x_input_1 = np.reshape(line1, (len(sin1[x]), 1))
            x_input_2 = np.reshape(line2, (len(sin1[x]), 1))
            x_input = np.hstack((x_input_1, x_input_2))
            x_input = np.reshape(
                      x_input, (1, x_input.shape[0], x_input.shape[1]))
            pos = (x*l_size + y)*l_size
            X_inputs[pos:pos+l_size] = x_input.ravel()
    X_inputs = np.reshape(X_inputs, (-1, x_input.shape[1], 2))
    if True:
        X1_input = X_inputs[:,:,0]
        X2_input = X_inputs[:,:,1]

        X1_len = X1_input.shape[1]

        X1_input = np.reshape(X1_input,(-1,X1_len,1))
        X2_input = np.reshape(X2_input,(-1,X1_len,1))

        sin_comp = model.predict([X1_input,X2_input], verbose=2, batch_size=256)
    else:
        sin_comp = model.predict(X_inputs, verbose=2, batch_size=256)
    sin_comp = np.reshape(sin_comp, (300, 300))
    return sin_comp


def create_circular_mask(h, w, center=None, radius=None, soft_edge=False):
    '''
    Artefacts occur if do sinogram on unmasked particle in noise
    This soft edged mask fixes this
    '''
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    #soft_edge
    if soft_edge:
        radius = radius - 7
        dist_adj = (1/(dist_from_center+0.001-radius))
        dist_adj[dist_adj < 0] = 1
        dist_adj[dist_adj > 1] = 1
        inds = np.where(np.isnan(dist_adj))
        dist_adj[inds] = np.take(1, inds[1])
        return dist_adj**1.5
    else:
        mask = dist_from_center <= radius
        return mask


def sino(image,num=1):
    ''' Make sinogram from image '''
    mask1 = create_circular_mask(image.shape[0], image.shape[0], soft_edge=True)
    mask2 = create_circular_mask(image.shape[0], image.shape[0])
    mask = mask1*mask2
    masked = mask*image
    theta = np.linspace(0., 360., 2*max(image.shape), endpoint=False)
    sinogram = radon(masked, theta=theta, circle=True)
    sinogram_mask = radon(mask, theta=theta, circle=True)
    plt.figure(num)
    plt.imshow(sinogram, cmap='gray')
    plt.figure(num+10)
    plt.imshow(masked, cmap='gray')
    return sinogram.T


def add_noise(image, sigma=0.2):  # Try colored noise and shot noise
    ''' Add gaussian noise to data '''
    dims = tuple(image.shape)
    mean = 0
    noise = np.random.normal(mean, sigma, dims)
    noisy_image = image + noise
    norm_noisy = norm_image(noisy_image)
    return norm_noisy
    

def make_data_alt(num_sample = 100):
    ''' 
    Create data set and save it as file 
    alt: no angle info
    '''
    X = np.array([])
    Y = np.array([])
    count = 0
    for sinogram in os.listdir('/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/0/'):
        mrc1 = load_mrc(f'/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/0/{sinogram}')
        mrc2 = mrc1
        norm_mrc1 = norm_image(mrc1)
        norm_mrc2 = norm_image(mrc2)
        noisy_mrc1 = add_noise(norm_mrc1)
        noisy_mrc2 = add_noise(norm_mrc2)
        sin1 = sino(noisy_mrc1)
        sin2 = sino(noisy_mrc2)
        if count < num_sample:
            print(f'{count}/{num_sample}')
            X, Y = split_sins_for_conv_no_angle(sin1, sin2, X, Y)
            count += 1
    np.save('datasets/X_02_bin_alt.npy', X)
    np.save('datasets/Y_02_bin_alt.npy', Y)


def make_data(num_sample = 100):
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
        if count < num_sample:
            print(f'{count}/{num_sample}')
            X, Y = split_sins_for_conv(sin1, sin2, 0, X, Y)
            # sin1_flip = np.fliplr(sin1)
            # sin2_flip = np.fliplr(sin2)
            # X, Y = split_sins_for_conv(sin1_flip, sin2_flip, 0, X, Y)
            count += 1
    np.save('datasets/X_02_bin.npy', X)
    np.save('datasets/Y_02_bin.npy', Y)
    

def summarize_results(scores, params):
    ''' For experiment '''
    print(scores, params)
    # summarize mean and standard deviation
    for i in range(len(scores)):
        m, s = np.mean(scores[i]), np.std(scores[i])
        print('Param=%d: %.3f%% (+/-%.3f)' % (params[i], m, s))
    # boxplot of scores
    plt.boxplot(scores, labels=params)
    plt.savefig('exp_cnn_kernel.png')


def run_experiment(X,Y, params, repeats=3):
    ''' Test a few params at once '''
    X_train, X_test, Y_train,  Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    # test each parameter
    all_scores = list()
    for p in params:
        # repeat experiment
        scores = list()
        for r in range(repeats):
            _, score = conv1D(X_train, Y_train, X_test, Y_test, 32, p, f'multi_head_k_size_{p}')
            score = score * 100.0
            print('>p=%d #%d: %.3f' % (p, r+1, score))
            scores.append(score)
        all_scores.append(scores)

    summarize_results(all_scores, params)


def visualise_layers(model):
    ''' visualise learnt 1D filters (first few) '''
    for layer in model.layers:
        if 'conv' not in layer.name:
            continue
        filters, biases = layer.get_weights()
        print(layer.name,filters.shape)
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        # plot first few filters
        n_filters, ix, cols = 50, 1, 5
        for i in range(n_filters):
            # get the filter
            f = filters[:, :, i]
            ax = plt.subplot(n_filters//cols, cols, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.plot(f)
            ix += 1
        # show the figure
        plt.show()
        

def main(retrain):
    if retrain:
        X = np.load('datasets/X_02_bin.npy')    # Read Data
        Y = np.load('datasets/Y_02_bin.npy')
        X_train, X_test, Y_train,  Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42)

        # model, score = conv1D(X_train, Y_train, X_test, Y_test, 128, 15, f'f128k15e50_linear')
        model, score = multi_model.multi_headed(X_train, Y_train, X_test, Y_test, 128, 19, f'multi_headed_128_19_1conv_02noise_large')
        visualise_layers(model)
        save_model(model)
    else:
        model = load_model()
        visualise_layers(model)

    count = 0
    for mrc in os.listdir('/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/0/'):
        if count < 10:
            mrc1 = load_mrc(f'/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/0/{mrc}')
            mrc2 = load_mrc(f'/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/1/{mrc}')
            norm_mrc1 = norm_image(mrc1)
            norm_mrc2 = norm_image(mrc2)
            noisy1 = add_noise(norm_mrc1)
            noisy2 = add_noise(norm_mrc2)
            # plt.figure(1)
            # plt.imshow(noisy1)
            # plt.show()
            sin1 = sino(noisy1,0)
            sin2 = sino(noisy2,1)

            print(count)
            print('75,75', eval_m(model, sin1[75], sin2[75]))

            comp_all = sin_comp(model, sin1, sin2)
            plt.figure(2)
            plt.imshow(comp_all)
            plt.show()

            count += 1


if __name__ == '__main__':
    # make_data() 
    # main(retrain=True)
    multi_model.carry_on()
    # X = np.load('datasets/X_large.npy')    # Read Data
    # Y = np.load('datasets/Y_large.npy')
    # k_sizes = [13,15,17,19]
    # filt_sizes = [16,32,64,128]
    # run_experiment(X,Y,k_sizes)
