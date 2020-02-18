import numpy as np
from keras import callbacks
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.models import model_from_json
import tensorflow as tf
from keras import metrics

'''
input_layer 150
conv
conv
maxpool
flatten
'''

def multi_headed(X_train, Y_train, X_test, Y_test, n_filts=64, k_size=3, test_name='no_name'):
    batch_size = 128
    X1_train = X_train[:,:,0]
    X2_train = X_train[:,:,1]
    X1_test = X_test[:,:,0]
    X2_test = X_test[:,:,1]

    X1_len = X1_train.shape[1]

    X1_train = np.reshape(X1_train,(-1,X1_len,1))
    X2_train = np.reshape(X2_train,(-1,X1_len,1))
    X1_test = np.reshape(X1_test,(-1,X1_len,1))
    X2_test = np.reshape(X2_test,(-1,X1_len,1))

    if test_name == 'no_name':
        now = datetime.now()
        test_name = now.strftime("%d/%m_%H:%M")

    logdir = "logs/scalars/" + test_name
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
    

    conv1 = Conv1D(filters=n_filts, kernel_size=k_size, activation='relu')
    conv2 = Conv1D(filters=n_filts, kernel_size=k_size, activation='relu')
    pool1 = MaxPooling1D(pool_size=2)
    flat1 = Flatten()
    
    input1 = Input(shape=(150,1))
    input2 = Input(shape=(150,1))

    #use the layers in head 1
    head1 = conv1(input1)   
    # head1 = conv2(head1)   
    head1 = Dropout(0.5)(head1)
    head1 = pool1(head1)   
    head1 = flat1(head1)
    #use the layers in head 2
    head2 = conv1(input2)   
    # head2 = conv2(head2)   
    head2 = Dropout(0.5)(head2)
    head2 = pool1(head2)   
    head2 = flat1(head2)

    out = Concatenate()([head1,head2])
 
    out = Dense(50, activation='relu')(out)
    out = Dense(1, activation='hard_sigmoid')(out)
    model = Model(inputs=[input1, input2], outputs=out)

    '''
    # Head 1
    inputs1 = Input(shape=(150))
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)

    # head 2
    inputs2 = Input(shape=(150))
    conv2 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)

    # merge
    merged = concatenate([flat1, flat2])
    '''

    # interpretation
    model.compile(loss='mse', optimizer='adam', metrics=[metrics.mae])

    # model.summary()

    training_history = model.fit(
        [X1_train,X2_train],
        Y_train,
        batch_size=batch_size,
        verbose=2,
        epochs=200,
        validation_data=([X1_test,X2_test], Y_test),
        callbacks=[tensorboard_callback],
        shuffle=True)

    print("Average test loss: ", np.average(training_history.history['loss']))
    _, accuracy = model.evaluate([X1_test,X2_test], Y_test, batch_size=batch_size, verbose=0)
    return model, accuracy

def carry_on():
        logdir = "logs/scalars/carrying_on" 
        tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
        model = load_model()
        model.compile(loss='mse', optimizer='adam', metrics=[metrics.mae])

        X = np.load('datasets/X_02_bin.npy')    # Read Data
        Y = np.load('datasets/Y_02_bin.npy')
        X_train, X_test, Y_train,  Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42)
        X1_train = X_train[:,:,0]
        X2_train = X_train[:,:,1]
        X1_test = X_test[:,:,0]
        X2_test = X_test[:,:,1]

        X1_len = X1_train.shape[1]

        X1_train = np.reshape(X1_train,(-1,X1_len,1))
        X2_train = np.reshape(X2_train,(-1,X1_len,1))
        X1_test = np.reshape(X1_test,(-1,X1_len,1))
        X2_test = np.reshape(X2_test,(-1,X1_len,1))

        training_history = model.fit(
            [X1_train,X2_train],
            Y_train,
            batch_size=128,
            verbose=2,
            epochs=200,
            validation_data=([X1_test,X2_test], Y_test),
            callbacks=[tensorboard_callback],
            shuffle=True)

        visualise_layers(model)
        save_model(model)
    
