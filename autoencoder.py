import numpy as np
from PIL import Image
from scipy.misc import toimage
from random import randint

from keras.datasets import mnist
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
K.set_image_dim_ordering('tf')

def build_model():
    input_img = Input(shape=(14, 14, 1))
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)
    
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(16, 3, 3, activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder

def load_data():
    (Y_train, _), (Y_test, _) = mnist.load_data() # load MNIST dataset

    X_train = [] # create scaled down images
    for x in Y_train:
        im = toimage(x)
        im = im.resize((14,14), Image.ANTIALIAS)
        a = np.asarray(im)
        X_train.append(a)
    X_train = np.asarray(X_train)

    X_test = [] # same with test data
    for x in Y_test:
        im = toimage(x)
        im = im.resize((14,14), Image.ANTIALIAS)
        a = np.asarray(im)
        X_test.append(a)
    X_test = np.asarray(X_test)
    # normalize data
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    Y_train = Y_train.astype('float32') / 255.
    Y_test = Y_test.astype('float32') / 255.
    # add 1 extra dimension so single input vector looks like [[[]]]
    X_train = np.reshape(X_train, (len(X_train), 14, 14, 1))
    X_test = np.reshape(X_test, (len(X_test), 14, 14, 1))
    Y_train = np.reshape(Y_train, (len(Y_train), 28, 28, 1))
    Y_test = np.reshape(Y_test, (len(Y_test), 28, 28, 1))
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = load_data()
autoencoder = build_model()
autoencoder.summary()
autoencoder.fit(X_train, Y_train,
                nb_epoch=15,
                batch_size=50,
                shuffle=True,
                validation_data=(X_test, Y_test))
          
while True:
    i = randint(1,len(X_test))
    representation = autoencoder.predict(X_test[i:i+1])[0]
    input_image = toimage(X_test[i].reshape((1,14,14))[0])
    repr_image = toimage(representation.reshape((1,28,28))[0])
    true_image = toimage(Y_test[i].reshape((1,28,28))[0])
    input_image.show()
    true_image.show()
    repr_image.show()
    raw_input()
    
    
