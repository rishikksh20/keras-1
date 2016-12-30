from __future__ import print_function
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras.backend as K
from keras.layers import Input
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.datasets import mnist
from keras.models import Model

K.set_image_dim_ordering('th')

random.seed(2016)

img_rows, img_cols = 28, 28

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


def plot_loss(losses):
        plt.figure(figsize=(10, 8))
        plt.plot(losses["disc_loss"], label='discriminitive loss')
        plt.plot(losses["gen_loss"], label='generative loss')
        plt.legend()
        plt.show()

# Generative model


def generator_model(gen_shape):
    gen_input = Input(shape=[gen_shape])
    model = Dense(200*14*14, init='glorot_normal')(gen_input)
    model = BatchNormalization(mode=2)(model)
    model = Activation('relu')(model)
    model = Reshape([200, 14, 14])(model)
    model = UpSampling2D(size=(2, 2))(model)
    model = Convolution2D(100, 3, 3, border_mode='same', init='glorot_uniform')(model)
    model = BatchNormalization(mode=2)(model)
    model = Activation('relu')(model)
    model = Convolution2D(50, 3, 3, border_mode='same', init='glorot_uniform')(model)
    model = BatchNormalization(mode=2)(model)
    model = Activation('relu')(model)
    model = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(model)
    gen_output = Activation('sigmoid')(model)
    generator = Model(gen_input, gen_output)
    return generator

# Discriminative model


def discriminator_model(disc_shape):
    disc_input = Input(shape=disc_shape)
    model = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode='same', activation='relu')(disc_input)
    model = LeakyReLU(0.2)(model)
    model = Dropout(0.2)(model)
    model = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode='same', activation='relu')(model)
    model = LeakyReLU(0.2)(model)
    model = Dropout(0.2)(model)
    model = Flatten()(model)
    model = Dense(256)(model)
    model = LeakyReLU(0.2)(model)
    model = Dropout(0.2)(model)
    disc_output = Dense(2, activation='softmax')(model)
    discriminator = Model(disc_input, disc_output)
    return discriminator

# GAN model


def gan(generator, discriminator):
    gan_input = Input(shape=[100])
    H = generator(gan_input)
    discriminator.trainable = False
    gan_output = discriminator(H)
    return Model(gan_input, gan_output)


gen_shape = 100
disc_shape = X_train.shape[1:]
nb_epoch = 5000
BATCH_SIZE = 32
discriminator = discriminator_model(disc_shape)
generator = generator_model(gen_shape)
gen_ad_network = gan(generator, discriminator)
generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4))
gen_ad_network.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4))
discriminator.trainable = True
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3))


# set up loss vector
gan_loss = {"disc_loss": [], "gen_loss": []}


def train(nb_epoch, BATCH_SIZE):
    progress_bar = Progbar(target=nb_epoch)
    for index in range(nb_epoch):
        progress_bar.update(index)
        # Generate image batch from training data
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE), :, :, :]
        # Generate random noise image
        noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        # Generate Images from generator
        generated_images = generator.predict(noise_gen)
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE, 2])
        y[0: BATCH_SIZE, 1] = 1
        y[BATCH_SIZE:, 0] = 1
        # Train discriminator on generated images
        disc_loss = discriminator.train_on_batch(X, y)
        gan_loss["disc_loss"].append(disc_loss)
        noise_tr = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        y2 = np.zeros([BATCH_SIZE, 2])
        y2[:, 1] = 1
        gen_loss = gen_ad_network.train_on_batch(noise_tr, y2)
        gan_loss["gen_loss"].append(gen_loss)

train(nb_epoch=6000, BATCH_SIZE=64)
plot_loss(gan_loss)


noise = np.random.uniform(0, 1, size=[25, 100])
generated_images = generator.predict(noise)
plt.figure(figsize=(12, 12))
for i in range(generated_images.shape[0]):
    plt.subplot(5, 5, i+1)
    img = generated_images[i, 0, :, :]
    plt.imshow(img)
    plt.axis('off')

plt.tight_layout()
plt.show()
