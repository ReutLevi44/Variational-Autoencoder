import numpy as np
import pandas as pd
import tensorflow.keras
from tensorflow.keras.losses import mse
from tensorflow.keras.models import Model
import os
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Reshape, Dropout, Input, Dense, Lambda
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.regularizers import l2


class VariationalAutoEncoder:
    def __init__(self, input_set, weights_path, encoding_dim=3, mu=None, log_sigma=None):
        self.encoding_dim = encoding_dim
        self.x = input_set
        self.input_shape = len(input_set[0])
        # different way
        # self.input_shape = input_set[0].shape[0]
        self.num_classes = 2  # binary classifier
        self.weights_path = weights_path
        self.mu = mu
        self.log_sigma = log_sigma

    def sample_z(self, args):
        mu, log_sigma = args
        eps = K.random_normal(shape=(K.shape(mu)[0], self.encoding_dim), mean=0.)
        return mu + K.exp(log_sigma / 2) * eps

    def vae_loss(self, y_true, y_pred):
        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
        # E[log P(X|z)]
        m = K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
        recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl = 0.5 * K.sum(K.exp(self.log_sigma) + K.square(self.mu) - 1. - self.log_sigma, axis=1)
        print(m, recon, kl)
        # return recon + kl
        return 5000 * m + kl

    def mean_squared_error(self, y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)

    def v_encoder_decoder(self):
        # Q(z|X) -- encoder
        inputs = Input(shape=self.x[0].shape)
        encoded1 = Dense(800, kernel_regularizer=l2(0.000001), bias_regularizer=l2(0.000001), activation='elu')(inputs)
        dropout1 = Dropout(0.2)(encoded1)
        encoded2 = Dense(1100, kernel_regularizer=l2(0.000001), bias_regularizer=l2(0.000001), activation='elu')(dropout1)
        dropout2 = Dropout(0.2)(encoded2)

        mu = Dense(self.encoding_dim, kernel_regularizer=l2(0.000001),
                   bias_regularizer=l2(0.000001), activation='linear')(dropout2)
        log_sigma = Dense(self.encoding_dim, kernel_regularizer=l2(0.000001),
                          bias_regularizer=l2(0.000001), activation='linear')(dropout2)
        self.mu = mu
        self.log_sigma = log_sigma

        # Sample z ~ Q(z|X)
        z = Lambda(self.sample_z, name='z_lambda')([mu, log_sigma])
        self.z = z

        encoder_model = Model(inputs, mu)
        self.encoder = encoder_model

        # P(X|z) -- decoder
        decoded1 = Dense(1100, kernel_regularizer=l2(0.000001), bias_regularizer=l2(0.000001), activation='elu')(z)
        dropout1 = Dropout(0.2)(decoded1)
        decoded2 = Dense(800, kernel_regularizer=l2(0.000001), bias_regularizer=l2(0.000001), activation='elu')(dropout1)
        dropout2 = Dropout(0.2)(decoded2)
        decoded3 = Dense(self.input_shape, kernel_regularizer=l2(0.000001),
                         bias_regularizer=l2(0.000001), activation='elu')(dropout2)
        reshape = Reshape((int(self.input_shape / 21), 21))(decoded3)
        decoded3 = Dense(21, activation='softmax')(reshape)
        reshape2 = Reshape(self.x[0].shape)(decoded3)

        # Overall VAE model, for reconstruction and training
        vae = Model(inputs, reshape2)

        self.v_autoencoder_model = vae

        return vae

    def fit_v_autoencoder(self, train_x, batch_size=10, epochs=300):
        self.v_autoencoder_model = multi_gpu_model(self.v_autoencoder_model, gpus=2)
        adam = tensorflow.keras.optimizers.Adam(lr=0.0001)
        self.v_autoencoder_model.compile(optimizer=adam, loss=self.vae_loss,
                                         experimental_run_tf_function=False, metrics=['mse'])
        log_dir = './log/'
        tb_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                             write_graph=True, write_images=True)
        es_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                               patience=20, verbose=0, mode='auto')
        results = self.v_autoencoder_model.fit(train_x, train_x, validation_split=0.2, verbose=2,
                                               epochs=epochs, batch_size=batch_size,
                                               callbacks=[tb_callback, es_callback])
        return results

    def save_vae(self):
        if not os.path.exists(r'./weights_' + self.weights_path):
            os.mkdir(r'./weights_' + self.weights_path)
        self.encoder.save(r'./weights_' + self.weights_path + '/encoder_weights.h5')
        # self.decoder.save(r'./weights_' + self.weights_path + '/decoder_weights.h5')
        self.v_autoencoder_model.save(r'./weights_' + self.weights_path + '/vae_weights.h5')
