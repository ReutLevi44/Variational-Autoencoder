import tensorflow.keras
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, Layer, Lambda
from tensorflow.keras.models import Model
import os
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import multi_gpu_model


class VariationalVAutoEncoder:
    def __init__(self, input_set, weights_path, vs_len, encoding_dim=3, mu=None, log_sigma=None):
        self.encoding_dim = encoding_dim
        self.x = input_set
        self.input_shape = len(input_set[0])
        self.vs_len = vs_len
        self.aa_shape = self.input_shape - self.vs_len
        self.lambda1 = self.create_lambda1()
        self.lambda2 = self.create_lambda2()
        self.weights_path = weights_path
        self.mu = mu
        self.log_sigma = log_sigma
        print(self.x)

    def sample_z(self, args):
        mu, log_sigma = args
        eps = K.random_normal(shape=(K.shape(mu)[0], self.encoding_dim), mean=0.)
        return mu + K.exp(log_sigma / 2) * eps

    def vae_loss(self, y_true, y_pred):
        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
        # E[log P(X|z)]
        m = K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl = 0.5 * K.sum(K.exp(self.log_sigma) + K.square(self.mu) - 1. - self.log_sigma, axis=1)
        return 5000 * m + kl

    def create_lambda1(self):
        return self.MyLambda1(self.vs_len, self.aa_shape)

    def create_lambda2(self):
        return self.MyLambda2(self.vs_len)

    class MyLambda1(Layer):
        # this class is Lambda layer for the CDR3 input one hot split
        # exactly as writing instead: decoded3_0 = Lambda(lambda x: x[:, :-self.vs_len])(x)
        def __init__(self, vs_n, aa_n, **kwargs):
            self.vs_n = vs_n
            self.aa_n = aa_n
            super(VariationalVAutoEncoder.MyLambda1, self).__init__(**kwargs)

        def build(self, input_shape):
            super(VariationalVAutoEncoder.MyLambda1, self).build(input_shape)  # Be sure to call this at the end

        def call(self, x):
            return x[:, :-self.vs_n]

        def compute_output_shape(self, input_shape):
            return input_shape[0], self.aa_n

        def get_config(self):
            config = super(VariationalVAutoEncoder.MyLambda1, self).get_config()
            config['vs_n'] = self.vs_n
            config['aa_n'] = self.aa_n
            return config

    class MyLambda2(Layer):
        # this class is Lambda layer for the Vs input one hot split
        # exactly as writing instead: decoded3_1 = Lambda(lambda x: x[:, -self.vs_len:])(x)
        def __init__(self, vs_n, **kwargs):
            self.vs_n = vs_n
            super(VariationalVAutoEncoder.MyLambda2, self).__init__(**kwargs)

        def build(self, input_shape):
            super(VariationalVAutoEncoder.MyLambda2, self).build(input_shape)  # Be sure to call this at the end

        def call(self, x):
            return x[:, -self.vs_n:]

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.vs_n)

        def get_config(self):
            config = super(VariationalVAutoEncoder.MyLambda2, self).get_config()
            config['vs_n'] = self.vs_n
            return config

    def v_encoder_decoder(self):
        # Q(z|X) -- encoder
        inputs = Input(shape=self.x[0].shape)
        encoded1 = Dense(800, kernel_regularizer=l2(0.000001), bias_regularizer=l2(0.000001), activation='elu')(
            inputs)
        dropout1 = Dropout(0.2)(encoded1)
        encoded2 = Dense(1100, kernel_regularizer=l2(0.000001), bias_regularizer=l2(0.000001), activation='elu')(
            dropout1)
        dropout2 = Dropout(0.2)(encoded2)

        mu = Dense(self.encoding_dim, kernel_regularizer=l2(0.000001),
                   bias_regularizer=l2(0.000001), activation='linear')(dropout2)
        log_sigma = Dense(self.encoding_dim, kernel_regularizer=l2(0.000001),
                          bias_regularizer=l2(0.000001), activation='linear')(dropout2)
        self.mu = mu
        self.log_sigma = log_sigma

        # Sample z ~ Q(z|X)
        z = Lambda(self.sample_z, name='z_lambda')([mu, log_sigma])

        encoder_model = Model(inputs, mu)
        self.encoder = encoder_model

        # P(X|z) -- decoder
        decoded1 = Dense(1100, kernel_regularizer=l2(0.000001), bias_regularizer=l2(0.000001), activation='elu')(z)
        dropout1 = Dropout(0.2)(decoded1)
        decoded2 = Dense(800, kernel_regularizer=l2(0.000001), bias_regularizer=l2(0.000001), activation='elu')(dropout1)
        dropout2 = Dropout(0.2)(decoded2)
        x = Dense(self.input_shape, kernel_regularizer=l2(0.000001),
                  bias_regularizer=l2(0.000001), activation='elu')(dropout2)
        # split the softmax layer for the 'aa one-hots' and 'v one-hot'
        decoded3_0 = self.lambda1(x)
        decoded3_1 = self.lambda2(x)
        reshape_0 = Reshape((int(self.aa_shape / 21), 21))(decoded3_0)
        reshape_1 = Reshape((1, self.vs_len))(decoded3_1)
        decoded4_0 = Dense(21, activation='softmax')(reshape_0)
        decoded4_1 = Dense(self.vs_len, activation='softmax')(reshape_1)
        reshape2_0 = Reshape((self.aa_shape,))(decoded4_0)
        reshape2_1 = Reshape((self.vs_len,))(decoded4_1)
        both = tensorflow.keras.layers.concatenate([reshape2_0, reshape2_1], axis=1)

        # Overall VAE model, for reconstruction and training
        # model = Model(inputs, [both, both])
        model = Model(inputs, both)

        self.v_autoencoder_model = model

        print(model.summary())

        return model

    def fit_v_autoencoder(self, train_x, batch_size=10, epochs=300):
        # self.v_autoencoder_model = multi_gpu_model(self.v_autoencoder_model, gpus=3)
        adam = tensorflow.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.v_autoencoder_model.compile(optimizer=adam, loss=self.vae_loss, experimental_run_tf_function=False)
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
        self.encoder.save(r'./weights_' + self.weights_path + '/v_encoder_weights.h5')
        # self.decoder.save(r'./weights_' + self.weights_path + '/v_decoder_weights.h5')
        self.v_autoencoder_model.save(r'./weights_' + self.weights_path + '/v_vae_weights.h5')
