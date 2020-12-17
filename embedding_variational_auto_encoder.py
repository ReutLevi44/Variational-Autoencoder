import tensorflow.keras
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, Lambda
from tensorflow.keras.models import Model
import numpy as np
import os
import tensorflow.keras.backend as K
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.regularizers import l2


class EmbeddingVariationalAutoEncoder:
    def __init__(self, input_set, D, weights_path, encoding_dim=3, batch_size=50, emb_alpha=0.1, mu=None, log_sigma=None):
        self.encoding_dim = encoding_dim
        self.batch_size = batch_size
        self.x = input_set
        self.input_shape = len(input_set[0])
        self.D = D
        self.weights_path = weights_path
        # embedding
        self.emb_alpha = emb_alpha
        self.mu = mu
        self.log_sigma = log_sigma
        print(self.x)

    def sample_z(self, args):
        mu, log_sigma = args
        eps = K.random_normal(shape=(K.shape(mu)[0], self.encoding_dim), mean=0.)
        return mu + K.exp(log_sigma / 2) * eps

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

        # create the model using our input (cdr3 sequences set) and
        # two separate outputs -- one for the reconstruction of the
        # data and another for the representations, respectively
        model = Model(inputs=inputs, outputs=[reshape2, mu])

        self.model = model
        print(model.summary())

        return model

    # 1. y_true 2. y_pred
    def emb_loss(self, D, ec_out):
        emb = 0
        for i in range(self.batch_size):
            for j in range(self.batch_size):
                if i < j:
                    # norm 2
                    dis_z = K.sqrt(K.sum(K.square(ec_out[i] - ec_out[j])))
                    emb += K.square(dis_z - D[i][j])
        emb = self.emb_alpha * emb
        return emb

    def vae_loss(self, y_true, y_pred):
        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
        # E[log P(X|z)]
        m = K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl = 0.5 * K.sum(K.exp(self.log_sigma) + K.square(self.mu) - 1. - self.log_sigma, axis=1)
        return 5000 * m + kl

    def generator(self, X, D, batch):
        # generate: A. batches of x B. distances matrix within batch
        while True:
            inds = []
            for i in range(batch):
                # choose random index in features
                index = np.random.choice(X.shape[0], 1)[0]
                if i == 0:
                    batch_X = np.array([X[index]])
                else:
                    batch_X = np.concatenate((batch_X, np.array([X[index]])), axis=0)
                inds.append(index)
            tmp = D[np.array(inds)]
            batch_D = tmp[:, np.array(inds)]
            # 1. training data-features 2. target data-labels
            yield batch_X, [batch_X, batch_D]

    def fit_generator(self, epochs=300):
        # self.model = multi_gpu_model(self.model, gpus=3)
        adam = tensorflow.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(optimizer=adam, loss=[self.vae_loss, self.emb_loss], experimental_run_tf_function=False)
        results = self.model.fit_generator(self.generator(self.x, self.D, self.batch_size),
                                           steps_per_epoch=self.x.shape[0] / self.batch_size,
                                           epochs=epochs, verbose=2)
        return results

    # def fit_v_autoencoder(self, epochs=300):
    #     # self.model = multi_gpu_model(self.model, gpus=3)
    #     adam = tensorflow.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #     self.model.compile(optimizer=adam, loss=['mse', self.vae_loss], metrics=['mae'])
    #     log_dir = './log/'
    #     tb_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
    #                                                          write_graph=True, write_images=True)
    #     es_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
    #                                                            patience=20, verbose=0, mode='auto')
    #     self.model.fit(x=self.x, y=[self.x, self.D], validation_split=0.2, verbose=2,
    #                    epochs=epochs, batch_size=self.batch_size,
    #                    callbacks=[tb_callback, es_callback])

    def save_vae(self):
        if not os.path.exists(r'./weights_' + self.weights_path):
            os.mkdir(r'./weights_' + self.weights_path)
        self.encoder.save(r'./weights_' + self.weights_path + '/embedding_encoder_weights.h5')
        # self.decoder.save(r'./weights_' + self.weights_path + '/embedding_decoder_weights.h5')
        self.model.save(r'./weights_' + self.weights_path + '/embedding_vae_weights.h5')
