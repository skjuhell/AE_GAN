import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Reshape,BatchNormalization,Dense, Conv2D,LeakyReLU,UpSampling2D,Dropout
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from tcn import TCN,tcn_full_summary
import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.constraints import Constraint

class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

class Encoder(Model):
    def __init__(self,encoding_units,seq_len,memory_len):

        super(Encoder, self).__init__()
        # weight constraint

        # define model
        # downsample to 14x14
        self.conv1 = Conv2D(8, kernel_size=3, strides=(2, 2), padding='same',input_shape=(seq_len, seq_len, memory_len))
        self.bn1 = BatchNormalization(axis=3)
        self.lr1 = LeakyReLU(alpha=0.2)
        # downsample to 7x7
        self.conv2 = Conv2D(8, kernel_size=3, strides=(2, 2), padding='same')
        self.bn2 = BatchNormalization(axis=3)
        self.lr2 = LeakyReLU(alpha=0.2)
        # scoring, linear activation
        self.flatten = Flatten()
        self.dense = Dense(encoding_units*memory_len,activation='tanh')
        self.dropout = Dropout(.2)
        self.reshape1 = Reshape((encoding_units,memory_len))



    @tf.function
    def call(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lr1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lr2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.reshape1(x)


        return x

class Decoder(Model):
    def __init__(self,encoding_units,seq_len,memory_len):

        super(Decoder, self).__init__()
        const = ClipConstraint(0.01)
        self.flatten0 = Flatten()
        self.dense1 = Dense(encoding_units*memory_len)
        self.lr1 = LeakyReLU(alpha=0.2)
        self.re1 = Reshape((int(np.sqrt(encoding_units)), int(np.sqrt(encoding_units)), memory_len))
        self.conv1 = Conv2D(4,kernel_size=3, padding='same',)
        self.lr2 = LeakyReLU(alpha=0.2)
        self.up1 = UpSampling2D()
        self.conv2 = Conv2D(8,kernel_size=3, padding='same',)
        self.lr3 = LeakyReLU(alpha=0.2)
        self.up2 = UpSampling2D()
        self.conv3 = Conv2D(8,kernel_size=3, padding='same',)
        self.lr4 = LeakyReLU(alpha=0.2)
        self.flatten1 = Flatten()
        self.dense2 = Dense(seq_len*seq_len*memory_len,activation='tanh')
        self.dropout = Dropout(.2)
        self.reshape = Reshape((seq_len,seq_len,memory_len))


    @tf.function
    def call(self, x):
        x = self.flatten0(x)
        x = self.dense1(x)
        x = self.lr1(x)
        x = self.re1(x)
        x = self.up1(x)
        x = self.conv1(x)
        x = self.lr2(x)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.lr3(x)
        x = self.up2(x)
        x = self.conv3(x)
        x = self.lr4(x)
        x = self.flatten1(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.reshape(x)
        return(x)

def pretrain_enc_dec(X, encoder, decoder, optim):
    with tf.GradientTape() as tape:
        H = encoder(X)
        X_tilde = decoder(H)
        E_loss_T0 = tf.math.reduce_mean(tf.keras.losses.MSE(X, X_tilde))

    d = tape.gradient(target=[E_loss_T0], sources=encoder.trainable_variables + decoder.trainable_variables)
    optim.apply_gradients(zip(d, encoder.trainable_variables + decoder.trainable_variables))

    return tf.math.sqrt(E_loss_T0), d , X, X_tilde





