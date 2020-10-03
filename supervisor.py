import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Reshape,BatchNormalization,Dense, Conv2D,LeakyReLU,UpSampling2D
from tensorflow.keras import Model
import matplotlib.pyplot as plt

from tensorflow.keras import backend
from tensorflow.keras.constraints import Constraint


class Supervisor(Model):
    def __init__(self,encoding_units,memory_len,dim):

        super(Supervisor, self).__init__()
        self.lstm1 = LSTM(10,return_sequences=False)
        self.dense = Dense(encoding_units*dim,activation='tanh')
        self.reshape1 = Reshape((encoding_units,dim))


    @tf.function
    def call(self,x):
        x = self.lstm1(x)
        x = self.dense(x)
        x = self.reshape1(x)
        return x

def train_sup_gen(X, X_tm1, encoder, supervisor, generator, optim):
    with tf.GradientTape() as tape:

        H_tm1 = encoder(X_tm1)
        H_hat = supervisor(H_tm1)
        X_hat = generator(H_hat)

        G_loss_S = tf.keras.losses.MSE(X_hat,X)

    d = tape.gradient(target=[G_loss_S], sources=generator.trainable_variables + supervisor.trainable_variables)
    optim.apply_gradients(zip(d, generator.trainable_variables + supervisor.trainable_variables))

    return G_loss_S