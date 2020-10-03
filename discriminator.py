import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Reshape,BatchNormalization,Dense, Conv2D,LeakyReLU,UpSampling2D
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from tcn import TCN,tcn_full_summary
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

class Discriminator(Model):
    def __init__(self,seq_len,dim):

        super(Discriminator, self).__init__()

        # weight constraint
        const = ClipConstraint(0.01)
        # define model
        # downsample to 14x14
        self.conv1 = Conv2D(64, kernel_size=3, strides=(2, 2), padding='same',kernel_constraint=const, input_shape=(seq_len, seq_len,dim))
        self.bn1 = BatchNormalization()
        self.lr1 = LeakyReLU(alpha=0.2)
        # downsample to 7x7
        self.conv2 = Conv2D(64, kernel_size=3, strides=(2, 2),kernel_constraint=const ,padding='same')
        self.bn2 = BatchNormalization()
        self.lr2 = LeakyReLU(alpha=0.2)
        # scoring, linear activation
        self.flatten = Flatten()
        self.dense = Dense(1,activation='sigmoid',kernel_constraint=const)

    @tf.function
    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lr1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lr2(x)
        x = self.flatten(x)
        x = self.dense(x)


        return x

def disc_train_step(X,X_tm1,encoder,supervisor,generator,discriminator,optim):
    with tf.GradientTape() as tape:
        H_tm1 = encoder(X_tm1)
        H_t_hat = supervisor(H_tm1)
        X_hat = generator(H_t_hat)

        y_fake = discriminator(X_hat)
        y_real = discriminator(X)
        #Critic Loss = [average critic score on real images] – [average critic score on fake images]
        # -1 =fake label, 1 is real label
        #D_loss = tf.reduce_mean(y_real)-tf.reduce_mean(y_fake)
        #D_loss = tf.keras.losses.binary_crossentropy(y_real,tf.ones_like(y_real))

        D_loss = -(tf.math.reduce_mean(y_real) - tf.math.reduce_mean(y_fake))
        # generator loss

    d = tape.gradient(target=D_loss, sources=discriminator.trainable_variables)
    optim.apply_gradients(zip(d, discriminator.trainable_variables))

    return D_loss
