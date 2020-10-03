import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Reshape,BatchNormalization,Dense, Conv2D,LeakyReLU,UpSampling2D,GaussianNoise
from tensorflow.keras import Model
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


class Generator(Model):
    def __init__(self,encoding_units,seq_len,dim):

        super(Generator, self).__init__()
        const = ClipConstraint(0.01)

        self.flatten0 = Flatten()
        self.dense1 = Dense(encoding_units*dim)
        self.noise1 = GaussianNoise(.2)
        self.lr = LeakyReLU(alpha=0.2)
        self.re1 = Reshape((3, 3, dim))
        self.conv1 = Conv2D(4,kernel_size=3, padding='same')
        self.lr = LeakyReLU(alpha=0.2)
        self.up1 = UpSampling2D()
        self.conv2 = Conv2D(8,kernel_size=3, padding='same')
        self.lr = LeakyReLU(alpha=0.2)
        self.up2 = UpSampling2D()
        self.conv3 = Conv2D(8,kernel_size=3, padding='same')
        self.lr = LeakyReLU(alpha=0.2)
        self.flatten1 = Flatten()

        self.dense2 = Dense(seq_len*seq_len*dim,activation='tanh')
        self.noise2 = GaussianNoise(.2)
        self.reshape1 = Reshape((seq_len,seq_len,dim))

    @tf.function
    def call(self, x):
        x = self.flatten0(x)
        x = self.dense1(x+tf.random.normal(shape=x.shape,mean=0,stddev=.05,dtype=tf.float64))
        x = self.noise1(x)
        x = self.lr(x)
        x = self.re1(x)
        x = self.up1(x)
        x = self.conv1(x)
        x = self.lr(x)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.lr(x)
        x = self.up2(x)
        x = self.conv3(x)
        x = self.lr(x)
        x = self.flatten1(x)
        x = self.dense2(x+tf.random.normal(shape=x.shape,mean=0,stddev=.05,dtype=tf.float64))
        x = self.reshape1(x)

        return(x)

def gen_train_step(X_tm1,encoder,supervisor,generator,discriminator,optim):
    for layer in generator.layers:
        layer.trainable = True

    with tf.GradientTape() as tape:
        H_tm1 = encoder(X_tm1)
        H_t_hat = supervisor(H_tm1)
        X_hat = generator(H_t_hat)
        y_fake = discriminator(X_hat)

        # Generator Loss = -[average critic score on fake images]
        # G_loss = -tf.reduce_mean(y_fake)
        #G_loss = tf.keras.losses.binary_crossentropy(y_fake,tf.ones_like(y_fake))
        G_loss = -tf.math.reduce_mean(y_fake)

    d = tape.gradient(target=[G_loss], sources=generator.trainable_variables + supervisor.trainable_variables)
    optim.apply_gradients(zip(d, generator.trainable_variables + supervisor.trainable_variables))

    return G_loss



