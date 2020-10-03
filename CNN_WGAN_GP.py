import tensorflow as tf
from recurrence_plots import rec_plot
import pandas as pd
import numpy as np
from cnn_encoder import Encoder, Decoder, pretrain_enc_dec
from supervisor import Supervisor, train_sup_gen
from generator import Generator, gen_train_step
from discriminator import Discriminator, disc_train_step
import datetime
from utils import batch_generator, write_loss, write_grads, sample_image
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')
stock_data = pd.read_csv('stock_data.csv').values
stock_data = stock_data[:,3]

stock_data = 2*(stock_data-np.min(stock_data))/(np.max(stock_data)-np.min(stock_data))-1
print(np.min(stock_data),np.max(stock_data))

seq_len = 24
encoding_units = 9

data = []
for idx in range(stock_data.shape[0]-seq_len):
    data.append(rec_plot(stock_data[idx:idx+seq_len]))
data = np.expand_dims(np.array(data, dtype='float64'), axis=3)
dim = data.shape[-1]



batch_size = 128
memory_len = 5
encoder = Encoder(encoding_units, seq_len, memory_len)
decoder = Decoder(encoding_units, seq_len, memory_len)
supervisor = Supervisor(encoding_units, memory_len, dim)
generator = Generator(encoding_units,seq_len,dim)
discriminator = Discriminator(encoding_units,dim)

enc_dec_optim = tf.keras.optimizers.Adam()
sup_gen_optim = tf.keras.optimizers.Adam()
gen_optim = tf.keras.optimizers.RMSprop(lr=0.00005)
disc_optim = tf.keras.optimizers.RMSprop(lr=0.00005)


epochs_enc_dec = 500
epochs_sup = 500
epochs_joint = 500


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/loss'
train_loss_writer = tf.summary.create_file_writer(train_log_dir)
image_writer = tf.summary.create_file_writer('logs/gradient_tape/' + current_time + '/images')
print('Train Encoder/Decoder')

for e in range(epochs_enc_dec):
    idx = np.random.randint(0,data.shape[0],batch_size)
    X = np.concatenate([data[idx - m] for m in range(memory_len)], axis=3)
    E_loss, d, X, X_tilde = pretrain_enc_dec(X=X, encoder=encoder, decoder=decoder, optim=enc_dec_optim)
    write_loss(train_loss_writer, np.mean(E_loss), 'E_loss', e)
    if (e%10)==0:
        print(e,np.round(np.mean(E_loss),4))

'''    if e==(epochs_enc_dec-1):
        fig, axes = plt.subplots(nrows=1, ncols=2)
        ax1,ax2 = axes
        ax1.matshow(X_tilde[0])
        ax2.matshow(X[0])
        plt.show()'''

print('Train Supervisor/Generator')
for e in range(epochs_sup):
    idx = np.random.randint(memory_len+1, data.shape[0],batch_size)
    X_tm1 = np.concatenate([data[idx - m] for m in range(memory_len)], axis=3)

    G_loss_S = train_sup_gen(X=data[idx], X_tm1=X_tm1, encoder=encoder, supervisor=supervisor, generator=generator, optim=sup_gen_optim)
    write_loss(train_loss_writer, np.mean(G_loss_S), 'G_loss_S', e)
    if (e%10)==0:
        print(e,np.round(np.mean(G_loss_S)*1000,5))

print('Train Generator/Discriminator')
for e in range(epochs_joint):
    idx = np.random.randint(0, data.shape[0], batch_size)
    X_tm1 = np.concatenate([data[idx - m] for m in range(memory_len)], axis=3)

    #Train Generator
    G_loss = gen_train_step(X_tm1=X_tm1, encoder=encoder, discriminator=discriminator, supervisor=supervisor, generator=generator, optim=gen_optim)
    write_loss(train_loss_writer, np.mean(G_loss), 'G_loss', e)

    #G_loss_S = train_sup_gen(X=data[idx], X_tm1=X_tm1, encoder=encoder, supervisor=supervisor, generator=generator, optim=sup_gen_optim)
    #write_loss(train_loss_writer, np.mean(G_loss_S), 'G_loss_S', e+epochs_sup)

    #Train Discriminator

    for _ in range(20):
        idx = np.random.randint(0, data.shape[0], batch_size)
        X_tm1 = np.concatenate([data[idx - m] for m in range(memory_len)], axis=3)
        D_loss = disc_train_step(X=data[idx], X_tm1=X_tm1, encoder=encoder,supervisor=supervisor, generator=generator, discriminator=discriminator, optim=disc_optim)
    write_loss(train_loss_writer, np.mean(D_loss), 'D_loss', e)

    if (e%10)==0:
        print(e, 'G_loss ', np.round(np.mean(G_loss),4),'D_loss ',np.round(np.mean(D_loss),4))
    if (e % 20) == 0:
        sample_image(writer=image_writer, enc=encoder,sup=supervisor, gen=generator, ori_data=data, memory_len=memory_len, step=e)


'''encoder.save('encoder')
decoder.save('decoder')
generator.save('generator')
supervisor.save('supervisor')
'''

idx = np.random.randint(0, data.shape[0], 1000)
X = data[idx]
X_tm1 = np.concatenate([data[idx - m] for m in range(memory_len)], axis=3)
H = encoder(X_tm1)
S = supervisor(H)
G = generator(S)
np.save('gen_data.npy', G)
np.save('input_data', X_tm1)


idx = np.random.randint(0, data.shape[0], 1)

X = data[idx]
X_tm1 = np.concatenate([data[idx - m] for m in range(memory_len)], axis=3)
H = encoder(X_tm1)
S = supervisor(H)

for i in range(100):
    G = generator(S)
    np.save('gen_data_'+str(i)+'_sim.npy',G)

np.save('X_in_sim.npy',X)

