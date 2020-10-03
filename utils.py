import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import io

def batch_generator(data, batch_size, x_tm1):
    idxs = np.random.randint(0, data.shape[0], batch_size)
    X = data[idxs]
    if x_tm1:
        X_tm1 = data[idxs - 1]
        return tf.cast(X, dtype=tf.float32), tf.cast(X_tm1, dtype=tf.float32)
    else:
        return tf.cast(X, dtype=tf.float32)


def sample_image(writer,enc,gen,sup,ori_data,memory_len,step):
    idx = int(np.random.randint(0, ori_data.shape[0], 1))
    X = ori_data[idx]
    X_tm1 = np.concatenate([ori_data[idx - m] for m in range(memory_len)], axis=2)

    X = np.expand_dims(X,axis=0)
    X_tm1 = np.expand_dims(X_tm1, axis=0)
    # Generator
    E_hat = enc(X_tm1)
    H_hat = sup(E_hat)
    H_hat_sup = gen(H_hat)

    def gen_plot():
        """Create a pyplot plot and save to buffer."""
        plt.figure()
        plt.matshow(H_hat_sup[0, :, :, 0])
        buf = io.BytesIO()
        plt.colorbar()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf

    # Prepare the plot
    plot_buf = gen_plot()

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    # Add image summary
    with writer.as_default():
        tf.summary.image('img/' + str(step), image, step)

    plt.close()

def reset_metrics(metrics):
    [m.reset_states() for m in metrics]

def write_loss(writer,loss,loss_name,step):
    if step>100:
        with writer.as_default():
            tf.summary.scalar(name='loss/'+str(loss_name), data=tf.reduce_mean(loss), step=step)

def write_grads(writer,grads,grads_name,step):
    if step>100:
        with writer.as_default():
            tf.summary.histogram(name='grads/'+str(grads_name)+'_fl', data=grads[1], step=step)
            tf.summary.histogram(name='grads/'+str(grads_name)+'_ll', data=grads[-2], step=step)