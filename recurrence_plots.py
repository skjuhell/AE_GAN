import numpy as np

def rec_plot(x):
    dim = x.shape[0]
    rec_mat = np.zeros(shape=(dim,dim))
    for i in range(dim):
        for j in range(dim):
            rec_mat[i, j] = x[i]-x[j]
            rec_mat[j, i] = x[i] - x[j]


    return rec_mat



