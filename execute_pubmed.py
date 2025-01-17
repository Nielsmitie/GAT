import tensorflow as tf

import execute_
from models import SpGAT

dataset = 'pubmed'

checkpt_file = 'pre_trained/{}/mod_{}.ckpt'.format(dataset, dataset)

# training params
batch_size = 1
nb_epochs = 100000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.001  # weight decay # changed in comparison to cora and citeseer
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 8] # additional entry for the output layer # changed in comparison to cora and citeseer
residual = False
nonlinearity = tf.nn.elu
model = SpGAT
nhood = 1

execute_.run_gat(dataset=dataset, batch_size=batch_size, nb_epochs=nb_epochs, patience=patience, lr=lr, l2_coef=l2_coef,
                 hid_units=hid_units, n_heads=n_heads, residual=residual, nonlinearity=nonlinearity, model=model,
                 checkpt_file=checkpt_file, nhood=nhood, sparse=True)
