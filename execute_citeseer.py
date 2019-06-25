import time
from models import GAT
import tensorflow as tf

import execute_

dataset = 'pubmed'

checkpt_file = 'pre_trained/{}/mod_{}.ckpt'.format(dataset, dataset)

# todo fit the right hyperparamters to the method
# training params
batch_size = 1
nb_epochs = 100000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT
nhood = 1

execute_.run_gat(dataset=dataset, batch_size=batch_size, nb_epochs=nb_epochs, patience=patience, lr=lr, l2_coef=l2_coef,
                 hid_units=hid_units, n_heads=n_heads, residual=residual, nonlinearity=nonlinearity, model=model,
                 checkpt_file=checkpt_file, nhood=nhood)
