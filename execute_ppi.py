import numpy as np
import tensorflow as tf
from utils import process
from utils import process_ppi
import os
import time
from models import GAT

import pandas as pd


batch_size = 2
nb_epochs = 1
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT
nhood = 1
param_attn_drop = 0.6
param_ffd_drop = 0.6

dataset = 'ppi'

checkpt_file = 'pre_trained/{}/mod_{}.ckpt'.format(dataset, dataset)

print('Dataset: ' + dataset)
print('batch_size: ' + str(batch_size))
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))
print('nhood: ' + str(nhood))
print('attn_drop: ' + str(param_attn_drop))
print('ffd_drop: ' + str(param_ffd_drop))


train_adj, val_adj, test_adj, train_feat,\
val_feat, test_feat, train_labels, val_labels,\
test_labels, train_nodes, val_nodes, test_nodes,\
tr_msk, vl_msk, ts_msk = process_ppi.process_p2p()

y_train = train_labels
y_val = val_labels
y_test = test_labels
train_mask = tr_msk
val_mask = vl_msk
test_mask = ts_msk

nb_nodes = train_feat.shape[1]
ft_size = train_feat.shape[2]
nb_classes = y_train.shape[2]


with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size), name='features')
        bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes), name='bias')
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes), name='label')
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes), name='mask')
        attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attention_drop')
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='feature_drop')
        is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')

    logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)
    log_resh = tf.reshape(logits, [-1, nb_classes], name='log_resh')
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes], name='lab_resh')
    msk_resh = tf.reshape(msk_in, [-1], name='mask_resh')
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    start = time.time()

    with tf.Session() as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = train_feat.shape[0]
            # todo paper says only 20 nodes per class are used for training, but all steps use the same data?!
            while tr_step * batch_size < tr_size:
                _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                    feed_dict={
                        ftr_in: train_feat[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_in: train_adj[tr_step * batch_size:(tr_step + 1) * batch_size],
                        lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: True,
                        attn_drop: param_attn_drop, ffd_drop: param_ffd_drop})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = val_feat.shape[0]
            while vl_step * batch_size < vl_size:

                loss_value_vl, acc_vl = sess.run([loss, accuracy],
                    feed_dict={
                        ftr_in: val_feat[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_in: val_adj[vl_step * batch_size:(vl_step + 1) * batch_size],
                        lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                    (train_loss_avg/tr_step, train_acc_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step))

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)

        ts_size = test_feat.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            loss_value_ts, acc_ts = sess.run([loss, accuracy],
                feed_dict={
                    ftr_in: test_feat[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: test_adj[ts_step * batch_size:(ts_step + 1) * batch_size],
                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

            print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step, ' at epoch: ', epoch, ' elapsed time', time.time() - start)