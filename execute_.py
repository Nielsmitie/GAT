import numpy as np
import tensorflow as tf
from utils import process
from utils import process_ppi
import os
import time

import pandas as pd


tracking_params = ["dataset", "lr", "l2_coef", "hid_units", "n_heads", "residual", "nonlinearity", "param_attn_drop",
                   "param_ffd_drop"]

result_cols = ["training_epochs", "elapsed_time", "min_validation_loss", "max_val_accuracy",
               "test_loss", "test_accuracy"]


def run_gat(dataset, batch_size, nb_epochs,
            patience, lr, l2_coef, hid_units, n_heads, residual, nonlinearity, model, checkpt_file, nhood, param_attn_drop=0.6,
            param_ffd_drop=0.6, sparse=False):
    # necessary work around to run on GPU
    # '''
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    # '''

    # redirect output to file
    import sys

    orig_stdout = sys.stdout
    if os.path.isfile(os.path.dirname(checkpt_file) + 'out.txt'):
        f = open(os.path.dirname(checkpt_file) + 'out.txt', 'a')
        sys.stdout = f
        print('\n\n\n\n')
    else:
        f = open(os.path.dirname(checkpt_file) + 'out.txt', 'w')
        sys.stdout = f

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

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
    features, spars = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = y_train.shape[1]

    features = features[np.newaxis]
    # adj = adj[np.newaxis]
    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    val_mask = val_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]

    if sparse:
        biases = process.preprocess_adj_bias(adj)
    else:
        adj = adj.todense()
        adj = adj[np.newaxis]
        biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

    # adj = adj.todense()
    # biases = process.adj_to_bias(adj, [nb_nodes], nhood=nhood)

    with tf.Graph().as_default():
        with tf.name_scope('input'):
            ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
            if sparse:
                # bias_idx = tf.placeholder(tf.int64)
                # bias_val = tf.placeholder(tf.float32)
                # bias_shape = tf.placeholder(tf.int64)
                bias_in = tf.sparse_placeholder(dtype=tf.float32)
            else:
                bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))

            # bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
            lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
            msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
            attn_drop = tf.placeholder(dtype=tf.float32, shape=())
            ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
            is_train = tf.placeholder(dtype=tf.bool, shape=())

        logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                    attn_drop, ffd_drop,
                                    bias_mat=bias_in,
                                    hid_units=hid_units, n_heads=n_heads,
                                    residual=residual, activation=nonlinearity)
        log_resh = tf.reshape(logits, [-1, nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])
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
                tr_size = features.shape[0]
                # todo paper says only 20 nodes per class are used for training, but all steps use the same data?!
                while tr_step * batch_size < tr_size:
                    if sparse:
                        bbias = biases
                    else:
                        bbias = biases[tr_step * batch_size:(tr_step + 1) * batch_size]

                    _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                        feed_dict={
                            ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                            bias_in: bbias,
                            lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                            msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                            is_train: True,
                            attn_drop: param_attn_drop, ffd_drop: param_ffd_drop})
                    train_loss_avg += loss_value_tr
                    train_acc_avg += acc_tr
                    tr_step += 1

                vl_step = 0
                vl_size = features.shape[0]

                while vl_step * batch_size < vl_size:
                    if sparse:
                        bbias = biases
                    else:
                        bbias = biases[vl_step * batch_size:(vl_step + 1) * batch_size]

                    loss_value_vl, acc_vl = sess.run([loss, accuracy],
                        feed_dict={
                            ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                            bias_in: bbias,
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

            ts_size = features.shape[0]
            ts_step = 0
            ts_loss = 0.0
            ts_acc = 0.0

            while ts_step * batch_size < ts_size:
                if sparse:
                    bbias = biases
                else:
                    bbias = biases[ts_step * batch_size:(ts_step + 1) * batch_size]
                loss_value_ts, acc_ts = sess.run([loss, accuracy],
                    feed_dict={
                        ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                        bias_in: bbias,
                        lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                        msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                ts_loss += loss_value_ts
                ts_acc += acc_ts
                ts_step += 1

            print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step, ' at epoch: ', epoch, ' elapsed time', time.time() - start)

            # log information about the training
            if os.path.isfile(os.path.dirname(checkpt_file) + 'log.csv'):
                print('loading existing log')
                df = pd.read_csv(os.path.dirname(checkpt_file) + 'log.csv', index_col=['run'])
                print('log: ' + str(df))
            else:
                print('Creating new log')
                df = pd.DataFrame(columns=tracking_params+result_cols)

            log = dict(zip(tracking_params + result_cols,
                           [dataset, lr, l2_coef, hid_units, n_heads, residual,
                            str(nonlinearity).split(' ')[1],
                            param_attn_drop, param_ffd_drop] +
                           [epoch, time.time() - start, vlss_mn, vacc_mx, ts_loss/ts_step, ts_acc/ts_step]))

            print('Adding entry: ' + str(log))

            df = df.append(log, ignore_index=True)
            print('saving logs')
            df.to_csv(os.path.dirname(checkpt_file) + 'log.csv', index_label='run')
            print('log save succesfull')

            sess.close()
    sys.stdout = orig_stdout
    f.close()
