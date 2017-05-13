#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tolstoy_reader

def get_default_gpu_session(fraction=0.333):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    return tf.Session(config=config)

def build_lstm_discrete_prediction_model(shape):
    # shape is dict with keys:
    # n_steps_per_batch, n_unique_ids, n_hidden_dim
    with tf.Graph().as_default() as g:
        X = tf.placeholder(tf.int64, [None, shape['n_steps_per_batch']])
        y = tf.placeholder(tf.int64, [None])

        with tf.variable_scope('weights'):
            h_0 = tf.get_variable('h_0', [1,shape['n_hidden_dim']])
            c_0 = tf.get_variable('c_0', [1,shape['n_hidden_dim']])
            w_i = tf.get_variable('w_i', [1, shape['n_hidden_dim']])
            w_c = tf.get_variable('w_c', [1, shape['n_hidden_dim']])
            w_f = tf.get_variable('w_f', [1, shape['n_hidden_dim']])
            w_o = tf.get_variable('w_o', [1, shape['n_hidden_dim']])
            w_yh = tf.get_variable('w_yh', [shape['n_hidden_dim'], shape['n_unique_ids']])
            v_o = tf.get_variable('v_o', [shape['n_hidden_dim'], shape['n_hidden_dim']])
            u_i = tf.get_variable('u_i', [shape['n_hidden_dim'], shape['n_hidden_dim']])
            u_c = tf.get_variable('u_c', [shape['n_hidden_dim'], shape['n_hidden_dim']])
            u_f = tf.get_variable('u_f', [shape['n_hidden_dim'], shape['n_hidden_dim']])
            u_o = tf.get_variable('u_o', [shape['n_hidden_dim'], shape['n_hidden_dim']])

        h_pre = h_0
        c_pre = c_0
        for t in range(shape['n_steps_per_batch']):
            x_t = tf.expand_dims(tf.cast(X[:,t],'float32'),1)
            i_t = tf.sigmoid(tf.matmul(x_t, w_i) + tf.matmul(h_pre, u_i ))
            c_bar_t = tf.tanh(tf.matmul(x_t, w_c) + tf.matmul(h_pre, u_c))
            f_t = tf.sigmoid(tf.matmul(x_t, w_f) + tf.matmul(h_pre, u_f))
            c_t = tf.multiply(i_t, c_bar_t) + tf.multiply(c_pre, f_t)
            o_t = tf.sigmoid(tf.matmul(x_t, w_o) + tf.matmul(h_pre, u_o) + tf.matmul(c_t, v_o))
            h_t = tf.multiply(o_t, tf.tanh(c_t))
            h_pre = h_t
            c_pre = c_t
        logits = tf.matmul(h_t, w_yh)
        y_onehot = tf.one_hot(indices = y, depth = shape['n_unique_ids'])
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
        pred = tf.argmax(logits, axis=1)
        train_op = tf.train.AdamOptimizer().minimize(loss)

    return {'inputs': [X, y], 'loss': loss, 'train_op': train_op,'graph': g, 'pred': pred}

def run_tolstoy_train(n_hid):
    # generate data
    btg, map_dict, backmap_dict = \
        tolstoy_reader.batch_tolstoy_generator(batch_size=200, seq_size=100)
    shape = dict(n_steps_per_batch=100, n_unique_ids=len(map_dict), n_hidden_dim=n_hid)
    model = build_lstm_discrete_prediction_model(shape)

    max_iter_i = 0
    with model['graph'].as_default() as g, get_default_gpu_session(0.9) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(10):
            for iter_i, data_batch in enumerate(btg):
                max_iter_i = max(iter_i, max_iter_i)
                global_step = epoch_i*max_iter_i+iter_i

                # run training step
                train_feed_dict = dict(zip(model['inputs'], data_batch))
                to_compute = [model['train_op'], model['loss']]
                _, loss_val = sess.run(to_compute, train_feed_dict)
                # test generation
                pred_length = 50
                data_input = next(iter(btg))[0][[0]]
                original_sample = data_input.copy()
                pred_seq = []
                for _ in range(pred_length):
                    pred = sess.run(model['pred'], {model['inputs'][0]: data_input})
                    pred_seq.append(pred[0])
                    data_input = np.roll(data_input, -1, axis=1)
                    data_input[0, -1] = pred[0]
                if iter_i % 100 == 0:
                    print(loss_val)
                    print('[%d] Input text:' % (iter_i))
                    print(''.join([backmap_dict[x] for x in original_sample[0]]))
                    print('[%d] Generated continuation:' % (iter_i))
                    print(''.join([backmap_dict[x] for x in pred_seq]))
                    print(pred_seq)
                    print()


hidden = 50
run_tolstoy_train(hidden)
