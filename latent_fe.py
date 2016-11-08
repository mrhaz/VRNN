"""VRNN model with hidden state sampled from latent distribution conditional on features
extracted from input word embedding."""
from __future__ import division

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import _linear
from utils import KLGaussianStdGaussian, FullyConnected, LatentHiddensVRNNCell, VRNNModel


class LatentFEVRNNModel(VRNNModel):

    def __init__(self, args):
        self.batch_size = batch_size = args.batch_size
        self.seq_length = seq_length = args.seq_length
        size = args.latent_dimensions
        num_layers = args.num_layers
        vocab_size = args.vocab_size
        
        x_dim = 200
        x2s_dim = 200

        self._input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
        self._targets = tf.placeholder(tf.int32, [batch_size, seq_length])

        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding',
                                        [vocab_size, x_dim],
                                        dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        inputs = [tf.squeeze(input_step, [1])
                  for input_step in tf.split(1, seq_length, inputs)]
        inputs = tf.reshape(inputs, [-1, seq_length * x_dim])

        phi_1 = FullyConnected(inputs,
                [x_dim * seq_length, x2s_dim],
                unit='relu',
                name='phi_1')
        
        phi_2 = FullyConnected(phi_1,
                [x2s_dim, x2s_dim],
                unit='relu',
                name='phi_2')
        
        phi_3 = FullyConnected(phi_2,
                [x2s_dim, x2s_dim],
                unit='relu',
                name='phi_3')

        phi_4 = FullyConnected(phi_3,
                [x2s_dim, x2s_dim],
                unit='relu',
                name='phi_4')

        rnn_inputs = \
                [tf.squeeze(tf.reshape(input_step,
                    [batch_size, 1, x2s_dim // seq_length]),[1])
                    for input_step in tf.split(1, seq_length, phi_4)]

        cell = LatentHiddensVRNNCell(size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        z, last_state = tf.nn.rnn(cell,
                rnn_inputs,
                initial_state=self._initial_state)

        z = tf.reshape(tf.concat(1, z), [-1, size])

        logits = FullyConnected(z,
                [size, vocab_size],
                unit='linear',
                name='logits')
        self._probs = tf.nn.softmax(logits)

        recon_loss = tf.nn.seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(self._targets, [-1])],
                [tf.ones([batch_size * seq_length], dtype=tf.float32)],
                vocab_size)
        kl_loss = [KLGaussianStdGaussian(z_mean, z_log_sigma_sq)
                for _, z_mean, z_log_sigma_sq in last_state]

        self._cost = tf.reduce_mean(kl_loss) \
                + tf.reduce_sum(recon_loss) / batch_size / seq_length

        self._final_state = last_state

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          args.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(tf.float32, shape=[],
                                      name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr)

