"""VRNN model with hidden state sampled from latent distribution conditional on the
input word embedding."""
from __future__ import division

import tensorflow as tf

from utils import KLGaussianStdGaussian, FullyConnected, LatentHiddensVRNNCell, VRNNModel


class LatentHiddensVRNNModel(VRNNModel):

    def __init__(self, args, infer=False):
        self.batch_size = batch_size = args.batch_size
        self.seq_length = seq_length = args.seq_length
        size = args.latent_dimensions
        num_layers = args.num_layers
        vocab_size = args.vocab_size
        
        self._input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
        self._targets = tf.placeholder(tf.int32, [batch_size, seq_length])

        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding',
                                        [vocab_size, size],
                                        dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        inputs = [tf.squeeze(input_step, [1])
                  for input_step in tf.split(1, seq_length, inputs)]

        cell = LatentHiddensVRNNCell(size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        z, last_state = tf.nn.seq2seq.rnn_decoder(inputs,
                self.initial_state,
                cell,
                loop_function=loop if infer else None, scope='rnnlm') 

        z = tf.reshape(tf.concat(1, z), [-1, size])

        logits = tf.matmul(z, softmax_w) + softmax_b
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

