"""An application of variational Bayesian techniques to recurrent neural
networks in tensorflow (https://arxiv.org/abs/1312.6114)."""
from __future__ import division

import collections
import tensorflow as tf

from tensorflow.python.ops.rnn_cell import _linear

def KLGaussianStdGaussian(z_mean, z_log_sigma_sq):
    return -0.5 * tf.reduce_sum(1 + tf.square(z_log_sigma_sq)
            - tf.square(z_mean)
            -tf.exp(z_log_sigma_sq), 1)

def FullyConnected(in_tensor, shape, unit='relu', name=None):
    with tf.name_scope(name or 'FullyConnected'):
        if unit == 'relu':
            activation = tf.nn.relu
        elif unit == 'sigmoid':
            activation = tf.nn.sigmoid
        elif unit == 'linear':
            activation = None
        else:
            raise ValueError('unit not recognised: %s' % unit)

        w = tf.Variable(tf.random_normal(shape), name='weights')
        b = tf.Variable(tf.zeros([shape[1]]), name='biases')
        if not activation:
            return tf.matmul(in_tensor, w) + b
        return activation(tf.matmul(in_tensor, w) + b)

_VRNNStateTuple = collections.namedtuple('VRNNStateTuple', ('z', 'z_mean', 'z_log_sigma_sq'))

class VRNNStateTuple(_VRNNStateTuple):
    """A construct that allows the hidden state to be stored alongside the latent means
    and standard deviations that it was sampled from. Increases efficiency over concatenation
    the samples with the means and standard deviations."""
    __slots__ = ()

    @property
    def dtype(self):
        (z, z_mean, z_log_sigma_sq) = self

        if not z.dtype == z_mean.dtype and z_mean.dtype == z_log_sigma_sq.dtype:
            raise TypeError('Inconsistent internal state: %s vs %s vs %s' %
                            (str(z.dtype), str(z_mean.dtype), str(z_log_sigma_sq.dtype)))

        return z.dtype

class BasicVRNNCell(tf.nn.rnn_cell.RNNCell): 

    def __init__(self, num_units, activation=tf.nn.sigmoid, state_is_tuple=True):
        if not state_is_tuple:
            raise ValueError('VRNN State must be a tuple')

        self._num_units = num_units
        self._activation = activation
        self._state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return VRNNStateTuple(self._num_units, self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Variational recurrent neural network cell (VRNN)."""
        with tf.variable_scope(scope or type(self).__name__):
            # Update the hidden state.
            z_t, z_mean_t, z_log_sigma_sq_t = state
            h_t_1 = self._activation(_linear(
                    [inputs, z_t, z_mean_t, z_log_sigma_sq_t],
                    2 * self._num_units,
                    True))
            z_mean_t_1, z_log_sigma_sq_t_1 = tf.split(1, 2, h_t_1)

            # Sample.
            eps = tf.random_normal((tf.shape(inputs)[0], self._num_units), 0.0, 1.0,
                    dtype=tf.float32)
            z_t_1 = tf.add(z_mean_t_1, tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq_t_1)),
                    eps))

            return z_t_1, VRNNStateTuple(z_t_1, z_mean_t_1, z_log_sigma_sq_t_1)

class VRNNModel(object):

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def probs(self):
        return self._probs

    def __init__(self, args):
        self.batch_size = batch_size = args.batch_size
        self.seq_length = seq_length = args.seq_length
        size = args.latent_dimensions
        num_layers = args.num_layers
        vocab_size = args.vocab_size
        
        x_dim = 200
        x2s_dim = 200
        z2s_dim = 200
        p_x_dim = 200

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

        cell = BasicVRNNCell(size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        z, last_state = tf.nn.rnn(cell,
                rnn_inputs,
                initial_state=self._initial_state)

        z = tf.reshape(tf.concat(1, z), [-1, size])
        theta_1 = FullyConnected(z,
                [size, p_x_dim],
                unit='relu',
                name='theta_1')

        theta_2 = FullyConnected(theta_1,
                [p_x_dim, p_x_dim],
                unit='relu',
                name='theta_2')

        theta_3 = FullyConnected(theta_2,
                [p_x_dim, p_x_dim],
                unit='relu',
                name='theta_3')

        theta_4 = FullyConnected(theta_3,
                [p_x_dim, p_x_dim],
                unit='linear',
                name='theta_4')

        logits = FullyConnected(theta_4,
                [p_x_dim, vocab_size],
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

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

