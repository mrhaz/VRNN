import tensorflow as tf

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


class LatentHiddensVRNNCell(tf.nn.rnn_cell.RNNCell): 

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

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

