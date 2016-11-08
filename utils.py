import collections
import tensorflow as tf
from tensorflow.python.util import nest

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
    __slots__ = ()

    @property
    def dtype(self):
        (z, z_mean, z_log_sigma_sq) = self

        if not z.dtype == z_mean.dtype and z_mean.dtype == z_log_sigma_sq.dtype:
            raise TypeError('Inconsistent internal state: %s vs %s vs %s' %
                            (str(z.dtype), str(z_mean.dtype), str(z_log_sigma_sq.dtype)))

        return z.dtype


_LatentLSTMVRNNStateTuple = collections.namedtuple('LatentLSTMVRNNStateTuple', ('z', 'lstm'))


class LatentLSTMVRNNStateTuple(_LatentLSTMVRNNStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (z, lstm) = self

        return z.dtype


class LatentHiddensVRNNCell(tf.nn.rnn_cell.RNNCell): 

    def __init__(self, num_units, activation=tf.nn.sigmoid, state_is_tuple=True):
        if not state_is_tuple:
            raise ValueError('LatentHiddensVRNNCell state must be a tuple')

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
        """VRNN Cell."""
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


class LatentLSTMVRNNCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, activation=tf.nn.sigmoid, state_is_tuple=True):
        if not state_is_tuple:
            raise ValueError('LatentLSTMVRNNCell state must be tuple')

        self._num_units = num_units
        self._activation = activation
        self._state_is_tuple = state_is_tuple
        self._lstm = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=state_is_tuple)

    @property
    def state_size(self):
        return LatentLSTMVRNNStateTuple(self._num_units, self._lstm.state_size())

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            z_t, lstm = state


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


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable(
        "Matrix", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size],
        dtype=dtype,
        initializer=tf.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term

