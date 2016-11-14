from __future__ import print_function

import tensorflow as tf
import numpy as np

from vrnn import VRNNModel
import ptb_reader as reader

from latent_hiddens import LatentHiddensVRNNModel
from latent_fe import LatentFEVRNNModel
from latent_lstm import LatentLSTMVRNNModel

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('--model', 'latent_hiddens', 'VRNNModel to sample from.')
flags.DEFINE_string('--data_dir', '../simple-examples/data', 'Directory containing PTB.')
flags.DEFINE_string('--save_dir', 'save', 'Directory to store checkpointed models.')
flags.DEFINE_integer('--latent_dimensions', 200, 'The size of the RNN hidden state.')
flags.DEFINE_integer('--num_layers', 128, 'The number of layers in the RNN.')
flags.DEFINE_integer('--batch_size', 1, 'Minibatch size.')
flags.DEFINE_integer('--seq_length', 1, 'The number of timesteps to unrol.')
flags.DEFINE_integer('--num_epochs', 50, 'The number of epochs.')
flags.DEFINE_float('--max_grad_norm', 5., 'Max gradient to clip.')
flags.DEFINE_integer('--save_every', 1, 'Save frequency.')
flags.DEFINE_float('--learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_float('--decay_rate', 0.97, 'Decay rate for the learning rate.')
flags.DEFINE_integer('--decay_start', 5, 'When to begin decreasing learning rate.')
flags.DEFINE_integer('--vocab_size', 10000, 'Number of unique tokens in vocabulary.')
flags.DEFINE_float('--init_scale', 0.1, 'Initialisation range.')
flags.DEFINE_string('--prime', 'hello', 'Priming text.')
flags.DEFINE_integer('--n', 50, 'Sample length.')

def main(args):
    if FLAGS.model == 'latent_hiddens':
        model = LatentHiddensVRNNModel
    elif FLAGS.model == 'latent_fe':
        model = LatentFEVRNNModel
    elif FLAGS.model == 'latent_fe_prior':
        model = LatentFEPriorVRNNModel
    elif FLAGS.model == 'latent_lstm':
        model = LatentLSTMVRNNModel
    print('Sampling from %s model' % FLAGS.model)

    with tf.variable_scope('model', reuse=None):
        m = model(args=FLAGS, infer=True)

    token_to_id, id_to_token = reader.two_way_mapping(FLAGS.data_dir)

    prime_text = FLAGS.prime.split()

    with tf.Session() as session:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            
            state = session.run(m.initial_state)
            x = np.zeros((1, 1))
            for token in prime_text[:-1]:
                x[0, 0] = token_to_id[token]
                feed = {m.input_data: x, m.initial_state: state}
                [state] = sess.run([self.final_state], feed)

            def weighted_pick(weights):
                t = np.cumsum(weights)
                s = np.sum(weights)
                return int(np.searchsorted(t, np.random.rand(1) * s))

            ret = ' '.join(prime_text)
            token = prime_text[-1]
            for n in range(FLAGS.n):
                x[0, 0] = token_to_id[token]
                feed = {m.input_data: x, m.initial_state: state}
                [probs, state] = session.run([m.probs, m.final_state], feed)
                p = probs[0]
                
                pred_id = np.argmax(p)
                pred = id_to_token[pred_id]
                if pred == '<eos>':
                    ret += '\n'
                else: 
                    ret += pred + ' '
                token = pred
        else:
            raise ValueError('ckpt or model_checkpoint_path not found')
    
    return ret

if __name__ == '__main__':
    tf.app.run()

