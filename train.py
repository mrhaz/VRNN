from __future__ import division

import tensorflow as tf
import numpy as np
import collections
import time
import os

from latent_hiddens import LatentHiddensVRNNModel
from latent_fe import LatentFEVRNNModel
from latent_lstm import LatentLSTMVRNNModel
import ptb_reader as reader

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('--model', 'latent_hiddens', 'VRNN Model to train.')
flags.DEFINE_string('--data_dir', '../simple-examples/data', 'Directory containing PTB.')
flags.DEFINE_string('--save_dir', 'save', 'Directory to store checkpointed models.')
flags.DEFINE_integer('--latent_dimensions', 200, 'The size of the RNN hidden state.')
flags.DEFINE_integer('--num_layers', 128, 'The number of layers in the RNN.')
flags.DEFINE_integer('--batch_size', 128, 'Minibatch size.')
flags.DEFINE_integer('--seq_length', 50, 'The number of timesteps to unrol.')
flags.DEFINE_integer('--num_epochs', 50, 'The number of epochs.')
flags.DEFINE_float('--max_grad_norm', 5., 'Max gradient to clip.')
flags.DEFINE_integer('--save_every', 1, 'Save frequency.')
flags.DEFINE_float('--learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_float('--decay_rate', 0.97, 'Decay rate for the learning rate.')
flags.DEFINE_integer('--decay_start', 5, 'When to begin decreasing learning rate.')
flags.DEFINE_integer('--vocab_size', 10000, 'Number of unique tokens in vocabulary.')
flags.DEFINE_float('--init_scale', 0.1, 'Initialisation range.')


def run_epoch(session, model, data, eval_op=None, verbose=False):
    epoch_size = ((len(data) // model.batch_size) - 1) // model.seq_length
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size,
                                                      model.seq_lengthh)):
        fetches = [model.cost, model.final_state, eval_op]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y

        for i, (z, z_mean, z_log_sigma_sq) in enumerate(model.initial_state):
            feed_dict[z] = state.z
            feed_dict[z_mean] = state.z_mean
            feed_dict[z_log_sigma_sq] = state.z_log_sigma_sq

        cost, state, _ = session.run(fetches, feed_dict)
        costs += cost
        iters += model.seq_length

        if verbose and step % (epoch_size // 10) == 10:
            print('Progress: %.3f; Perplexity: %.3f; Speed: %.0f wps'
                  % (step * 1.0 / epoch_size,
                     np.exp(costs / iters),
                     iters * model.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def main(args):
    if FLAGS.model == 'latent_hiddens':
        model = LatentHiddensVRNNModel
    elif FLAGS.model == 'latent_fe':
        model = LatentFEVRNNModel
    elif FLAGS.model == 'latent_fe_prior':
        model = LatentFEPriorVRNNModel
    elif FLAGS.model = 'latent_lstm':
        model = LatentLSTMVRNNModel
    print('training %s model', FLAGS.model)

    raw_data = reader.ptb_raw_data(FLAGS.data_dir)
    train_data, valid_data, test_data, _ = raw_data

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale,
                                                    FLAGS.init_scale)

        with tf.variable_scope('model', reuse=None, initializer=initializer):
            m = model(args=FLAGS)
        with tf.variable_scope('model', reuse=True, initializer=initializer):
            mvalid = model(args=FLAGS)
            mtest = model(args=FLAGS)

        tf.initialize_all_variables().run()

        saver = tf.train.Saver(tf.all_variables())

        for i in range(FLAGS.num_epochs):
            lr_decay = FLAGS.decay_rate ** max(i - FLAGS.decay_start, 0.0)
            m.assign_lr(session, FLAGS.learning_rate * lr_decay)

            print('Epoch: %d Learning rate: %.3f' % (i + 1, session.run(m.lr)))    
            train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                         verbose=True)
            print('Epoch: %d Train Perplexity: %.3f' % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
            print('Epoch: %d Valid Perplexity: %.3f' % (i + 1, valid_perplexity))
            
            checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
            saver.save(session, checkpoint_path, global_step=i + 1)
            print('Model saved to {}'.format(checkpoint_path))
        
        test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
        print('Test perplexity: %.3f' % test_perplexity)


if __name__ == '__main__':
    tf.app.run()

