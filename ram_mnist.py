"""Recurrent Models of Visual Attention V. Mnih et al."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np

from glimpse import GlimpseNet, LocNet
from utils import weight_variable, bias_variable, loglikelihood
from config import Config
import os
from tensorflow.examples.tutorials.mnist import input_data

gpus = [1]
os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(i) for i in gpus])

logging.getLogger().setLevel(logging.INFO)

rnn_cell = tf.contrib.rnn
seq2seq = tf.contrib.legacy_seq2seq

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

config = Config()
n_steps = config.step

loc_mean_arr = []
sampled_loc_arr = []


def get_next_input(output, i):
  loc, loc_mean = loc_net(output)
  gl_next = gl(loc)
  loc_mean_arr.append(loc_mean)
  sampled_loc_arr.append(loc)
  return gl_next

# placeholders
images_ph = tf.placeholder(tf.float32,
                           [None, config.original_size * config.original_size *
                            config.num_channels])
labels_ph = tf.placeholder(tf.int64, [None])

# Build the aux nets.
with tf.variable_scope('glimpse_net'):
  gl = GlimpseNet(config, images_ph)
with tf.variable_scope('loc_net'):
  loc_net = LocNet(config)

# number of examples
N = tf.shape(images_ph)[0]
init_loc = tf.random_uniform((N, 2), minval=-1, maxval=1)
init_glimpse = gl(init_loc)
# Core network.
lstm_cell = rnn_cell.LSTMCell(config.cell_size, state_is_tuple=True)
init_state = lstm_cell.zero_state(N, tf.float32)
inputs = [init_glimpse]
inputs.extend([0] * (config.num_glimpses))
outputs, _ = seq2seq.rnn_decoder(
    inputs, init_state, lstm_cell, loop_function=get_next_input)

# Time independent baselines
with tf.variable_scope('baseline'):
  w_baseline = weight_variable((config.cell_output_size, 1))
  b_baseline = bias_variable((1,))
baselines = []
for t, output in enumerate(outputs[1:]):
  baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)
  baseline_t = tf.squeeze(baseline_t)
  baselines.append(baseline_t)
baselines = tf.stack(baselines)  # [timesteps, batch_sz]
baselines = tf.transpose(baselines)  # [batch_sz, timesteps]

# Take the last step only.
output = outputs[-1]
# Build classification network.
with tf.variable_scope('cls'):
  w_logit = weight_variable((config.cell_output_size, config.num_classes))
  b_logit = bias_variable((config.num_classes,))
logits = tf.nn.xw_plus_b(output, w_logit, b_logit)
softmax = tf.nn.softmax(logits)

# cross-entropy.
xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph)
xent = tf.reduce_mean(xent)
pred_labels = tf.argmax(logits, 1)
# 0/1 reward.
reward = tf.cast(tf.equal(pred_labels, labels_ph), tf.float32)
rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
rewards = tf.tile(rewards, (1, config.num_glimpses))  # [batch_sz, timesteps]
logll = loglikelihood(loc_mean_arr, sampled_loc_arr, config.loc_std)
advs = rewards - tf.stop_gradient(baselines)
logllratio = tf.reduce_mean(logll * advs)
reward = tf.reduce_mean(reward)

baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
var_list = tf.trainable_variables()
# hybrid loss
loss = -logllratio + xent + baselines_mse 
grads = tf.gradients(loss, var_list)
grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)

# learning rate
global_step = tf.get_variable(
    'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
training_steps_per_epoch = mnist.train.num_examples // config.batch_size
starter_learning_rate = config.lr_start
# decay per training epoch
learning_rate = tf.train.exponential_decay(
    starter_learning_rate,
    global_step,
    training_steps_per_epoch*10,
    0.99,
    staircase=True)
learning_rate = tf.maximum(learning_rate, config.lr_min)
#opt = tf.train.AdamOptimizer(learning_rate)
opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)

print(var_list, grads) 
for var in var_list:
    tf.summary.histogram(var.op.name + "values", var)
for grad, var in zip(grads, var_list):
    #if grad is None:
    #    continue
    tf.summary.histogram(var.op.name + "gradients", grad)
  
#tf.summary.image('images_ph', tf.reshape(images_ph, [tf.shape(images_ph)[0], config.original_size, config.original_size, config.num_channels]), max_outputs = 1)
tf.summary.scalar('learning_rate', learning_rate)
tf.summary.scalar('reward', reward)
tf.summary.scalar('total_loss', loss)
tf.summary.scalar('baselines_mse', baselines_mse)
tf.summary.scalar('reinforcement_loss', -logllratio)
tf.summary.scalar('classification_loss', xent)

restore_path = None #'./save/ram-3626698'

gpu_config = tf.ConfigProto(allow_soft_placement=True)
gpu_config.gpu_options.allow_growth=True

with tf.Session(config = gpu_config) as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  if restore_path is not None:
      saver.restore(sess, restore_path)
      logging.info('{} model restored!'.format(restore_path))

  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('./summary/train', sess.graph)
  test_writer = tf.summary.FileWriter('./summary/test')

  for i in xrange(n_steps):
    images, labels = mnist.train.next_batch(config.batch_size)
    # duplicate M times, see Eqn (2)
    images = np.tile(images, [config.M, 1])
    labels = np.tile(labels, [config.M])
    loc_net.samping = True
    adv_val, baselines_mse_val, xent_val, logllratio_val, \
        reward_val, loss_val, lr_val, summary, _ = sess.run(
            [advs, baselines_mse, xent, logllratio,
             reward, loss, learning_rate, merged, train_op],
            feed_dict={
                images_ph: images,
                labels_ph: labels
            })
    if i and i % 100 == 0:
      logging.info('step {}: lr = {:3.6f}'.format(i, lr_val))
      logging.info(
          'step {}: reward = {:3.4f}\tloss = {:3.4f}\txent = {:3.4f}'.format(
              i, reward_val, loss_val, xent_val))
      logging.info('llratio = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
          logllratio_val, baselines_mse_val))
      train_writer.add_summary(summary, i)

    if i and i % (training_steps_per_epoch*10) == 0:
      # Evaluation
      for dataset in [mnist.validation, mnist.test]:
        steps_per_epoch = dataset.num_examples // config.eval_batch_size
        correct_cnt = 0
        num_samples = steps_per_epoch * config.batch_size
        loc_net.sampling = False
        for test_step in xrange(steps_per_epoch):
          images, labels = dataset.next_batch(config.batch_size)
          labels_bak = labels
          # Duplicate M times
          images = np.tile(images, [config.M_TEST, 1])
          labels = np.tile(labels, [config.M_TEST])
          softmax_val, summary = sess.run([softmax, merged],
                                 feed_dict={
                                     images_ph: images,
                                     labels_ph: labels
                                 })
          softmax_val = np.reshape(softmax_val,
                                   [config.M_TEST, -1, config.num_classes])
          softmax_val = np.mean(softmax_val, 0)
          pred_labels_val = np.argmax(softmax_val, 1)
          pred_labels_val = pred_labels_val.flatten()
          correct_cnt += np.sum(pred_labels_val == labels_bak)
          test_writer.add_summary(summary, i)
        acc = correct_cnt / num_samples
        if dataset == mnist.validation:
          logging.info('valid accuracy = {}'.format(acc))
        else:
          logging.info('test accuracy = {}'.format(acc))

      saver.save(sess, "./save/ram", global_step = i)
