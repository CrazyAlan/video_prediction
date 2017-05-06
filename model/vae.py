# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Cross Convolutional Model.

https://arxiv.org/pdf/1607.02586v1.pdf
"""
import math
import sys

import numpy as np
from datetime import datetime
import tensorflow as tf
import os
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from tensorflow.python.ops import control_flow_ops
from slim.nets import resnet_v1
from slim.nets import nets_factory

slim = tf.contrib.slim

FLAGS = flags.FLAGS

if FLAGS.model == 'ana_sprite_3':
  from network import ana_sprite_3
  network = ana_sprite_3


def _add_loss_summaries(total_loss):
    """Add summaries for losses.
  
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
#     losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply([total_loss])
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +'raw', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
  
    return loss_averages_op

def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train

def _get_variables_to_train_with_option(option=None):
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if option is None:
    return None
  else:
    scopes = [scope.strip() for scope in option.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train

def _soft_loss(label, pred):
  labels = np.ones((FLAGS.batch_size), dtype=np.int)*int(label)
  tf_labels = slim.one_hot_encoding(labels, 2)

  cost = tf.losses.softmax_cross_entropy(
          tf_labels, pred)

  correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(tf_labels,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return cost, accuracy

def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
#   # ignoring the checkpoint anyway.
#   if tf.train.latest_checkpoint(FLAGS.train_dir):
#     tf.logging.info(
#         'Ignoring --checkpoint_path because a checkpoint already exists in %s'
#         % FLAGS.train_dir)
#     return None

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
    for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars)

class Model(object):

  def __init__(self, batch_sprites, batch_masks, global_step, is_training):
    """Constructor.
    """
    self.batch_sprites = batch_sprites
    self.batch_masks = batch_masks
    self.global_step = global_step
    self.is_training = is_training


  def Build(self):
    # with tf.device('/gpu:0'):
    arg_scope = network.arg_sco()

    self.learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step,
                 FLAGS.decay_step, 0.1, staircase=True)

    with slim.arg_scope(arg_scope):

      inc = self._BuildAnalogyInc()
      # Add the increasing information to log
      tf.summary.histogram('inc_info', inc)
      
      encoded_image_info = self.ref
      decoded_image_info = encoded_image_info + inc 
      self._BuildImageDecoder(decoded_image_info)
      # Build Discriminator 
      self._BuildDisc()
      # Build Loss
      self._BuildLoss()

    if self.is_training:

      self._BuildTrainOp(self.global_step,
                         FLAGS.optimizer,
                         self.learning_rate,
                         0.9999)

    self.summary_op = tf.summary.merge_all()
    self.init_from_checkpoint = _get_init_fn()

  def _BuildTrainOp(self, global_step, optimizer, learning_rate, moving_average_decay):
    loss_averages_op = _add_loss_summaries(self.loss)
    
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
      
      if FLAGS.train_enc:
        enc_var = _get_variables_to_train_with_option(option=FLAGS.model+'_enc')
        enc_grads = opt.compute_gradients(self.loss, enc_var)
        self.enc_update_ops = opt.apply_gradients(enc_grads)
        # keep grads histogram
        self._grad_hist(enc_grads)
      else:
        self.enc_update_ops = tf.no_op(name='enc_train')

      if FLAGS.train_gen:
        gen_scope = ('{model}_dec_rgb,{model}_dec_mask,{model}_inc_net,{model}_inc_info_enc,{model}_inc_info_dec').format(model=FLAGS.model)
        gen_var = _get_variables_to_train_with_option(option=gen_scope)
        gen_grads = opt.compute_gradients(self.loss, gen_var)
        self.gen_update_ops = opt.apply_gradients(gen_grads, global_step=global_step)
        # keep grads histgram
        self._grad_hist(gen_grads)
      else:
        self.gen_update_ops = tf.no_op(name='gen_train')

      if FLAGS.train_vae_inc:
        vae_inc_scope = ('{model}_inc_info_enc,{model}_inc_info_dec').format(model=FLAGS.model)
        vae_inc_var = _get_variables_to_train_with_option(option=vae_inc_scope)
        vae_inc_grads = opt.compute_gradients(self.loss, vae_inc_var)
        self.vae_inc_update_ops = opt.apply_gradients(vae_inc_grads)
        # keep grads hist
        self._grad_hist(vae_inc_grads)
      else:
        self.vae_inc_update_ops = tf.no_op(name='inc_info_train')

      if FLAGS.train_disc:
        disc_var = _get_variables_to_train_with_option(option=FLAGS.model+'_disc')
        disc_grads = opt.compute_gradients(self.disc_loss, disc_var)
        self.disc_update_ops = opt.apply_gradients(disc_grads)
        # keep grads hist
        self._grad_hist(disc_grads)
      else:
        self.disc_update_ops = tf.no_op(name='disc_train')

    # Add histograms for trainable variables.
    if FLAGS.log_histograms:
      for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
  
  def _grad_hist(sefl, grads):
      
    # Add histograms for gradients.
    if FLAGS.log_histograms:
      for grad, var in grads:
        if grad is not None:
          tf.summary.histogram(var.op.name + '/gradients', grad)

  def _BuildLoss(self):
    # 1. reconstr_loss seems doesn't do better than l2 loss.
    # 2. Only works when using reduce_mean. reduce_sum doesn't work.
    # 3. It seems kl loss doesn't play an important role.
    self.loss = 0
    self.recon_loss = 0
    self.kl_loss = 0
    self.feat_loss = 0
    self.disc_loss = 0
    self.disc_gen_loss = 0

    with tf.variable_scope('loss'):
      self.recon_loss = self.cost_con(self.batch_sprites[1],
                                 self.batch_masks[1],
                                 self.pred_sprites,
                                 self.pred_masks)
      
      if FLAGS.kl_loss:
        self.kl_loss = (0.5 * tf.reduce_mean(
            tf.square(self.z_mean) + tf.square(self.z_stddev) -
            2 * self.z_stddev_log - 1))
        tf.summary.scalar('kl_loss', self.kl_loss)
        self.loss += self.kl_loss

      if FLAGS.feat_loss:
        self.feat_loss = tf.nn.l2_loss(self.out_endpoints[self.out_endpoints.keys()[2]] \
                                        - self.out_pred_endpoints[self.out_pred_endpoints.keys()[2]])

      if FLAGS.disc_loss:
        disc_real_loss, self.disc_real_acc = _soft_loss(1, self.real_label)
        disc_pred_loss, self.disc_pred_acc = _soft_loss(0, self.pred_label)
        self.disc_loss =  disc_pred_loss + disc_real_loss
        self.disc_gen_loss, _ = _soft_loss(1, self.pred_label)
        self.discr_loss_ratio = (disc_real_loss + disc_pred_loss) / self.disc_gen_loss
      
      self.loss = FLAGS.lambda_img*self.recon_loss +  FLAGS.lambda_adv*self.disc_gen_loss + FLAGS.lambda_feat*self.feat_loss
        
      tf.summary.scalar('loss', self.loss)

  def cost_con(self, X, M, Xt, Mt, masked=True):
    if masked:
      err_x = tf.multiply((X - Xt), Mt)
    else:
      err_x = X - Xt

    err_m = M - Mt

    cost_x = tf.nn.l2_loss(err_x)
    cost_m = tf.nn.l2_loss(err_m)

    cost = cost_x*FLAGS.lambda_rgb + cost_m*FLAGS.lambda_mask

    return cost

  def _BuildAnalogyInc(self):
    
    # Analogy encoder
    self.ref, self.ref_endpoints = network.enc(self.batch_sprites[0], collection_name='ref')    
    self.out, self.out_endpoints = network.enc(self.batch_sprites[1], collection_name='out', reuse=True)

    inc_info = network.ana_inc(self.out, self.ref, self.out, option='Deep')     
    
    z, _ = network.inc_info_enc(inc_info)

    self.z_mean, self.z_stddev_log = tf.split(
        axis=1, num_or_size_splits=2, value=z)
    self.z_stddev = tf.exp(self.z_stddev_log)

    tf.summary.histogram('z_mean', self.z_mean)
    tf.summary.histogram('z_stddev', self.z_stddev)

    epsilon = tf.random_normal(
        [FLAGS.batch_size, self.z_mean.get_shape().as_list()[1]], 0, 1, dtype=tf.float32)
    hid =  self.z_mean + tf.multiply(self.z_stddev, epsilon)

    net, _ = network.inc_info_dec(hid)

    return net

  def _BuildImageDecoder(self, decoded_image_info):
    """Decode the hidden into the predicted images and masks"""
    self.pred_masks, _ = network.dec_mask(decoded_image_info)
    self.pred_sprites, _ = network.dec_rgb(decoded_image_info)

    self.pred_comb = tf.multiply(self.pred_masks, self.pred_sprites)

  def _BuildDisc(self):
    self.real_label, _ = network.disc(self.out_endpoints[self.out_endpoints.keys()[2]],
                                 self.batch_sprites[1])

    # Forward predicted image in encoder, to get the feat map
    self.out_pred, self.out_pred_endpoints = network.enc(self.pred_comb, collection_name='pred', reuse=True)

    self.pred_label, _ = network.disc(self.out_endpoints[self.out_endpoints.keys()[2]],
                                 self.pred_comb,
                                 reuse=True)
