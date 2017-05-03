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

def cost_con(X, M, Xt, Mt, masked=True):
  if masked:
    err_x = tf.multiply((X - Xt), Mt)
  else:
    err_x = X - Xt

  err_m = M - Mt

  cost_x = tf.nn.l2_loss(err_x)
  cost_m = tf.nn.l2_loss(err_m)

  cost = cost_x*FLAGS.lambda_rgb + cost_m*FLAGS.lambda_mask

  return cost

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

def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):
    loss_averages_op = _add_loss_summaries(total_loss)
  
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

    grads = opt.compute_gradients(total_loss, update_gradient_vars)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
      apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
   
    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    with tf.control_dependencies([apply_gradient_op]):
      train_op = tf.no_op(name='train')
    
    return train_op

def update(total_loss, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):
    loss_averages_op = _add_loss_summaries(total_loss)
  
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

    grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
   
    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    apply_gradient_op = opt.apply_gradients(grads)

    with tf.control_dependencies([apply_gradient_op]):
      update_op = tf.no_op()
    
    return update_op


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

def soft_loss(label, pred):
  labels = np.ones((FLAGS.batch_size), dtype=np.int)*int(label)
  tf_labels = slim.one_hot_encoding(labels, 2)

  cost = tf.losses.softmax_cross_entropy(
          tf_labels, pred)

  correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(tf_labels,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  return cost, accuracy

class Model(object):

  def __init__(self,
               batch_sprites,
               batch_masks,
               global_step,
               reuse=None):

    arg_scope = network.arg_sco()

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                   FLAGS.decay_step, 0.1, staircase=True)

    update_disc_op = None
    update_gen_op = None
    update_enc_op = None
    disc_acc = None
    discr_loss_ratio = None
    gen_loss = None
    disc_loss= None
    disc_gen_loss = None
    feat_loss = None
    disc_real_loss = None
    disc_pred_loss = None, 
    with slim.arg_scope(arg_scope):
      if reuse == None:

        ref, _ = network.enc(batch_sprites[0], collection_name='ref')
        out, _ = network.enc(batch_sprites[1], collection_name='out', reuse=True)
        query, _ = network.enc(batch_sprites[2], collection_name = 'query', reuse=True)
        target, target_end_points = network.enc(batch_sprites[3], collection_name='target', reuse=True)

        #Vector addition for analogy
        top = query + network.ana_inc(out, ref, query, option='Deep')

        pred_masks, _ = network.dec_mask(top)
        pred_sprites, _ = network.dec_rgb(top)

        # calculate reconstruction cost
        recon_loss = cost_con(pred_sprites, pred_masks, batch_sprites[3], batch_masks[3])

        # caluculate feature loss, use sprites only
        pred_comb = tf.multiply(pred_masks, pred_sprites)
        target_pred, target_pred_end_points = network.enc(pred_comb, collection_name='pred', reuse=True)
        # feat_lname = FLAGS.model+'_enc/Flatten'

        feat_loss = tf.nn.l2_loss(target_end_points[target_end_points.keys()[2]]-target_pred_end_points[target_pred_end_points.keys()[2]])

        # import pdb
        # pdb.set_trace()
        # calculate disc cost
        real_label, _ = network.disc(target_end_points[target_end_points.keys()[2]],
                                  batch_sprites[3])
        disc_real_loss, disc_real_acc = soft_loss(1, real_label)

        pred_label, _ = network.disc(target_end_points[target_end_points.keys()[2]],
                                  pred_comb,
                                  reuse=reuse)
        disc_pred_loss, disc_pred_acc = soft_loss(0, pred_label)
        disc_loss = disc_pred_loss + disc_real_loss

        disc_acc = [disc_real_acc, disc_pred_acc]

        # calculate disc_gen_loss
        disc_gen_loss, _ = soft_loss(1, pred_label)
        

        # update disc 
        update_disc_op = update(disc_loss, FLAGS.optimizer, 
                                learning_rate, 0.9999, 
                                _get_variables_to_train_with_option(option=FLAGS.model+'_disc'))

        # update generator(decoder)
        gen_loss = FLAGS.lambda_img*recon_loss + FLAGS.lambda_adv*disc_gen_loss + FLAGS.lambda_feat*feat_loss
        gen_scope = FLAGS.model+'_dec_rgb,'+FLAGS.model+'_dec_mask,' + FLAGS.model+'_inc_net'
        update_gen_op = update(gen_loss, FLAGS.optimizer, 
                               learning_rate, 0.9999, 
                               _get_variables_to_train_with_option(option=gen_scope))
        # update encoder
        enc_loss = gen_loss
        enc_scope = FLAGS.model+'_enc'                        
        update_enc_op = update(enc_loss, FLAGS.optimizer, 
                               learning_rate, 0.9999, 
                               _get_variables_to_train_with_option(option=enc_scope))

        discr_loss_ratio = (disc_real_loss+disc_pred_loss) / disc_gen_loss

        # train_op = train(recon_loss, global_step,
        #         FLAGS.optimizer, learning_rate,
        #         0.9999, _get_variables_to_train())
      else:
        # import pdb
        # pdb.set_trace()

        ref, _ = network.enc(batch_sprites[0], reuse=reuse)
        out, _ = network.enc(batch_sprites[1], reuse=reuse)
        query, _ = network.enc(batch_sprites[2], reuse=reuse)

        #Vector addition for analogy
        top = query + network.ana_inc(out, ref, query, option='Deep', reuse=True)

        pred_masks, _ = network.dec_mask(top, reuse=reuse)
        pred_sprites, _ = network.dec_rgb(top, reuse=reuse)

        # calculate cost
        recon_loss = cost_con(pred_sprites, pred_masks, batch_sprites[3], batch_masks[3])

        pred_comb = tf.multiply(pred_masks, pred_sprites)


      summary_op = tf.summary.merge_all()

    init_from_checkpoint = _get_init_fn()

    self.init_from_checkpoint = init_from_checkpoint
    self.update_enc_op = update_enc_op
    self.update_disc_op = update_disc_op
    self.update_gen_op = update_gen_op
    self.disc_acc = disc_acc
    self.discr_loss_ratio = discr_loss_ratio
    self.gen_loss = gen_loss
    self.other_loss = [disc_real_loss, disc_pred_loss, disc_gen_loss, feat_loss]
    self.recon_loss = recon_loss

    self.summary_op = summary_op
    self.learning_rate = learning_rate
    self.predict = [pred_sprites, pred_masks]
    self.pred_comb = pred_comb