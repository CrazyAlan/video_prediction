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

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      
      # expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(g)

    # Average over the 'tower' dimension.
    # grad = tf.concat(0, grads)
    # import pdb
    # pdb.set_trace()
    grad = tf.reduce_mean(grads, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(tf.reduce_mean(total_loss,0))

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
    
    grads = []   
    for loss in total_loss:
      grads.append(opt.compute_gradients(loss, update_gradient_vars))
    
    grads_mean = average_gradients(grads)
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads_mean, global_step=global_step)
  
    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
   
    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads_mean:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
    # import pdb
    # pdb.set_trace()

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      with tf.control_dependencies(update_ops):
        train_op = tf.no_op(name='train')
  
    return train_op

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

  def __init__(self,
               images,
               labels,
               global_step,
               reuse=None):

    labels = slim.one_hot_encoding(
      tf.string_to_number(labels, out_type=tf.int32), FLAGS.nrof_classes)
    # inputs has shape [batch, 224, 224, 3]
    init_from_checkpoint = None
    train_op = None
    cross_entropy = None

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                           FLAGS.decay_step, 0.1, staircase=True)


    if reuse is None:    
      network_fn = nets_factory.get_network_fn(
        FLAGS.model,
        num_classes=FLAGS.nrof_classes,
        partial_bn=FLAGS.partial_bn,
        is_training= not FLAGS.partial_bn,
        weight_decay=FLAGS.weight_decay,
        batch_norm_decay = FLAGS.batch_norm_decay)

      images_batch = tf.split(images, num_or_size_splits=FLAGS.batch_size/FLAGS.sub_batch_size, axis=0)
      labels_batch = tf.split(labels, num_or_size_splits=FLAGS.batch_size/FLAGS.sub_batch_size, axis=0)

      train_loss = []
      # correct_prediction = []
      accuracy = 0.0

      for i in range(FLAGS.batch_size/FLAGS.sub_batch_size):
        if i == 0:
          net, end_points = network_fn(images_batch[i])
        else:
          net, end_points = network_fn(images_batch[i], reuse=True)
        cross_entropy = tf.losses.softmax_cross_entropy(
          labels_batch[i],tf.squeeze(net))

        correct_prediction = tf.equal(tf.argmax(tf.squeeze(end_points['predictions']),1), tf.argmax(labels_batch[i],1))
        accuracy += tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        train_loss.append(cross_entropy)

      init_from_checkpoint = _get_init_fn()

      train_op = train(train_loss, global_step,
                     FLAGS.optimizer, learning_rate, 
                     0.9999, _get_variables_to_train()) 

      accuracy /= (FLAGS.batch_size/FLAGS.sub_batch_size)
      
      tf.summary.scalar('train_acc', accuracy)
      self.cross_entropy = tf.reduce_mean(train_loss,0)


    else:
      network_fn = nets_factory.get_network_fn(
        FLAGS.model,
        num_classes=FLAGS.nrof_classes,
        is_training=False,
        weight_decay=FLAGS.weight_decay)

      net, end_points = network_fn(images, reuse=True)

      logits_mean = tf.reshape(tf.reduce_mean(net, 0), [1,-1])
      # import pdb
      # pdb.set_trace()
      single_prediction = slim.softmax(logits_mean, scope='single_predictions')
      correct_prediction = tf.equal(tf.argmax(tf.squeeze(single_prediction),0), tf.argmax(labels[0],0))
      accuracy = tf.squeeze(tf.cast(correct_prediction, tf.float32))

      tf.summary.scalar('val_acc', accuracy)
  
      # cross_entropy = tf.losses.softmax_cross_entropy(
      #   labels,tf.squeeze(net))

      # correct_prediction = tf.equal(tf.argmax(tf.squeeze(end_points['predictions']),1), tf.argmax(labels,1))
      # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # correct_prediction = tf.equal(tf.argmax(tf.squeeze(end_points['predictions']),1), tf.argmax(labels,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    summary_op = tf.summary.merge_all()

    self.init_from_checkpoint = init_from_checkpoint
    self.accuracy = accuracy
    self.train_op = train_op
    self.summary_op = summary_op
    self.learning_rate = learning_rate
