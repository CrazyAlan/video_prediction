from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

slim = tf.contrib.slim

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def arg_sco(weight_decay=0.0005):
  """Defines the ana_sprite_3 arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

def enc(batch_sprites,
		is_training=True,
		dropout_keep_prob=0.5,
		scope='ana_sprite_3_enc'):
	
	with tf.variable_scope(scope, 'ana_sprite_3_enc', [batch_sprites]) as sc:
		end_points_collection = sc.name + '_end_points'
		with slim.arg_scope([slim.conv2d, slim.max_pool2d],
            outputs_collections=end_points_collection):
			net = slim.conv2d(batch_sprites, 64, [5,5], stride=2, scope='conv1')
			net = slim.conv2d(net, 32, [5,5], stride=2, scope='conv2')
			net = slim.flatten(net)
			net = slim.fully_connected(net, 2048, scope='fc3')
			net = slim.fully_connected(net, 1024, scope='fc4', activation_fn=None)

			end_points = slim.utils.convert_collection_to_dict(end_points_collection)

			return net, end_points

# Decode for RGB
def dec_rgb(hid,
		is_training=True,
		dropout_keep_prob=0.5,
		scope='ana_sprite_3_dec_rgb'):
	with tf.variable_scope(scope, 'ana_sprite_3_dec_rgb', [hid]) as sc:
		end_points_collection = sc.name + '_end_points'
		with slim.arg_scope([slim.conv2d, slim.max_pool2d],
            outputs_collections=end_points_collection):
			net = slim.fully_connected(hid, 2048, scope='fc1')
			net = slim.fully_connected(net, 7200, scope='fc2')
			net = tf.reshape(net, [FLAGS.batch_size, 15, 15, 32], name='fc_reshape')
			# It has relu activation for this up_sampling
			net = slim.layers.conv2d_transpose(net, 32, 2, stride=2, scope='up_sample_3')
			net = slim.conv2d(net, 48, [5,5], stride=1, scope='conv4')
			net = slim.layers.conv2d_transpose(net, 48, 2, stride=2, scope='up_sample_4')
			net = slim.conv2d(net, 3, [5,5], stride=1, scope='conv5')

			end_points = slim.utils.convert_collection_to_dict(end_points_collection)

			return net, end_points

def dec_mask(hid,
		is_training=True,
		dropout_keep_prob=0.5,
		scope='ana_sprite_3_dec_mask'):
	with tf.variable_scope(scope, 'ana_sprite_3_dec_mask', [hid]) as sc:
		end_points_collection = sc.name + '_end_points'
		with slim.arg_scope([slim.conv2d, slim.max_pool2d],
            outputs_collections=end_points_collection):
			
			net = slim.fully_connected(hid, 2048, scope='fc1')
			net = slim.fully_connected(net, 5400, scope='fc2')
			# import pdb
			# pdb.set_trace()

			net = tf.reshape(net, [FLAGS.batch_size, 15, 15, 24], name='fc_reshape')
			# It has relu activation for this up_sampling
			net = slim.layers.conv2d_transpose(net, 24, 2, stride=2, scope='up_sample_3')
			net = slim.conv2d(net, 24, [5,5], stride=1, scope='conv4')
			net = slim.layers.conv2d_transpose(net, 24, 2, stride=2, scope='up_sample_4')
			net = slim.conv2d(net, 1, [5,5], stride=1, scope='conv5')

			end_points = slim.utils.convert_collection_to_dict(end_points_collection)

			return net, end_points