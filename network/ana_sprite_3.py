from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

slim = tf.contrib.slim

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

model_name = 'ana_sprite_3'
def arg_sco(weight_decay=0.0005):
  """Defines themodel_name  arg scope.

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
        scope=model_name+'_enc',
        collection_name='',
        reuse=None):
    
    with tf.variable_scope(scope, model_name+'_enc', [batch_sprites], reuse=reuse) as sc:
        end_points_collection = sc.name + collection_name +'_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.flatten],
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
        scope=model_name+'_dec_rgb',
        reuse=None):
    with tf.variable_scope(scope, model_name+'_dec_rgb', [hid], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
            outputs_collections=end_points_collection):
            net = slim.fully_connected(hid, 2048, scope='fc1')
            net = slim.fully_connected(net, 7200, scope='fc2')

            net = tf.reshape(net, [-1, 15, 15, 32], name='fc_reshape')
            # It has relu activation for this up_sampling
            net = slim.conv2d_transpose(net, 32, 2, stride=2, scope='up_sample_3')
            net = slim.conv2d(net, 48, [5,5], stride=1, scope='conv4')
            net = slim.conv2d_transpose(net, 48, 2, stride=2, scope='up_sample_4')
            net = slim.conv2d(net, 3, [5,5], stride=1, scope='conv5')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points

def dec_mask(hid,
        is_training=True,
        dropout_keep_prob=0.5,
        scope=model_name+'_dec_mask',
        reuse=None):
    with tf.variable_scope(scope, model_name+'_dec_mask', [hid], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
            outputs_collections=end_points_collection):
            
            net = slim.fully_connected(hid, 2048, scope='fc1')
            net = slim.fully_connected(net, 5400, scope='fc2')
            # import pdb
            # pdb.set_trace()

            net = tf.reshape(net, [-1, 15, 15, 24], name='fc_reshape')
            # It has relu activation for this up_sampling
            net = slim.conv2d_transpose(net, 24, 2, stride=2, scope='up_sample_3')
            net = slim.conv2d(net, 24, [5,5], stride=1, scope='conv4')
            net = slim.conv2d_transpose(net, 24, 2, stride=2, scope='up_sample_4')
            net = slim.conv2d(net, 1, [5,5], stride=1, scope='conv5')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points

def inc_net(trans,
            query,
            is_training=True,
            scope=model_name+'_inc_net',
            reuse=None):
    with tf.variable_scope(scope, model_name+'_inc_net', [trans, query], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.fully_connected],
            outputs_collections=end_points_collection): 

            net_1 = slim.fully_connected(trans, 300, scope='fc1_1', activation_fn=None)
            net_2 = slim.fully_connected(query, 300, scope='fc1_2', activation_fn=None)
            net = net_1 + net_2
            net = tf.nn.relu(net, name=scope+'r1')
            net = slim.fully_connected(net, 300, scope='fc2')
            net = slim.fully_connected(net, 1024, scope = 'fc3')
            
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            # net = tf.multiply(swith, net, name='switch')
            return net, end_points 
    
def ana_inc(out, ref, query, option='Add', reuse=None):
  if option == 'Add':
    inc = out - ref
  elif option == 'Deep':
    inc, _ = inc_net(out-ref, query, reuse=reuse)
  return inc

def inc_info_enc(inc_info,
                 is_training=True,
                 scope=model_name+'_inc_info_enc',
                 reuse=None):
    with tf.variable_scope(scope, model_name+'_inc_info_enc', [inc_info], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.fully_connected],
            outputs_collections=end_points_collection): 

            net = slim.fully_connected(inc_info, 1024, scope='fc1')
            net = slim.fully_connected(net, 1024, scope='fc2')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points

def inc_info_dec(hid,
                is_training=True,
                scope=model_name+'_inc_info_dec',
                reuse=None):
    with tf.variable_scope(scope, model_name+'_inc_info_dec', [hid], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.fully_connected],
            outputs_collections=end_points_collection): 

            net = slim.fully_connected(hid, 1024, scope='fc1')
            net = slim.fully_connected(net, 1024, scope='fc2')
            
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points


def disc(feat, 
         batch_sprites,
         is_training=True,
         scope=model_name+'_disc',
         reuse=None):
    with tf.variable_scope(scope, model_name+'_disc', [feat, batch_sprites], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
            outputs_collections=end_points_collection): 
            
            net = slim.conv2d(batch_sprites, 64, [5,5], stride=2, scope='conv1')
            net = slim.conv2d(net, 32, [5,5], stride=2, scope='conv2')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 2048, scope='fc3')
            net = slim.fully_connected(net, 1024, scope='fc4')

            net_feat = slim.fully_connected(feat, 2048, scope='feat_fc3')
            net_feat = slim.fully_connected(net_feat, 1024, scope='feat_fc4')

            net = tf.concat([net, net_feat], 1, name='feat_net_concat')

            net = slim.fully_connected(net, 1024, scope='fc5')
            net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='drop_5')
            net = slim.fully_connected(net, 1024, scope='fc6')
            net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='drop_6')
            net = slim.fully_connected(net, 2, scope='fc7', activation_fn=None)

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points 

