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

"""Model architecture for prednet"""

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from lstm_ops import basic_conv_lstm_cell

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12

def init_error(img_height, img_width,stack_sizes, batch_size, nb_layers):
  batch_size = tf.to_int32(batch_size)
  init_error = []
  for l in range(nb_layers):
    factor = 2**l
    init_error.append(tf.Variable(tf.zeros([batch_size,img_height/factor,img_width/factor,stack_sizes[l]*2])))
  return init_error

def construct_model(images,
                    nb_layers=4,
                    iter_num=-1.0,
                    k=-1,
                    use_state=False,
                    context_frames=2,
                    pixel_max=1.):
  """Build convolutional lstm video predictor using STP, CDNA, or DNA.

  Args:
    images: tensor of ground truth image sequences
    iter_num: tensor of the current training iteration (for sched. sampling)
    k: constant used for scheduled sampling. -1 to feed in own prediction.
    context_frames: number of ground truth frames to pass in before
                    feeding in own predictions
  Returns:
    gen_images: predicted future image frames

  Raises:
    ValueError: if more than one network option specified or more than 1 mask
    specified for DNA model.
  """
 
  batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
  batch_size, img_height, img_width, color_channels = int(batch_size), int(img_height), int(img_width), int(color_channels)
  lstm_func = basic_conv_lstm_cell

  # Generated robot states and images.
  gen_images = []

  if k == -1:
    feedself = True
  else:
    # Scheduled sampling:
    # Calculate number of ground-truth frames to pass in.
    num_ground_truth = tf.to_int32(
        tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(iter_num / k)))))
    feedself = False

  # LSTM state sizes and states.
  stack_sizes = np.int32(np.array([color_channels, 48, 96, 192]))
  R_stack_sizes = stack_sizes
  A_filt_sizes = np.int32(np.array([3, 3, 3]))
  Ahat_filt_sizes = np.int32(np.array([3, 3, 3, 3]))
  R_filt_sizes = np.int32(np.array([3, 3, 3, 3]))
  
  lstm_state = nb_layers*[None]
  hidden = nb_layers*[None]
  e_state = init_error(img_height, img_width,stack_sizes, batch_size, nb_layers)

  for image in images[:-1]:
    # Last till the last 2nd image
    # Reuse variables after the first timestep.
    reuse = bool(gen_images)

    done_warm_start = len(gen_images) > context_frames - 1
    with slim.arg_scope(
        [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
         tf_layers.layer_norm, slim.layers.conv2d_transpose],
        reuse=reuse):

      if feedself and done_warm_start:
        # Feed in generated image.
        prev_image = gen_images[-1]
      elif done_warm_start:
        # Scheduled sampling
        prev_image = scheduled_sample(image, gen_images[-1], batch_size,
                                      num_ground_truth)
      else:
        # Always feed in ground_truth
        prev_image = image

      # Predicted state is always fed back in
      # state_action = tf.concat(axis=1, values=[action, current_state])

      # Update R states
      for l in reversed(range(nb_layers)):
        if l == nb_layers-1:
          hidden[l], lstm_state[l] = lstm_func(e_state[l], lstm_state[l], stack_sizes[l], scope='lstm'+str(l))
        else:
          # Add upsampling from previous layer
          r_up = slim.layers.conv2d_transpose(hidden[l+1], hidden[l+1].get_shape()[3], 3, stride=2, scope='up_sample'+str(l))
          # import pdb
          # pdb.set_trace()
          # r_up = tf.image.resize_images(hidden[l+1], [int(hidden[l+1].get_shape()[1])*2, int(hidden[l+1].get_shape()[2])*2])
          feature = tf.concat([e_state[l], r_up], axis=3)
          hidden[l], lstm_state[l] = lstm_func(feature, lstm_state[l], stack_sizes[l], scope='lstm'+str(l))
        hidden[l] = tf_layers.layer_norm(hidden[l], scope='layer_norm'+str(l))
          
      # Update A and A_hat    
      A=[prev_image] 
      A_hat=[]

      for l in range(nb_layers):
        
          net=slim.conv2d(hidden[l], hidden[l].get_shape()[3], [Ahat_filt_sizes[l],Ahat_filt_sizes[l]], stride=1, scope='conv'+str(l))
          net=tf.nn.relu(net - RELU_SHIFT) + RELU_SHIFT
          if l == 0:
            net=tf.minimum(net, tf.ones_like(net)*pixel_max)
          A_hat.append(net)
          # print('Aand A_hat', A[l], A_hat[l])
          # Computer the error 
          # import pdb
          # pdb.set_trace()
          e_pos = tf.nn.relu(A[l]-A_hat[l] - RELU_SHIFT) + RELU_SHIFT
          e_neg = tf.nn.relu(A_hat[l]-A[l] - RELU_SHIFT) + RELU_SHIFT
          e_state[l] = tf.concat([e_pos, e_neg], 3)

          if l < nb_layers-1:
            conv_e = slim.conv2d(e_state[l], stack_sizes[l+1], [A_filt_sizes[l], A_filt_sizes[l]] ,stride=1, scope='conv_e'+str(l))
            A.append(tf.nn.max_pool(conv_e, [1,2,2,1], [1,2,2,1],'SAME'))

      gen_images.append(A_hat[0])

  return gen_images





def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
  """Sample batch with specified mix of ground truth and generated data points.

  Args:
    ground_truth_x: tensor of ground-truth data points.
    generated_x: tensor of generated data points.
    batch_size: batch size
    num_ground_truth: number of ground-truth examples to include in batch.
  Returns:
    New batch with num_ground_truth sampled from ground_truth_x and the rest
    from generated_x.
  """
  idx = tf.random_shuffle(tf.range(int(batch_size)))
  ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
  generated_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))

  ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
  generated_examps = tf.gather(generated_x, generated_idx)
  return tf.dynamic_stitch([ground_truth_idx, generated_idx],
                           [ground_truth_examps, generated_examps])
