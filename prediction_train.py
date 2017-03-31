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

"""Code for training the prediction model."""

import numpy as np
from datetime import datetime
import tensorflow as tf
import os

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from data.prediction_input import build_tfrecord_input
from model.prediction_model import construct_model

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 2000

# tf record data location:
DATA_DIR = '/cs/vml4/xca64/robot_data/push/push_train'

# local output directory
OUT_DIR = '/cs/vml4/xca64/robot_data/checkpoints'

# summary output dir
SUM_DIR = '/cs/vml4/xca64/robot_data/summaries'

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
flags.DEFINE_string('event_log_dir',SUM_DIR, 'directory for writing summary.')
flags.DEFINE_integer('num_iterations', 100000, 'number of training iterations.')
flags.DEFINE_string('pretrained_model', '' ,
                    'filepath of a pretrained model to initialize from.')

flags.DEFINE_integer('sequence_length', 10,
                     'sequence length, including context frames.')
flags.DEFINE_integer('context_frames', 2, '# of frames before predictions.')
flags.DEFINE_integer('use_state', 1,
                     'Whether or not to give the state+action to the model')

flags.DEFINE_string('model', 'CDNA',
                    'model architecture to use - CDNA, DNA, or STP')

flags.DEFINE_integer('num_masks', 10,
                     'number of masks, usually 1 for DNA, 10 for CDNA, STN.')
flags.DEFINE_float('schedsamp_k', 900.0,
                   'The k hyperparameter for scheduled sampling,'
                   '-1 for no scheduled sampling.')
flags.DEFINE_float('train_val_split', 0.95,
                   'The percentage of files to use for the training set,'
                   ' vs. the validation set.')

flags.DEFINE_integer('batch_size', 32, 'batch size for training')
flags.DEFINE_float('learning_rate', 0.001,
                   'the base learning rate of the generator')



## Helper functions
def peak_signal_to_noise_ratio(true, pred):
  """Image quality metric based on maximal signal power vs. power of the noise.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    peak signal to noise ratio (PSNR)
  """
  return 10.0 * tf.log(1.0 / mean_squared_error(true, pred)) / tf.log(10.0)


def mean_squared_error(true, pred):
  """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))

import moviepy.editor as mpy
def npy_to_gif(npy, filename):
     clip = mpy.ImageSequenceClip(list(npy), fps=10)
     clip.write_gif(filename)

class Model(object):

  def __init__(self,
               images=None,
               actions=None,
               states=None,
               sequence_length=None,
               reuse_scope=None):

    if sequence_length is None:
      sequence_length = FLAGS.sequence_length

    self.prefix = prefix = tf.placeholder(tf.string, [])
    self.iter_num = tf.placeholder(tf.float32, [])
    summaries = []

    # Split into timesteps.
    actions = tf.split(actions, actions.get_shape().as_list()[1],1)
    actions = [tf.squeeze(act) for act in actions]          
    
    states = tf.split( states, states.get_shape().as_list()[1],1)
    states = [tf.squeeze(st) for st in states]

    images = tf.split(images, images.get_shape().as_list()[1],1 )
    images = [tf.squeeze(img) for img in images]
  
    if reuse_scope is None:
      gen_images, gen_states = construct_model(
          images,
          actions,
          states,
          iter_num=self.iter_num,
          k=FLAGS.schedsamp_k,
          use_state=FLAGS.use_state,
          num_masks=FLAGS.num_masks,
          cdna=FLAGS.model == 'CDNA',
          dna=FLAGS.model == 'DNA',
          stp=FLAGS.model == 'STP',
          context_frames=FLAGS.context_frames)
    else:  # If it's a validation or test model.
      with tf.variable_scope(reuse_scope, reuse=True):
        gen_images, gen_states = construct_model(
            images,
            actions,
            states,
            iter_num=self.iter_num,
            k=FLAGS.schedsamp_k,
            use_state=FLAGS.use_state,
            num_masks=FLAGS.num_masks,
            cdna=FLAGS.model == 'CDNA',
            dna=FLAGS.model == 'DNA',
            stp=FLAGS.model == 'STP',
            context_frames=FLAGS.context_frames)

    # L2 loss, PSNR for eval.
    loss, psnr_all = 0.0, 0.0

    try:
       tf.equal(prefix,'train').eval()
    except:
       prefixs='initial'
    else:
       if tf.equal(prefix,'train').eval():
           prefixs='train'
       else:
           prefixs='val'

    print('Prefix is', prefixs)
    for i, x, gx in zip(
        range(len(gen_images)), images[FLAGS.context_frames:],
        gen_images[FLAGS.context_frames - 1:]):
      recon_cost = mean_squared_error(x, gx)
      psnr_i = peak_signal_to_noise_ratio(x, gx)
      psnr_all += psnr_i

      summaries.append(
          tf.summary.scalar(prefixs + '_recon_cost' + str(i), recon_cost))

      summaries.append(tf.summary.scalar(prefixs + '_psnr' + str(i), psnr_i))
      
      loss += recon_cost

    for i, state, gen_state in zip(
        range(len(gen_states)), states[FLAGS.context_frames:],
        gen_states[FLAGS.context_frames - 1:]):
      state_cost = mean_squared_error(state, gen_state) * 1e-4
      summaries.append(
          tf.summary.scalar(prefixs + '_state_cost' + str(i), state_cost))
      loss += state_cost
    summaries.append(tf.summary.scalar(prefixs + '_psnr_all', psnr_all))
    self.psnr_all = psnr_all

    self.loss = loss = loss / np.float32(len(images) - FLAGS.context_frames)

    summaries.append(tf.summary.scalar(prefixs + '_loss', loss))

    # Add image to summary

    summaries.append(tf.summary.image(prefixs + '_gen', gen_images[0], max_outputs=3))    
    summaries.append(tf.summary.image(prefixs + '_org', images[0], max_outputs=3))

    self.lr = tf.placeholder_with_default(FLAGS.learning_rate, ())

    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
    self.summ_op = tf.summary.merge(summaries)

    self.gen_images = gen_images
    self.images = images

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    print('Constructing models and inputs.')
    with tf.variable_scope('model', reuse=None) as training_scope:
      images, actions, states = build_tfrecord_input(training=True)
      model = Model(images, actions, states, FLAGS.sequence_length)

    with tf.variable_scope('val_model', reuse=None):
      val_images, val_actions, val_states = build_tfrecord_input(training=False)
      val_model = Model(val_images, val_actions, val_states,
                        FLAGS.sequence_length, training_scope)

    print('Constructing saver.')
    # Make saver.
    saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=0)
    saver_dir = os.path.join(os.path.expanduser(FLAGS.output_dir), datetime.now().strftime('%Y-%m-%d-%H%M%S'))
    if not os.path.isdir(saver_dir):
      os.mkdir(saver_dir)

    # Make training session.
    # sess = tf.InteractiveSession()
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(
        FLAGS.event_log_dir, graph=sess.graph, flush_secs=10)

    

    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    tf.logging.info('iteration number, cost')

    if FLAGS.pretrained_model:
      files = os.listdir(FLAGS.pretrained_model)
      meta_files = [s for s in files if s.endswith('.meta')]
      if len(meta_files) == 0:
        raise ValueError('No pretrained model find under the directory '+ FLAGS.pretrained_model)
      # saver = tf.train.import_meta_graph(os.path.join(FLAGS.pretrained_model, meta_files[-1]))
      # print(tf.train.latest_checkpoint(FLAGS.pretrained_model))
      print('Start to load pretrained model')
      saver.restore(sess, tf.train.latest_checkpoint(FLAGS.pretrained_model))
      print('Successfully Restored the model')

    print('Start Tranning')
    # Run training.
    for itr in range(FLAGS.num_iterations):
      # print('In iteration ', itr)
      # Generate new batch of data.
      feed_dict = {model.prefix: 'train',
                   model.iter_num: np.float32(itr),
                   model.lr: FLAGS.learning_rate}
      cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op],
                                      feed_dict)

      # Print info: iteration #, cost.
      tf.logging.info('  In Iteration ' + str(itr) + ', Cost ' + str(cost))
      
      if (itr) % VAL_INTERVAL == 2:
        # Run through validation set.
        feed_dict = {val_model.lr: 0.0,
                     val_model.prefix: 'val',
                     val_model.iter_num: np.float32(itr)}
        _, val_summary_str, gen_images, images = sess.run([val_model.train_op, val_model.summ_op, val_model.gen_images, val_model.images],
                                       feed_dict)
        summary_writer.add_summary(val_summary_str, itr)
        # Output a gif file 
        gen_images = np.transpose(np.asarray(gen_images), (1,0,2,3,4))
        images = np.transpose(np.asarray(images), (1,0,2,3,4))
        for i in range(FLAGS.batch_size):        
          npy_to_gif(gen_images[i]*255, '/cs/vml4/xca64/robot_data/gif/gen_' + str(i) + '.gif')
          npy_to_gif(images[i]*255, '/cs/vml4/xca64/robot_data/gif/org_' + str(i) + '.gif')

      if (itr) % SAVE_INTERVAL == 2:
        tf.logging.info('Saving model.')
        saver.save(sess,  os.path.join(os.path.expanduser(saver_dir), 'model' + str(itr)))

      if (itr) % SUMMARY_INTERVAL:
        summary_writer.add_summary(summary_str, itr)

    tf.logging.info('Saving model.')
    saver.save(sess, os.path.join(os.path.expanduser(saver_dir), 'model'))
    tf.logging.info('Training complete')
    tf.logging.flush()


if __name__ == '__main__':
  app.run()
