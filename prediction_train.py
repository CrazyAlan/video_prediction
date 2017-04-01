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

from model.prednet import Model

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




import moviepy.editor as mpy
def npy_to_gif(npy, filename):
     clip = mpy.ImageSequenceClip(list(npy), fps=10)
     clip.write_gif(filename)



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
      
      if (itr) % VAL_INTERVAL == 20:
        # Run through validation set.
        feed_dict = {val_model.lr: 0.0,
                     val_model.prefix: 'val',
                     val_model.iter_num: np.float32(itr)}
        _, val_summary_str, gen_images, images = sess.run([val_model.train_op, val_model.summ_op, val_model.gen_images, val_model.images],
                                       feed_dict)
        summary_writer.add_summary(val_summary_str, itr)
        # Output a gif file 
        gen_images = np.transpose(np.asarray(gen_images[FLAGS.context_frames - 1:]), (1,0,2,3,4))
        images = np.transpose(np.asarray(images), (1,0,2,3,4))
        for i in range(5):        
          npy_to_gif(gen_images[i]*255, '/cs/vml4/xca64/robot_data/gif/gen_' + str(i) + '.gif')
          npy_to_gif(images[i]*255, '/cs/vml4/xca64/robot_data/gif/org_' + str(i) + '.gif')

      if (itr) % SAVE_INTERVAL == 20:
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
