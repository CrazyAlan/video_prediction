import numpy as np
from datetime import datetime
import tensorflow as tf
import os

from tensorflow.python.platform import app
from tensorflow.python.platform import flags



# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 400

# tf record data location:
DATA_DIR = '/home/xca64/vml4/dataset/ucf101/ucf101_imgs'

# local output directory
OUT_DIR = '/cs/vml4/xca64/robot_data/result'

# summary output dir
SUM_DIR = '/cs/vml4/xca64/robot_data/summaries'

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')
flags.DEFINE_string('dataset_name', 'ucf', 'dataset used')
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
flags.DEFINE_string('gif_dir', '/cs/vml4/xca64/robot_data/gif/' , 'directory gif result')
flags.DEFINE_integer('gif_nums', 5 , 'number of gif files to save')
flags.DEFINE_string('event_log_dir',SUM_DIR, 'directory for writing summary.')
flags.DEFINE_integer('num_iterations', 100000, 'number of training iterations.')
flags.DEFINE_string('pretrained_model', '' ,
                    'filepath of a pretrained model to initialize from.')

flags.DEFINE_integer('sequence_length', 10,
                     'sequence length, including context frames.')
flags.DEFINE_integer('nrof_classes', 101,
                     'Number of classes')
flags.DEFINE_integer('context_frames', 2, '# of frames before predictions.')
flags.DEFINE_integer('use_state', 1,
                     'Whether or not to give the state+action to the model')

flags.DEFINE_string('model', 'prednet',
                    'model architecture to use - prediction, prednet')

flags.DEFINE_string('optimizer', 'ADAM',
                    'model architecture to use - prediction, prednet')

flags.DEFINE_integer('num_masks', 10,
                     'number of masks, usually 1 for DNA, 10 for CDNA, STN.')
flags.DEFINE_float('schedsamp_k', 900.0,
                   'The k hyperparameter for scheduled sampling,'
                   '-1 for no scheduled sampling.')
flags.DEFINE_float('train_val_split', 0.95,
                   'The percentage of files to use for the training set,'
                   ' vs. the validation set.')

flags.DEFINE_float('gpu_memory_fraction', 1.0,
                   'gpu percentage')

flags.DEFINE_integer('batch_size', 32, 'batch size for training')
flags.DEFINE_float('learning_rate', 0.001,
                   'the base learning rate of the generator')

#####################
# Fine-Tuning Flags #
#####################
flags.DEFINE_string(
    'checkpoint_path', '/home/xca64/vml4/resnet/model/resnet_v1_50.ckpt',
    'The path to a checkpoint from which to fine-tune.')
flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'resnet_v1_50/logits,predictions',
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
flags.DEFINE_boolean(
    'ignore_missing_vars', True,
    'When restoring a checkpoint would ignore missing variables.')

from data.ucf101_img_input import build_tfrecord_input
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim

from model.spatial_stream import Model

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    print('Constructing models and inputs.')
    images, labels = build_tfrecord_input()
   
    global_step = tf.Variable(0, trainable=False)

    model = Model(images, labels, global_step)

    # import pdb
    # pdb.set_trace()
   
    
    print('Constructing saver.')
    # Make saver.
    saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=0)
    time_info = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    saver_dir = os.path.join(os.path.expanduser(FLAGS.output_dir), FLAGS.model, time_info, 'checkpoints')
    if not os.path.isdir(saver_dir):
      os.makedirs(saver_dir)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))        
    
    sess = tf.Session()

    summary_dir = os.path.join(os.path.expanduser(FLAGS.output_dir), FLAGS.model, time_info, 'summaries')
    if os.path.isdir(summary_dir):
      os.makedirs(summary_dir)
    summary_writer = tf.summary.FileWriter(
        summary_dir, graph=sess.graph, flush_secs=10)

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord, sess=sess)
    sess.run(tf.global_variables_initializer())

    # import pdb
    # pdb.set_trace()
    # Init from checkpoint
    if FLAGS.checkpoint_path is not None:
      model.init_from_checkpoint(sess)

    print('Start Tranning')
    # Run training.
    for itr in range(FLAGS.num_iterations):
      # import pdb
      # pdb.set_trace()
      cost, _, summary_str, acc = sess.run([model.cross_entropy, model.train_op, model.summary_op, model.accuracy])
      tf.logging.info('  In Iteration ' + str(itr) + ', Cost ' + str(cost) + ', Accuracy ' + str(acc))

      if (itr) % VAL_INTERVAL == 20:
        print('Should run the validation now')

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