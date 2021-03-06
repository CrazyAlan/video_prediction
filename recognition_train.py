import numpy as np
from datetime import datetime
import tensorflow as tf
import os

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

#export PYTHONPATH=/home/xca64/vml4/github/video_prediction:/home/xca64/vml4/github/video_prediction/slim

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to save a model checkpoint

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
flags.DEFINE_integer('num_iterations', 4500, 'number of training iterations.')
flags.DEFINE_integer('decay_step', 2000, 'number of steps to decrease the learning rate')
flags.DEFINE_integer('test_images', 3783, 'number of training iterations.')
flags.DEFINE_string('pretrained_model', '' ,
                    'filepath of a pretrained model to initialize from.')

flags.DEFINE_integer('sequence_length', 10,
                     'sequence length, including context frames.')

flags.DEFINE_integer('val_capacity', 5,
                     'sequence length, including context frames.')

flags.DEFINE_integer('nrof_classes', 101,
                     'Number of classes')
flags.DEFINE_integer('context_frames', 2, '# of frames before predictions.')
flags.DEFINE_integer('use_state', 1,
                     'Whether or not to give the state+action to the model')

flags.DEFINE_string('model', 'resnet_v1_50',
                    'model architecture to use - prediction, prednet')

flags.DEFINE_string('optimizer', 'MOM',
                    'model architecture to use - prediction, prednet')

flags.DEFINE_integer('num_masks', 10,
                     'number of masks, usually 1 for DNA, 10 for CDNA, STN.')
flags.DEFINE_float('schedsamp_k', 900.0,
                   'The k hyperparameter for scheduled sampling,'
                   '-1 for no scheduled sampling.')
flags.DEFINE_float('train_val_split', 0.95,
                   'The percentage of files to use for the training set,'
                   ' vs. the validation set.')

flags.DEFINE_float('weight_decay', 0.0001,
                   'Regularizer weight decay')

flags.DEFINE_float('batch_norm_decay', 0.997,
                   'Batch norm decay')

flags.DEFINE_float('gpu_memory_fraction', 0.5,
                   'gpu percentage')

flags.DEFINE_integer('batch_size', 64, 'batch size for training')
flags.DEFINE_integer('nrof_accum_batch', 1, 'number of times to accumulate the gradients')
flags.DEFINE_integer('sub_batch_size', 8, 'batch size for training')

flags.DEFINE_integer('print_interval', 10, 'print_interval')
flags.DEFINE_integer('VAL_INTERVAL', 1000, 'Validation Start')
flags.DEFINE_integer('val_start',FLAGS.VAL_INTERVAL/2 , 'Validation Start')
flags.DEFINE_integer('SAVE_INTERVAL', 2500, 'Save Interval')
flags.DEFINE_integer('SUMMARY_INTERVAL', 40, 'Save Interval')



flags.DEFINE_float('learning_rate', 0.001,
                   'the base learning rate of the generator')

#####################
# Fine-Tuning Flags #
#####################
flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
flags.DEFINE_boolean(
    'ignore_missing_vars', True,
    'When restoring a checkpoint would ignore missing variables.')
flags.DEFINE_boolean(
    'partial_bn', False,
    'Whether or not using partial batch_norm')


from data.ucf101_img_input import build_tfrecord_input
from data.ucf101_img_input import build_tfrecord_input_val
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim

from model.spatial_stream import Model

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    print('Constructing models and inputs.')
    
    global_step = tf.Variable(0, trainable=False)

    images, labels = build_tfrecord_input(training=True)
    model = Model(images, labels, global_step)

    val_images, val_labels = build_tfrecord_input_val(training=False)
    val_model = Model(val_images, val_labels, global_step, reuse=True)
   
    
    print('Constructing saver.')
    time_info = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    base_dir = os.path.join(os.path.expanduser(FLAGS.output_dir), FLAGS.model, time_info)
    if not os.path.isdir(base_dir):
      os.makedirs(base_dir)

    with open(os.path.join(base_dir, 'params.txt'), 'w') as f:
        for key, value in FLAGS.__flags.iteritems():
            f.write(str(key) + ':' + str(value) + '\n')    
    # Make saver.
    saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=0)
    saver_dir = os.path.join(base_dir, 'checkpoints')
    if not os.path.isdir(saver_dir):
      os.makedirs(saver_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))        
    
    # sess = tf.Session()

    summary_dir = os.path.join(base_dir, 'summaries')
    if os.path.isdir(summary_dir):
      os.makedirs(summary_dir)
    summary_writer = tf.summary.FileWriter(
        summary_dir, graph=sess.graph, flush_secs=10)

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord, sess=sess)
    sess.run(tf.global_variables_initializer())

    if FLAGS.checkpoint_path is not None:
      model.init_from_checkpoint(sess)

    print('Start Tranning')
    # Run training.
    for itr in range(0,FLAGS.num_iterations):

      # Running Training, accumunate grads before update
      _ = sess.run([model.zero_ops])
      train_cost = []
      summary_str = None
      train_acc = []
      learning_rate = None 
      for inner_itr in range(FLAGS.nrof_accum_batch):
        cost, _,  summary_str, acc, learning_rate = sess.run([model.cross_entropy, model.accum_ops, model.summary_op, model.accuracy, model.learning_rate])
        train_cost.append(cost)
        train_acc.append(acc)
      # Update Gradient 
      _ = sess.run([model.train_op])

      if itr % FLAGS.print_interval == 0:
        tf.logging.info('  In Iteration ' + str(itr) + ', Cost ' + str(np.mean(train_cost)) + ', Accuracy ' + str(np.mean(train_acc)) + ', Learning Rate is ' + str(learning_rate))

      if (itr) % FLAGS.VAL_INTERVAL ==  FLAGS.val_start:
        print('Running Validation Now')
        val_acc = []
        for val_itr in range(FLAGS.test_images):
          summary_str, acc = sess.run([val_model.summary_op, val_model.accuracy])
          val_acc.append(acc)
          if val_itr % FLAGS.print_interval == 0:
            tf.logging.info('In Training Iteration ' + str(itr) + ',  In Val Iteration ' + str(val_itr) + ', acc ' + str(acc) + ' , Accuracy ' + str(np.mean(val_acc)))

      if (itr) % FLAGS.SAVE_INTERVAL == FLAGS.SAVE_INTERVAL-20:
        tf.logging.info('Saving model.')
        saver.save(sess,  os.path.join(os.path.expanduser(saver_dir), 'model' + str(itr)))

      if (itr) % FLAGS.SUMMARY_INTERVAL:
        summary_writer.add_summary(summary_str, itr)

    tf.logging.info('Saving model.')
    saver.save(sess, os.path.join(os.path.expanduser(saver_dir), 'model'))
    tf.logging.info('Training complete')
    #tf.logging.flush()
    ## Final Validation
    print('Running Final Validation Now')
    val_acc = []
    for val_itr in range(FLAGS.test_images):
      acc = sess.run([val_model.accuracy])
      val_acc.append(acc)
      if val_itr % 10 == 0:
        tf.logging.info('In Training Iteration ' + str(itr) + ',  In Val Iteration ' + str(val_itr) + ', acc ' + str(acc) + ' , Accuracy ' + str(np.mean(val_acc)))
    tf.logging.info('In Training Iteration ' + str(itr) + ',  In Val Iteration ' + str(val_itr) + ', Accuracy ' + str(np.mean(val_acc)))



if __name__ == '__main__':
  app.run()
