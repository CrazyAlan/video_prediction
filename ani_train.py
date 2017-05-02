import numpy as np
from datetime import datetime
import tensorflow as tf
import os

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils.img_saving import imsave
#export PYTHONPATH=/home/xca64/vml4/github/video_prediction:/home/xca64/vml4/github/video_prediction/slim

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to save a model checkpoint


# local output directory
OUT_DIR = '/cs/vml4/xca64/robot_data/result'

# summary output dir
SUM_DIR = '/cs/vml4/xca64/robot_data/summaries'

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_name', 'sprite', 'dataset used')
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
flags.DEFINE_string('gif_dir', '/cs/vml4/xca64/robot_data/gif/' , 'directory gif result')
flags.DEFINE_integer('gif_nums', 5 , 'number of gif files to save')
flags.DEFINE_string('event_log_dir',SUM_DIR, 'directory for writing summary.')
flags.DEFINE_integer('num_iterations', 10000, 'number of training iterations.')
flags.DEFINE_integer('decay_step', 5000, 'number of steps to decrease the learning rate')
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

flags.DEFINE_string('model', 'ana_sprite_3',
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

flags.DEFINE_float('weight_decay', 0.0001,
                   'Regularizer weight decay')

flags.DEFINE_float('batch_norm_decay', 0.997,
                   'Batch norm decay')

flags.DEFINE_float('gpu_memory_fraction', 0.5,
                   'gpu percentage')

flags.DEFINE_integer('batch_size', 25, 'batch size for training')
flags.DEFINE_integer('val_iterations', 5, 'batch size for training')
flags.DEFINE_integer('print_interval', 10, 'print_interval')
flags.DEFINE_integer('VAL_INTERVAL', 2500, 'Validation Start')
flags.DEFINE_integer('val_start',FLAGS.VAL_INTERVAL/2 , 'Validation Start')
flags.DEFINE_integer('SAVE_INTERVAL', 2500, 'Save Interval')
flags.DEFINE_integer('SUMMARY_INTERVAL', 40, 'Save Interval')



flags.DEFINE_float('learning_rate', 0.001,
                   'the base learning rate of the generator')
flags.DEFINE_float('lambda_rgb', 0.01,
                   'the base learning rate of the generator')
flags.DEFINE_float('lambda_mask', 0.001,
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


from data.load_ani import Loader

from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim

from model.ani import Model

height = 60
width = 60
dim = 3
def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    print('Constructing models and inputs.')
    
    global_step = tf.Variable(0, trainable=False)

    batch_sprites_holder = tf.placeholder(tf.float32, shape=(4, None, height, width, dim))
    batch_masks_holder = tf.placeholder(tf.float32, shape=(4, None,  height, width, 1))
    model = Model(batch_sprites_holder, batch_masks_holder, global_step)

    model_val = Model(batch_sprites_holder, batch_masks_holder, global_step, reuse=True)

       
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
    if not os.path.isdir(summary_dir):
      os.makedirs(summary_dir)
    summary_writer = tf.summary.FileWriter(
        summary_dir, graph=sess.graph, flush_secs=10)

    gif_dir = os.path.join(base_dir, 'gif')
    if not os.path.isdir(gif_dir):
      os.makedirs(gif_dir)


    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord, sess=sess)
    sess.run(tf.global_variables_initializer())

    if FLAGS.checkpoint_path is not None:
      model.init_from_checkpoint(sess)

    # Data loader
    loader = Loader()
    print('Start Tranning')
    # Run training.
    for itr in range(0,FLAGS.num_iterations):

      # Running Training, accumunate grads before update
      batch_sprites, batch_masks = loader.next()

      train_cost, _,  summary_str, learning_rate = sess.run([model.cost, model.train_op, model.summary_op, model.learning_rate], feed_dict ={
                                                              batch_sprites_holder : batch_sprites, 
                                                              batch_masks_holder: batch_masks})
      
      if itr % FLAGS.print_interval == 0:
        tf.logging.info('  In Iteration ' + str(itr) + ', Cost ' + str(np.mean(train_cost)) + ', Learning Rate is ' + str(learning_rate))

      if (itr) % FLAGS.VAL_INTERVAL ==  FLAGS.val_start:
        print('Running Validation Now')
        for val_itr in range(FLAGS.val_iterations):
          batch_sprites_val, batch_masks_val = loader.next_val()

          val_cost, summary_str, predictions = sess.run([model_val.cost, model_val.summary_op, model_val.predict], feed_dict ={
                                                              batch_sprites_holder : batch_sprites_val, 
                                                              batch_masks_holder: batch_masks_val})

          if val_itr % FLAGS.print_interval == 0:
            tf.logging.info('In Training Iteration ' + str(itr) + ',  In Val Iteration ' + str(val_itr) 
                            + ', Cost ' + str(val_cost))
        # save the last image
        for img_id, img in enumerate(predictions[0]):
          path = os.path.join(gif_dir, 'pre_' + str(itr) + '_' + str(img_id)+'.png')
          imsave(path, img)

        for img_id, img in enumerate(batch_sprites_val[3]):
          path = os.path.join(gif_dir, 'rel_' + str(itr) + '_' + str(img_id)+'.png')
          imsave(path, img)

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
    # print('Running Final Validation Now')
    # val_acc = []
    # for val_itr in range(FLAGS.test_images):
    #   acc = sess.run([val_model.accuracy])
    #   val_acc.append(acc)
    #   if val_itr % 10 == 0:
    #     tf.logging.info('In Training Iteration ' + str(itr) + ',  In Val Iteration ' + str(val_itr) + ', acc ' + str(acc) + ' , Accuracy ' + str(np.mean(val_acc)))
    # tf.logging.info('In Training Iteration ' + str(itr) + ',  In Val Iteration ' + str(val_itr) + ', Accuracy ' + str(np.mean(val_acc)))



if __name__ == '__main__':
  app.run()
