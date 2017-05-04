import numpy as np
from datetime import datetime
import tensorflow as tf
import os

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils.img_saving import imsave
from utils.img_saving import merge

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

flags.DEFINE_float('lambda_img', 2e-3,
                   'image reconstruction loss percentage')
flags.DEFINE_float('lambda_adv', 10,
                   'adversrial loss percentage')
flags.DEFINE_float('lambda_feat', 0.01,
                   'feature loss percentage')

flags.DEFINE_integer('batch_size', 250, 'batch size for training')
flags.DEFINE_integer('val_iterations', 5, 'batch size for training')
flags.DEFINE_integer('print_interval', 10, 'print_interval')
flags.DEFINE_integer('VAL_INTERVAL', 2500, 'Validation Start')
flags.DEFINE_integer('val_start',FLAGS.VAL_INTERVAL/2 , 'Validation Start')
flags.DEFINE_integer('SAVE_INTERVAL', 2500, 'Save Interval')
flags.DEFINE_integer('SUMMARY_INTERVAL', 40, 'Save Interval')



flags.DEFINE_float('learning_rate', 0.0002,
                   'the base learning rate of the generator')
flags.DEFINE_float('lambda_rgb', 1,
                   'the base learning rate of the generator')
flags.DEFINE_float('lambda_mask', 0.1,
                   'the base learning rate of the generator')

flags.DEFINE_boolean(
    'kl_loss', True,
    'Whether or not using kl_loss in vae')

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

from model.vae import Model

height = 60
width = 60
dim = 3

# def update_train_op():
#   if discr_loss_ratio < 1e-1 and TRAIN_DIS:
#     TRAIN_DIS = False
#     TRAIN_GEN = True
#   if discr_loss_ratio > 5e-1 and not TRAIN_DIS:
#     TRAIN_DIS = True
#     TRAIN_GEN = True
#   if discr_loss_ratio > 1e1 and TRAIN_GEN:
#     TRAIN_GEN = False
#     TRAIN_DIS = True

# def _update_scope(model, TRAIN_GEN, TRAIN_DIS):
#   train_ops = [model.gen_loss,
#                 model.recon_loss,
#                 model.disc_acc, 
#                 model.summary_op, 
#                 model.learning_rate,
#                 model.discr_loss_ratio,
#                 model.other_loss]
#   if TRAIN_GEN and TRAIN_DIS:
#     train_ops += [model.update_disc_op,
#                  model.update_gen_op,
#                  model.update_enc_op]
#   elif TRAIN_GEN:

#     train_ops += [model.update_gen_op,
#                   model.update_enc_op]

#   elif TRAIN_DIS:

#     train_ops += [model.update_disc_op,
#                   model.update_enc_op]
  
#   return train_ops

def main(unused_argv):

  TRAIN_ENC = True
  TRAIN_GEN = True
  TRAIN_DIS = True
  discr_loss_ratio = 1


  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    print('Constructing models and inputs.')
    
    global_step = tf.Variable(0, trainable=False)

    batch_sprites_holder = tf.placeholder(tf.float32, shape=(4, None, height, width, dim))
    batch_masks_holder = tf.placeholder(tf.float32, shape=(4, None,  height, width, 1))
    model = Model(batch_sprites_holder, batch_masks_holder, global_step, is_training=True)
    model.Build()

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

      _, kl_loss, loss, summary_str, pred_comb = sess.run([model.train_op, model.kl_loss,\
                                  model.loss, model.summary_op, model.pred_comb],\
                                  feed_dict ={
                                  batch_sprites_holder : batch_sprites, 
                                  batch_masks_holder: batch_masks})

      if itr % FLAGS.print_interval == 0:
        log_str = ('In iteration {itr}, kl_loss: {kl_loss},'\
          ' loss: {loss}').format(itr=itr,
                                  kl_loss=str(kl_loss),
                                  loss=str(loss))

        tf.logging.info(log_str)

      if (itr) % FLAGS.VAL_INTERVAL ==  FLAGS.val_start:
        print('Running Validation Now')
        # for val_itr in range(FLAGS.val_iterations):
        #   batch_sprites_val, batch_masks_val = loader.next()

        #   # import pdb
        #   # pdb.set_trace()

        #  _, kl_loss, loss, summary_str = sess.run([model.train_op, model.kl_loss,\
        #                                            model.loss, model.summary_op]\
        #                                             feed_dict ={
        #                                             batch_sprites_holder : batch_sprites_val, 
        #                                             batch_masks_holder: batch_masks_val})

        #   if val_itr % FLAGS.print_interval == 0:
        #     tf.logging.info('In Training Iteration ' + str(itr) + ',  In Val Iteration ' + str(val_itr) 
        #                     + ', Cost ' + str(val_cost))

        mrg_img = merge(zip(*[batch_sprites[0], batch_sprites[1], batch_sprites[2], batch_sprites[3], predictions[0], pred_comb]))
        # path = os.path.join(gif_dir, str(itr) + '_' + 'val'+ '_' + str(val_itr) +'.png')
        imsave(path, mrg_img)

      # if (itr) % FLAGS.SAVE_INTERVAL == FLAGS.SAVE_INTERVAL-20:
      #   tf.logging.info('Saving model.')
      #   saver.save(sess,  os.path.join(os.path.expanduser(saver_dir), 'model' + str(itr)))

      # if (itr) % FLAGS.SUMMARY_INTERVAL:
      #   summary_writer.add_summary(summary_str, itr)

    tf.logging.info('Saving model.')
    saver.save(sess, os.path.join(os.path.expanduser(saver_dir), 'model'))
    tf.logging.info('Training complete')

if __name__ == '__main__':
  app.run()
