import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile


FLAGS = flags.FLAGS

# Original image dimensions
ORIGINAL_WIDTH = 720
ORIGINAL_HEIGHT = 480
COLOR_CHAN = 3

# Default image dimensions.
IMG_WIDTH = 160
IMG_HEIGHT = 128

# Dimension of the state and action.
STATE_DIM = 5


data_dir = '/home/xca64/vml4/dataset/ucf_sports_actions/ucf_action'


def get_image_paths_and_labels(data_dir, training=True):
    labels = os.listdir(data_dir)
    fileseq = []
    labelsseq = []
    for label in labels:
        folder_names = gfile.Glob(os.path.join(data_dir, label, '*'))
        for folder in folder_names:
            imgs = gfile.Glob(os.path.join(folder, '*.jpg'))
            nrof_chuncks = len(imgs)/FLAGS.sequence_length
            if nrof_chuncks != 0:
                imgs = np.asarray(imgs[:nrof_chuncks*FLAGS.sequence_length])
                lbs = np.asarray(len(imgs)*[label])
                fileseq += np.split(imgs,nrof_chuncks)
                labelsseq += np.split(lbs, nrof_chuncks)

    index = int(np.floor(FLAGS.train_val_split * len(fileseq)))
    if training:
        fileseq = fileseq[:index]
        labelsseq = labelsseq[:index]
    else:
        fileseq = fileseq[index:]   
        labelsseq = labelsseq[index:]


    return fileseq, labelsseq

def build_tfrecord_input(training=True):

    fileseq, labelsseq = get_image_paths_and_labels(data_dir, training=training)

    input_queue = tf.train.input_producer(np.asarray(np.transpose([fileseq, labelsseq], (1,2,0))), shuffle=True)

    images_and_labels = []
    fileinfo = input_queue.dequeue()
    filenames = fileinfo[:,0]
    labels = fileinfo[:,1]
    images = []
    for filename in tf.unstack(filenames):
        file_contents = tf.read_file(filename)
        image = tf.image.decode_jpeg(file_contents)

        crop_size = min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
        image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
        image = tf.reshape(image, [1, crop_size, crop_size, 3])
        image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
        image = tf.cast(image, tf.float32) / 255.0
        #pylint: disable=no-member
        image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, 3])
        image.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
        images.append(image)

    image_batch, labels_batch = tf.train.batch(
            [images, labels],
            FLAGS.batch_size,
            num_threads=FLAGS.batch_size,
            capacity=100 * FLAGS.batch_size)
    zeros_batch = tf.zeros([FLAGS.batch_size, FLAGS.sequence_length, STATE_DIM])
    
    return image_batch, zeros_batch, zeros_batch

# import pdb
# pdb.set_trace()

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# coord = tf.train.Coordinator()
# tf.train.start_queue_runners(coord=coord, sess=sess)

# imgseq = sess.run(image_batch)
# image = imgseq[0,0,:,:,:]
# print(np.shape(image))