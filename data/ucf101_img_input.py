import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
import random

FLAGS = flags.FLAGS

# Original image dimensions
ORIGINAL_WIDTH = 320
ORIGINAL_HEIGHT = 240
COLOR_CHAN = 3

# Default image dimensions.
IMG_WIDTH = 224
IMG_HEIGHT = 224
NROF_SAMPLES = 20

data_dir = '/cs/vml4/xca64/dataset/ucf101/ucf101_imgs/'
list_dir = '/home/xca64/vml4/github/video_prediction/data/ucfTrainTestlist'


def get_image_paths_and_labels(list_dir, split='1', training=True):
    
    fileseq = []
    labelsseq = []
    if training:
        # Load tranning samples
        train_test_list = os.path.join(os.path.expanduser(list_dir),'trainlist_with_length0' + split + '.txt')
    else:
        train_test_list = os.path.join(os.path.expanduser(list_dir),'testlist_with_length0' + split + '.txt')

    with open(train_test_list,"r") as text_file:
        lines = text_file.readlines()
        for line in lines:
            line = line.strip()
            tmp = line.split(' ')

            labelsseq.append(NROF_SAMPLES*[tmp[1]])

            file_folder = os.path.join(os.path.expanduser(data_dir), tmp[0])
            filenames = []
            for i in random.sample(range(int(tmp[2])),  NROF_SAMPLES):
                filenames.append(os.path.join(file_folder, 'frame'+str(i)+'.jpg'))
            fileseq.append(filenames)
            
    return fileseq, labelsseq

def build_tfrecord_input(training=True):

    fileseq, labelsseq = get_image_paths_and_labels(list_dir, training=training)

    input_queue = tf.train.input_producer(np.asarray(np.transpose([fileseq, labelsseq], (1,2,0))), shuffle=True)

    images_and_labels = []
    fileinfo = input_queue.dequeue()

    indx = random.randint(0,NROF_SAMPLES)
    filename = fileinfo[indx,0]
    label = fileinfo[indx,1]
    
    file_contents = tf.read_file(filename)
    image = tf.image.decode_jpeg(file_contents)

    crop_size = min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
    image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
    image = tf.reshape(image, [1, crop_size, crop_size, 3])
    image = tf.image.resize_bilinear(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    #pylint: disable=no-member
    image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, 3])
    image.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
    # images.append(image)

    image_batch, label_batch = tf.train.batch(
            [image, label],
            FLAGS.batch_size,
            num_threads=32,
            capacity=100 * FLAGS.batch_size)
    
    return image_batch, label_batch

# import pdb
# pdb.set_trace()

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# coord = tf.train.Coordinator()
# tf.train.start_queue_runners(coord=coord, sess=sess)

# imgseq = sess.run(image_batch)
# image = imgseq[0,0,:,:,:]
# print(np.shape(image))