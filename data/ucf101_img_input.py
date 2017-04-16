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
IMG_WIDTH = 340
IMG_HEIGHT = 256

# Output image dimensions
OUT_WIDTH = 224
OUT_HEIGHT = 224

crop_size = []
for i in [256,224,192,168]:
    crop_size.append([0,0,i,i])
    crop_size.append([IMG_HEIGHT-i,0,i,i])
    crop_size.append([0,IMG_WIDTH-i,i,i])
    crop_size.append([IMG_HEIGHT-i,IMG_WIDTH-i,i,i])
    crop_size.append([(IMG_HEIGHT-i)/2 , (IMG_WIDTH-i)/2,i,i])


NROF_SAMPLES = 200
NROF_VAL_SAMPLES = 25
NROF_VAL_SEGMENTS = 10

data_dir = '/cs/vml4/xca64/dataset/ucf101/ucf101_imgs/'
list_dir = '/home/xca64/vml4/github/video_prediction/data/ucfTrainTestlist'

def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image



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

            file_folder = os.path.join(os.path.expanduser(data_dir), tmp[0])

            if training:
                for i in np.random.choice(int(tmp[2]),  NROF_SAMPLES):
                    fileseq.append(os.path.join(file_folder, 'frame'+str(i)+'.jpg'))
                    labelsseq.append(tmp[1])
            # For Validation Set
            else:
                # Validation set, 10*25 for each video 
                filesegs = []
                for seg in range(NROF_VAL_SEGMENTS):
                    filenames = []
                    labels = []     
                    for i in random.sample(range(int(tmp[2])),  NROF_VAL_SAMPLES):
                        filenames.append(os.path.join(file_folder, 'frame'+str(i)+'.jpg'))
                    filesegs.append(filenames)

                fileseq.append(filesegs)
                labelsseq.append(NROF_VAL_SEGMENTS*[NROF_VAL_SAMPLES*[tmp[1]]])
        
        # Validation test, with 25 images sampled from the testing set
        if not training:
            fileseq, labelsseq = np.asarray(fileseq), np.asarray(labelsseq)
            fileseq, labelsseq = np.transpose(fileseq,[1,0,2]), np.transpose(labelsseq,[1,0,2])
            fileseq, labelsseq = np.reshape(fileseq,[-1, NROF_VAL_SAMPLES]), np.reshape(labelsseq,[-1, NROF_VAL_SAMPLES])
            fileseq, labelsseq = list(fileseq), list(labelsseq)

    return fileseq, labelsseq

def build_tfrecord_input_val(training=False):
    fileseq, labelsseq = get_image_paths_and_labels(list_dir, training=training)
    input_queue = tf.train.input_producer(np.asarray(np.transpose([fileseq, labelsseq], (1,2,0))), shuffle=True)

    fileinfo = input_queue.dequeue()
    
    filenames = fileinfo[:,0]
    label = fileinfo[:,1]

    images = []

    for filename in tf.unstack(filenames):

        file_contents = tf.read_file(filename)
        image = tf.image.decode_jpeg(file_contents)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [IMG_HEIGHT, IMG_WIDTH])
        image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, 3])

        for i in range(5):
            tmp = tf.image.crop_to_bounding_box(image, crop_size[i][0], crop_size[i][1], crop_size[i][2], crop_size[i][3])
            
            tmp = tf.cast(tmp, tf.float32) / 255.0
            tmp = tf.image.per_image_standardization(tmp)    
            tmp = tf.reshape(tmp, [1, crop_size[i][2], crop_size[i][3], 3])
            tmp = tf.image.resize_bilinear(tmp, [OUT_HEIGHT, OUT_WIDTH])
            tmp = tf.reshape(tmp, [OUT_HEIGHT, OUT_WIDTH, 3])

            images.append(tmp)
            tmp = tf.image.flip_left_right(tmp)
            images.append(tmp)

    # labels = 10*25
    labels = 10*[label]
    labels = tf.reshape(labels,[NROF_VAL_SAMPLES*10,-1])

    image_batch, label_batch = tf.train.batch(
        [images, labels],
        1,
        num_threads=NROF_VAL_SAMPLES,
        capacity=FLAGS.val_capacity)

    image_batch = tf.reshape(image_batch, [NROF_VAL_SAMPLES*10, OUT_HEIGHT,OUT_WIDTH,3])
    label_batch = tf.reshape(label_batch, [NROF_VAL_SAMPLES*10])

    # import pdb
    # pdb.set_trace()

    return image_batch, label_batch

def build_tfrecord_input(training=True):

    fileseq, labelsseq = get_image_paths_and_labels(list_dir, training=training)

    input_queue = tf.train.input_producer(np.asarray(np.transpose([fileseq, labelsseq], (1,0))), shuffle=True)

    images_and_labels = []
    fileinfo = input_queue.dequeue()
    # import pdb
    # pdb.set_trace()
    # indx = random.randint(0,NROF_SAMPLES-1)
    filename = fileinfo[0]
    label = fileinfo[1]
    
    crop_size_tf = tf.constant(crop_size)
    
    file_contents = tf.read_file(filename)
    image = tf.image.decode_jpeg(file_contents)

    if training:
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [IMG_HEIGHT, IMG_WIDTH])
        image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, 3])

        # Random Flip
        image = tf.image.random_flip_left_right(image)

        # Random Crop
        #box_indx = random.randint(0,19)
        box_indx = tf.random_uniform([], minval=0, maxval=20, dtype=tf.int32) 
        image = tf.image.crop_to_bounding_box(image, crop_size_tf[box_indx][0], crop_size_tf[box_indx][1], crop_size_tf[box_indx][2], crop_size_tf[box_indx][3])

        # Reduce Mean and Cast Image
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.per_image_standardization(image)    

        # Resize to 
        image = tf.reshape(image, [1, crop_size_tf[box_indx][2], crop_size_tf[box_indx][3], 3])
        image = tf.image.resize_bilinear(image, [OUT_HEIGHT, OUT_WIDTH])

        #pylint: disable=no-member
        image = tf.reshape(image, [OUT_HEIGHT, OUT_WIDTH, 3])

    image_batch, label_batch = tf.train.batch(
            [image, label],
            FLAGS.batch_size,
            num_threads=32,
            capacity=10 * FLAGS.batch_size)
    
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
