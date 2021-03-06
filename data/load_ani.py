import os
import scipy.io
import scipy.misc
import numpy as np
from time import gmtime, strftime
from numpy.random import choice
import h5py
import hdf5storage
import shutil

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class Loader(object):
    """docstring for Loader"""
    def __init__(self):
        
        self.datadir = '/cs/vml4/xca64/github/visual-analogy-tensorflow/data';
        # Copy data to local-scrath, if it doesn't exist
        local_scratch_dir='/local-scratch/xca64/video_prediction/'
        if not os.path.isdir(local_scratch_dir):
            print('Copy data file to local-scratch')
            #os.makedirs(local_scratch_dir)
            shutil.copytree(self.datadir, local_scratch_dir)
            self.datadir=local_scratch_dir
        else:
            print('File exist on local-scratch, dont need to copy')
            self.datadir=local_scratch_dir
        
        self.batch_size = 25
        self.width = 60
        self.height = 60
        self.dim = 3
        self.mat_path = os.path.join(os.path.expanduser(self.datadir), 'sprites/sprites_splits.mat')
        self.splits = scipy.io.loadmat(self.mat_path)
        self.trainidx = self.splits['trainidx'][0]
        self.validx = self.splits['validx'][0]
        self.testidx = self.splits['testidx'][0] 

        self.trainfiles=[]
        self.valfiles=[]
        self.testfiles=[]

        for idx in self.trainidx:
            path = os.path.join(os.path.expanduser(self.datadir),'sprites', 'sprites_'+str(idx-1)+'.mat')
            self.trainfiles.append(path)

        for idx in self.validx:
            path = os.path.join(os.path.expanduser(self.datadir),'sprites', 'sprites_'+str(idx-1)+'.mat')
            self.valfiles.append(path)
        
        for idx in self.testidx:
            path = os.path.join(os.path.expanduser(self.datadir),'sprites', 'sprites_'+str(idx-1)+'.mat')
            self.testfiles.append(path)
                        

    def fix_wpn(self, label, idx_anim):
        L = label
        if L[6]==0:
            if ((idx_anim >= 17) and (idx_anim <= 20)):
                L[6] = 1
            else:
                L[6] = 0
        elif L[6]==1:
            if ((idx_anim >= 5) and (idx_anim <= 12)):
                L[6] = 2
            else:
                L[6] = 0
        return L

    def _batch_file(self, filenames):
        # Running Training, accumunate grads before update
        batch_sprites_dict_list = []
        batch_masks_dict_list = []
        for i in range(FLAGS.batch_size / self.batch_size):
            _, batch_sprites_dict, batch_masks_dict = self.sample_analogy_add(filenames)
            
            batch_sprites_dict_list.append(batch_sprites_dict)
            batch_masks_dict_list.append(batch_masks_dict)

        batch_sprites = []
        batch_masks = []
        for key in ['X1', 'X2', 'X3', 'X4']:
            tmp_sprites = []
            tmp_masks = []
            for i in range(len(batch_sprites_dict_list)):           
                tmp_sprites += list(batch_sprites_dict_list[i][key])
                tmp_masks += list(batch_masks_dict_list[i][key])
            batch_sprites.append(tmp_sprites)
            batch_masks.append(tmp_masks)

        return np.array(batch_sprites), np.array(batch_masks)

    def _batch_file_sample(self, filenames):
        # Running Training, accumunate grads before update
        batch_sprites_dict_list = []
        batch_masks_dict_list = []
        for i in range(FLAGS.batch_size / self.batch_size):
            _, batch_sprites_dict, batch_masks_dict = self.sample_analogy_add(filenames)
            
            
            batch_sprites_dict_list.append(batch_sprites_dict)
            batch_masks_dict_list.append(batch_masks_dict)

        batch_sprites = []
        batch_masks = []
        for key in ['X1', 'X2', 'X3', 'X4']:
            tmp_sprites = []
            tmp_masks = []
            for i in range(len(batch_sprites_dict_list)): 
                if key=='X3':
                    tmp_sprites += self.batch_size*[list(batch_sprites_dict_list[i][key][0])]
                else:
                    tmp_sprites += list(batch_sprites_dict_list[i][key])
                tmp_masks += list(batch_masks_dict_list[i][key])
            batch_sprites.append(tmp_sprites)
            batch_masks.append(tmp_masks)

        return np.array(batch_sprites), np.array(batch_masks)


    def next(self, set_option=None):
        # Running Training, accumunate grads before update
        return self._batch_file(self.trainfiles)

    def next_val(self, set_option=None):
        # Running Training, accumunate grads before update
        return self._batch_file(self.valfiles)

    def next_test(self, set_option=None):
        # Running Training, accumunate grads before update
        return self._batch_file(self.testfiles)

    def next_sample(self, set_option=None):
        # Running Training, accumunate grads before update
        tmp = self.batch_size
        self.batch_size = 5
        result = self._batch_file_sample(self.valfiles)
        self.batch_size = tmp
        return result

    def sample_analogy_add(self, files, pars=None):
        batch_labels = {}
        batch_sprites = {}
        batch_masks = {}


        random_idx = choice(range(len(files)), 2, replace=False)
        idx1 = random_idx[0];
        idx2 = random_idx[1];
            
        data1 = hdf5storage.loadmat(files[idx1])
        data2 = hdf5storage.loadmat(files[idx2])
            
        batch_labels['L1'] = data1['labels'][0]
        batch_labels['L2'] = data1['labels'][0]
        batch_labels['L3'] = data2['labels'][0]
        batch_labels['L4'] = data2['labels'][0]


        anim1_sprite = data1['sprites'][0]
        anim1_mask = data1['masks'][0]
        anim2_sprite = data2['sprites'][0]
        anim2_mask = data2['masks'][0]

        idx_anim = np.random.randint(len(anim1_sprite))

        batch_labels['L1'] = self.fix_wpn(batch_labels['L1'], idx_anim)
        batch_labels['L2'] = self.fix_wpn(batch_labels['L2'], idx_anim)
        batch_labels['L3'] = self.fix_wpn(batch_labels['L3'], idx_anim)
        batch_labels['L4'] = self.fix_wpn(batch_labels['L4'], idx_anim)

        sprite1 = np.array(anim1_sprite[idx_anim])
        mask1 = np.array(anim1_mask[idx_anim])
        sprite2 = np.array(anim2_sprite[idx_anim])
        mask2 = np.array(anim2_mask[idx_anim])

        t1_idx = choice(range(np.shape(mask2)[1]), self.batch_size, replace=True)
        t2_idx = t1_idx[np.random.permutation(len(t1_idx))]

        batch_sprites['X1'] = np.reshape(sprite1[:,t1_idx], (self.width, self.height, self.dim, len(t1_idx)), order='F')
        batch_sprites['X2'] = np.reshape(sprite1[:,t2_idx], (self.width, self.height, self.dim, len(t2_idx)), order='F')
        batch_sprites['X3'] = np.reshape(sprite2[:,t1_idx], (self.width, self.height, self.dim, len(t1_idx)), order='F')
        batch_sprites['X4'] = np.reshape(sprite2[:,t2_idx], (self.width, self.height, self.dim, len(t2_idx)), order='F')

        batch_masks['X1'] = np.reshape(mask1[:,t1_idx], (self.width, self.height, 1, len(t1_idx)), order='F')
        batch_masks['X2'] = np.reshape(mask1[:,t2_idx], (self.width, self.height, 1, len(t2_idx)), order='F')
        batch_masks['X3'] = np.reshape(mask2[:,t1_idx], (self.width, self.height, 1, len(t1_idx)), order='F')
        batch_masks['X4'] = np.reshape(mask2[:,t2_idx], (self.width, self.height, 1, len(t2_idx)), order='F')

        for key in ['X1', 'X2', 'X3', 'X4']:
            
            batch_sprites[key] = np.transpose(batch_sprites[key], (3,0,1,2))
            batch_masks[key] = np.transpose(batch_masks[key], (3,0,1,2))

        return batch_labels, batch_sprites, batch_masks

if __name__ == '__main__':
    tmp = Loader()
    # import pdb
    # pdb.set_trace()
    batch_sprites, batch_masks = tmp.next()


