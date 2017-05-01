import os
import scipy.io
import scipy.misc
import numpy as np
from time import gmtime, strftime
from numpy.random import choice
import h5py
import hdf5storage
class Loader(object):
    """docstring for Loader"""
    def __init__(self):
        
        self.datadir = '/cs/vml4/xca64/github/visual-analogy-tensorflow/data';
        self.batchsize = 20
        self.width = 60
        self.height = 60
        self.dim = 3
        self.mat_path = os.path.join(os.path.expanduser(self.datadir), 'sprites/sprites_splits.mat')
        self.splits = scipy.io.loadmat(self.mat_path)
        self.trainidx = self.splits['trainidx'][0]

        self.trainfiles=[]
        for idx in self.trainidx:
            path = os.path.join(os.path.expanduser(self.datadir),'sprites', 'sprites_'+str(idx)+'.mat')
            self.trainfiles.append(path)

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

    def next(self, set_option=None):
        return self.sample_analogy_add(self.trainfiles)

    def next_test(self, set_option=None):
        return self.sample_analogy_add(self.trainfiles, set_option)

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

        t1_idx = choice(range(np.shape(mask2)[1]), self.batchsize, replace=True)
        t2_idx = t1_idx[np.random.permutation(len(t1_idx))]

        batch_sprites['X1'] = np.reshape(sprite1[:,t1_idx], (self.width, self.height, self.dim, len(t1_idx)), order='F')
        batch_sprites['X2'] = np.reshape(sprite1[:,t2_idx], (self.width, self.height, self.dim, len(t2_idx)), order='F')
        batch_sprites['X3'] = np.reshape(sprite2[:,t1_idx], (self.width, self.height, self.dim, len(t1_idx)), order='F')
        batch_sprites['X4'] = np.reshape(sprite2[:,t2_idx], (self.width, self.height, self.dim, len(t2_idx)), order='F')

        batch_masks['X1'] = np.reshape(mask1[:,t1_idx], (self.width, self.height, 1, len(t1_idx)), order='F')
        batch_masks['X2'] = np.reshape(mask1[:,t2_idx], (self.width, self.height, 1, len(t2_idx)), order='F')
        batch_masks['X3'] = np.reshape(mask2[:,t1_idx], (self.width, self.height, 1, len(t1_idx)), order='F')
        batch_masks['X4'] = np.reshape(mask2[:,t2_idx], (self.width, self.height, 1, len(t2_idx)), order='F')

        for key in batch_sprites:
            batch_sprites[key] = np.transpose(batch_sprites[key], (3,0,1,2))
            batch_masks[key] = np.transpose(batch_masks[key], (3,0,1,2))

        return batch_labels, batch_sprites, batch_masks

if __name__ == '__main__':
    tmp = Loader()
    # import pdb
    # pdb.set_trace()
    batch_labels, batch_sprites, batch_masks = tmp.next()


