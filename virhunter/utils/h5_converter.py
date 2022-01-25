import h5py
import numpy as np
import random


path_dat = "/home/gsukhorukov/classifier/data/train_ds_sliding/encoded-fragments_tomato_500_3cl.npy"
path_lab = "/home/gsukhorukov/classifier/data/train_ds_sliding/labels_tomato_500_3cl.npy"
# path_fasta = "/home/gsukhorukov/classifier/data/train_ds_sliding/seqs_tomato_1000_3cl.fasta"
path_h5 = "/home/gsukhorukov/classifier/data/train_ds_sliding/encoded_tomato_500_3cl.hdf5"

f = h5py.File(path_h5, "w")
seqs = np.load(path_dat)
labels = np.load(path_lab)
f.create_dataset("sequences", data=seqs)
f.create_dataset("labels", data=labels)
del seqs, labels, f
#
# f = h5py.File(path_h5, "r")
# dset = f["sequences"]
# l = dset.shape
# print(l)
# indexes = [i for i in range(l[0])]
# random.shuffle(indexes)
# for i in range(10000):
#     subset = indexes[i*64:(i+1)*64]
#     subset.sort()
#     print(subset)
#     z = dset[subset, ...]
#     print(z.shape)
