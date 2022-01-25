from Bio import SeqIO
from pathlib import Path
import preprocess as pp
import h5py
import ray
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


ds_path = '/home/gsukhorukov/classifier/data/train/encoded_train_500.hdf5'

f = h5py.File(ds_path, "r")
fragments = f["fragments"]
fragments_rc = f["fragments_rc"]
labels = f["labels"]

new_fragments = np.concatenate((fragments[...], fragments_rc[...]))
new_fragments_rc = np.concatenate((fragments_rc[...], fragments[...]))
new_labels = np.concatenate((labels[...], labels[...]))
print(fragments[...].shape[0], new_fragments_rc.shape[0], new_fragments.shape[0], new_labels.shape[0])

out_path = '/home/gsukhorukov/classifier/data/train/encoded_train_500_fw_and_rvc.hdf5'
pp.storing_encoded(new_fragments, new_fragments_rc, new_labels, out_path,)
