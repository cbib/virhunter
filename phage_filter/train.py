#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Credits: Grigorii Sukhorukov, Macha Nikolski
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

import fire
import yaml
import tensorflow as tf
import numpy as np
import h5py
import random
from pathlib import Path
import pandas as pd
from joblib import dump, load
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils.batch_loader import BatchLoader, BatchGenerator
from models import model_10


def fetch_batches(fragments, fragments_rc, labels, random_seed, batch_size):
    # for reverse contigs we use half a batch size, as the second half is filled with reverse fragments
    batch_size_ = int(batch_size / 2)
    random.seed(a=random_seed)
    n_seqs = int(fragments.shape[0] / 2)
    ind_0 = shuffle(list(range(n_seqs)), random_state=random.randrange(1000000))
    ind_1 = shuffle(list(range(n_seqs, 2 * n_seqs)), random_state=random.randrange(1000000))
    batches = []
    for i in range(n_seqs // batch_size_):
        batch_idx = []
        batch_idx.extend(ind_0[i * batch_size_:(i + 1) * batch_size_])
        batch_idx.extend(ind_1[i * batch_size_:(i + 1) * batch_size_])
        batches.append(batch_idx)
    train_batches = batches[:int(len(batches) * 0.9)]
    val_batches = batches[int(len(batches) * 0.9):]
    train_gen = BatchLoader(fragments, fragments_rc, labels, train_batches, rc=True,
                            random_seed=random.randrange(1000000))
    val_gen = BatchLoader(fragments, fragments_rc, labels, val_batches, rc=True, random_seed=random.randrange(1000000))
    return train_gen, val_gen


def train_nn(
        ds_path,
        out_path,
        length,
        epochs=10,
        batch_size=256,
        random_seed=None,
):
    assert Path(out_path).is_dir(), 'out_path was not provided correctly'
    # initializing random generator with the random seed
    random.seed(a=random_seed)
    # callbacks for training of models
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto',
            min_delta=0.000001, cooldown=3, ),
    ]
    try:
        f = h5py.File(Path(ds_path, f"encoded_phage_train_{length}.hdf5"), "r")
        fragments = f["fragments"]
        fragments_rc = f["fragments_rc"]
        labels = f["labels"]
    except FileNotFoundError:
        raise Exception("dataset was not found. Change ds_path or launch prepare_ds script")
    print(f'using {random_seed} random_seed in batch generation')
    print(np.shape(fragments))
    train_gen, val_gen = fetch_batches(fragments,
                                       fragments_rc,
                                       labels,
                                       random_seed=random_seed,
                                       batch_size=batch_size,)

    model = model_10.model(length)
    model.fit(x=train_gen,
              validation_data=val_gen,
              epochs=epochs,
              callbacks=callbacks,
              batch_size=batch_size,
              verbose=1)
    # taking into account validation data
    model.fit(x=val_gen,
              epochs=1,
              batch_size=batch_size,
              verbose=2)
    model.save_weights(Path(out_path, "model_10.h5"))
    print('finished training model_10 network')




def train(config):
    with open(config, "r") as yamlfile:
        cf = yaml.load(yamlfile, Loader=yaml.FullLoader)

    assert Path(cf["train"]["ds_path"]).exists(), f'{cf["prepare_ds"]["ds_path"]} does not exist'

    Path(cf["train"]["out_path"], "500").mkdir(parents=True, exist_ok=True)
    Path(cf["train"]["out_path"], "1000").mkdir(parents=True, exist_ok=True)

    for l_ in 500, 1000:
        train_nn(
            ds_path=Path(cf["train"]["ds_path"], f"{l_}"),
            out_path=Path(cf["train"]["out_path"], f"{l_}"),
            length=l_,
            epochs=cf["train"]["epochs"],
            random_seed=cf["train"]["random_seed"],
        )
        print(f"finished training NN  for {l_} fragment size\n")
    print(f"NN weights are stored in {cf['train']['out_path']}")


if __name__ == '__main__':
    fire.Fire(train)
