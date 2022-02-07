#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Credits: Grigorii Sukhorukov, Macha Nikolski
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import fire
import yaml
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import h5py
import random
from pathlib import Path
import pandas as pd
from utils.batch_loader import BatchLoader, BatchGenerator
from models import model_5, model_7, model_10


def fetch_batches(fragments, fragments_rc, labels, random_seed, batch_size, train_fr):
    # for reverse contigs we use half a batch size, as the second half is filled with reverse fragments
    batch_size_ = int(batch_size / 2)
    # shuffle and separate into train and validation
    # complicated approach to prepare balanced batches
    random.seed(a=random_seed)
    n_seqs = int(fragments.shape[0] / 3)
    ind_0 = shuffle(list(range(n_seqs)), random_state=random.randrange(1000000))
    ind_1 = shuffle(list(range(n_seqs, 2 * n_seqs)), random_state=random.randrange(1000000))
    ind_2 = shuffle(list(range(2 * n_seqs, 3 * n_seqs)), random_state=random.randrange(1000000))

    batches = []
    ind_n_, to_add = divmod(batch_size_, 3)
    # memory of what class was used in the last batch
    last_class = [0, 0, 0]
    for i in range(n_seqs * 3 // batch_size_):
        batch_idx = []
        # trick to reduce batches in turn, should work for any batch size
        # we balance batches in turns (batches are even, and we have 3 classes)
        class_to_add = i % 3
        if to_add == 1:
            ind_n = [ind_n_] * 3
            ind_n[class_to_add] += 1
        elif to_add == 2:
            ind_n = [ind_n_ + 1] * 3
            ind_n[class_to_add] -= 1
        else:
            ind_n = [ind_n_] * 3
        batch_idx.extend(ind_0[last_class[0]:last_class[0] + ind_n[0]])
        batch_idx.extend(ind_1[last_class[1]:last_class[1] + ind_n[1]])
        batch_idx.extend(ind_2[last_class[2]:last_class[2] + ind_n[2]])
        batches.append(batch_idx)
        # updating starting coordinate for the i array, so that we do not lose any entries
        last_class = [x + y for x, y in zip(last_class, ind_n)]
    print("Finished preparation of balanced batches")
    assert train_fr < 1.0
    train_batches = batches[:int(len(batches) * train_fr)]
    val_batches = batches[int(len(batches) * train_fr):]
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
        f = h5py.File(Path(ds_path, f"encoded_train_{length}.hdf5"), "r")
        fragments = f["fragments"]
        fragments_rc = f["fragments_rc"]
        labels = f["labels"]
    except FileNotFoundError:
        raise Exception("dataset was not found. Change ds_path or launch prepare_ds script")
    print(f'using {random_seed} random_seed in batch generation')
    train_gen, val_gen = fetch_batches(fragments,
                                       fragments_rc,
                                       labels,
                                       random_seed=random_seed,
                                       batch_size=batch_size,
                                       train_fr=0.9)

    models_list = zip(["model_5", "model_7", "model_10"], [model_5, model_7, model_10])

    for model_, model_obj in models_list:
        model = model_obj.model(length)
        model.fit(x=train_gen,
                  validation_data=val_gen,
                  epochs=epochs,
                  callbacks=callbacks,
                  batch_size=batch_size,
                  verbose=2)
        # taking into account validation data
        model.fit(x=val_gen,
                  epochs=1,
                  batch_size=batch_size,
                  verbose=2)
        model.save_weights(Path(out_path, f"{model_}.h5"))
        print(f'finished training {model_} network')


def launch_train_nn(config):
    with open(config, "r") as yamlfile:
        cf = yaml.load(yamlfile, Loader=yaml.FullLoader)
    train_nn(
        ds_path=cf[3]["train_nn"]["ds_path"],
        out_path=cf[3]["train_nn"]["out_path"],
        length=cf[3]["train_nn"]["fragment_length"],
        epochs=cf[3]["train_nn"]["epochs"],
        random_seed=cf[3]["train_nn"]["random_seed"],
    )


if __name__ == '__main__':
    fire.Fire(launch_train_nn)
