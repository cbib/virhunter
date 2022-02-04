#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Credits: Grigorii Sukhorukov, Macha Nikolski

import os
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import h5py
import random
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
from virhunter.utils.batch_loader import BatchLoader, BatchGenerator
from virhunter.models import model_5, model_7, model_10
import virhunter.predict as pr


def load_ds(path_ds, label, family=None, loaded=False):
    df = path_ds if loaded else pd.read_csv(path_ds, sep=',')
    df = df.drop(['length', 'fragment'], 1)
    df["label"] = label
    if family is not None:
        df["family"] = family
    return df


def merge_ds(path_ds_v, path_ds_pl, path_ds_b, fract, rs, family=None, loaded=False):
    df_vir = load_ds(path_ds_v, label=1, family=family, loaded=loaded)
    df_plant = load_ds(path_ds_pl, label=0, family=family, loaded=loaded)
    df_bact = load_ds(path_ds_b, label=2, family=family, loaded=loaded)
    # balancing datasets by downsampling to the size of the smallest dataset
    l_ = min(df_vir.shape[0], df_bact.shape[0], df_plant.shape[0])
    df_vir = df_vir.sample(frac=l_ / df_vir.shape[0], random_state=rs)
    df_plant = df_plant.sample(frac=l_ / df_plant.shape[0], random_state=rs)
    df_bact = df_bact.sample(frac=l_ / df_bact.shape[0], random_state=rs)
    # splitting on train and test
    df_vir_train, df_vir_test = train_test_split(df_vir, test_size=fract, shuffle=False)
    df_plant_train, df_plant_test = train_test_split(df_plant, test_size=fract, shuffle=False)
    df_bact_train, df_bact_test = train_test_split(df_bact, test_size=fract, shuffle=False)
    # merging dfs
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    df_train = df_train.append([df_vir_train, df_plant_train, df_bact_train], ignore_index=True)
    df_test = df_test.append([df_vir_test, df_plant_test, df_bact_test], ignore_index=True)
    return df_train, df_test


def fit_clf(df, save_path, rs):
    df_reshuffled = df.sample(frac=1, random_state=rs)
    X = df_reshuffled[["pred_plant_5", "pred_vir_5", "pred_plant_7", "pred_vir_7", "pred_plant_10", "pred_vir_10", ]]
    y = df_reshuffled["label"]
    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, max_samples=0.3)
    clf.fit(X, y)
    dump(clf, save_path)
    return clf


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


def subset_df(df_path, org, thr=0.8, final_df_size=1000):
    if thr == 1.0:
        df = pd.read_csv(df_path, sep=',')
        df_1 = df.sample(n=final_df_size)
    else:
        df = pd.read_csv(df_path, sep=',')
        df_1 = df.query(f'pred_{org}_5 <= {thr} | pred_{org}_7 <= {thr} | pred_{org}_10 <= {thr}')
        print(df_1.shape[0])
        if df_1.shape[0] < int(final_df_size/2):
            print('too little bad predictions')
            df_1 = df
        df_1 = df_1.sample(n=int(final_df_size/2))
        df_2 = df.query(f'pred_{org}_5 > {thr} & pred_{org}_7 > {thr} & pred_{org}_10 > {thr}')
        df_2 = df_2.sample(n=int(final_df_size/2))
        df_1 = df_1.append(df_2, sort=False)
    return df_1


def train(
        ds_path,
        out_path,
        length,
        test_nn_path,
        n_cpus=1,
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

    weights = [
        Path(out_path, "model_5.h5"),
        Path(out_path, "model_7.h5"),
        Path(out_path, "model_10.h5"),
    ]
    print('predictions for test dataset')
    for org in ['virus', 'plant', 'bacteria']:
        df = pr.predict(
            weights=weights,
            length=length,
            ds=Path(test_nn_path, f"seqs_{org}_sampled_{length}_20000.fasta"),
            n_cpus=n_cpus,
            batch_size=256
        )
        name = f"pred-{org}.csv"
        saving_path = Path(out_path, name)
        df.to_csv(saving_path)
    # subsetting predictions
    df_v = subset_df(Path(out_path, "pred-virus.csv"), 'vir', thr=0.8)
    df_pl = subset_df(Path(out_path, "pred-plant.csv"), 'plant', thr=1.0)
    df_b = subset_df(Path(out_path, "pred-bacteria.csv"), 'bact', thr=1.0)
    print('training ml classifier')
    df_train, _ = merge_ds(path_ds_v=df_v,
                                    path_ds_pl=df_pl,
                                    path_ds_b=df_b,
                                    fract=0.2,
                                    rs=random_seed, loaded=True)
    df_train = df_train.append(_, sort=False)
    _ = fit_clf(df_train, Path(out_path, "RF.joblib"), random_seed)
    return out_path
