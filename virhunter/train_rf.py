#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Credits: Grigorii Sukhorukov, Macha Nikolski
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import fire
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
import predict as pr


def subset_df(df, org, thr=0.8, final_df_size=1000):
    if thr == 1.0:
        df_1 = df.sample(n=final_df_size)
    else:
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


def load_ds(df, label, family=None,):
    df = df.drop(['length', 'fragment'], 1)
    df["label"] = label
    if family is not None:
        df["family"] = family
    return df


def merge_ds(path_ds_v, path_ds_pl, path_ds_b, fract, rs, family=None,):
    df_vir = load_ds(path_ds_v, label=1, family=family,)
    df_plant = load_ds(path_ds_pl, label=0, family=family,)
    df_bact = load_ds(path_ds_b, label=2, family=family,)
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


def train_rf(nn_weights_path, ds_rf_path, out_path, length, n_cpus, random_seed):
    weights = [
        Path(nn_weights_path, "model_5.h5"),
        Path(nn_weights_path, "model_7.h5"),
        Path(nn_weights_path, "model_10.h5"),
    ]
    print('predictions for test dataset')
    dfs = []
    for org in ['virus', 'plant', 'bacteria']:
        df = pr.predict(
            ds_path=Path(ds_rf_path, f"seqs_{org}_sampled_{length}_20000.fasta"),
            weights_path=nn_weights_path,
            length=length,
            n_cpus=n_cpus,
            batch_size=256
        )
        dfs.append(df)
    # subsetting predictions
    df_v = subset_df(dfs[0], 'vir', thr=0.8)
    df_pl = subset_df(dfs[1], 'plant', thr=1.0)
    df_b = subset_df(dfs[2], 'bact', thr=1.0)
    print('training ml classifier')
    df_train, _ = merge_ds(path_ds_v=df_v,
                                    path_ds_pl=df_pl,
                                    path_ds_b=df_b,
                                    fract=0.2,
                                    rs=random_seed,)
    df_train = df_train.append(_, sort=False)
    _ = fit_clf(df_train, Path(out_path, "RF.joblib"), random_seed)


def launch_train_rf(config):
    with open(config, "r") as yamlfile:
        cf = yaml.load(yamlfile, Loader=yaml.FullLoader)
    train_rf(
        nn_weights_path=cf[4]["train_rf"]["nn_weights_path"],
        ds_rf_path=cf[4]["train_rf"]["ds_rf_path"],
        out_path=cf[4]["train_rf"]["out_path"],
        length=cf[4]["train_rf"]["fragment_length"],
        n_cpus=cf[4]["train_rf"]["n_cpus"],
        random_seed=cf[4]["train_rf"]["random_seed"],
    )

if __name__ == '__main__':
    fire.Fire(launch_train_rf)
