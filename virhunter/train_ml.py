import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load


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
