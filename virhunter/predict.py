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
import tensorflow as tf
import numpy as np
from Bio import SeqIO
import pandas as pd
import ray
from utils import preprocess as pp
from pathlib import Path
from models import model_5, model_7, model_10
from joblib import load
import psutil


def predict_nn(ds_path, nn_weights_path, length, n_cpus=3, batch_size=256):
    """
    Breaks down contigs into fragments
    and uses pretrained neural networks to give predictions for fragments
    """
    pid = psutil.Process(os.getpid())
    pid.cpu_affinity(range(n_cpus))

    print("loading sequences for prediction")
    try:
        seqs_ = list(SeqIO.parse(ds_path, "fasta"))
    except FileNotFoundError:
        raise Exception("test dataset was not found. Change ds variable")

    print("generating viral fragments and labels")
    out_table = {
        "id": [],
        "length": [],
        "fragment": [],
        "pred_plant_5": [],
        "pred_vir_5": [],
        "pred_bact_5": [],
        "pred_plant_7": [],
        "pred_vir_7": [],
        "pred_bact_7": [],
        "pred_plant_10": [],
        "pred_vir_10": [],
        "pred_bact_10": [],
    }
    if not seqs_:
        raise ValueError("All sequences were smaller than length of the model")
    test_fragments = []
    test_fragments_rc = []
    ray.init(num_cpus=n_cpus, num_gpus=0, include_dashboard=False)
    for seq in seqs_:
        fragments_, fragments_rc, _ = pp.fragmenting([seq], length, max_gap=0.8,
                                                     sl_wind_step=int(length / 2))
        test_fragments.extend(fragments_)
        test_fragments_rc.extend(fragments_rc)
        for j in range(len(fragments_)):
            out_table["id"].append(seq.id)
            out_table["length"].append(len(seq.seq))
            out_table["fragment"].append(j)
    it = pp.chunks(test_fragments, int(len(test_fragments) / n_cpus + 1))
    test_encoded = np.concatenate(ray.get([pp.one_hot_encode.remote(s) for s in it]))
    it = pp.chunks(test_fragments_rc, int(len(test_fragments_rc) / n_cpus + 1))
    test_encoded_rc = np.concatenate(ray.get([pp.one_hot_encode.remote(s) for s in it]))
    print('Encoding sequences finished')
    print(
        f"{np.shape(test_encoded)[0]} + {np.shape(test_encoded_rc)[0]} fragments generated")
    ray.shutdown()

    print('Starting sequence prediction')
    for model, s in zip([model_5.model(length), model_7.model(length), model_10.model(length)], [5, 7, 10]):
        model.load_weights(Path(nn_weights_path, f"model_{s}.h5"))
        prediction = model.predict([test_encoded, test_encoded_rc], batch_size)
        out_table[f"pred_plant_{s}"].extend(list(prediction[..., 0]))
        out_table[f"pred_vir_{s}"].extend(list(prediction[..., 1]))
        out_table[f"pred_bact_{s}"].extend(list(prediction[..., 2]))
    print('Exporting predictions to csv file')
    return pd.DataFrame(out_table)

def predict_rf(df, rf_weights_path):
    """
    Using predictions by predict_nn and weights of a trained RF classifier gives a single prediction for a fragment
    """
    clf = load(Path(rf_weights_path, "RF.joblib"))
    X = df[
        ["pred_plant_5", "pred_vir_5", "pred_plant_7", "pred_vir_7", "pred_plant_10", "pred_vir_10", ]]
    y_pred = clf.predict(X)
    mapping = {0: "plant", 1: "virus", 2: "bacteria"}
    df["RF_decision"] = np.vectorize(mapping.get)(y_pred)
    prob_classes = clf.predict_proba(X)
    df["RF_pred_plant"] = prob_classes[..., 0]
    df["RF_pred_vir"] = prob_classes[..., 1]
    df["RF_pred_bact"] = prob_classes[..., 2]
    return df


def predict_contigs(df):
    """
    Based on predictions of predict_rf for fragments gives a final prediction for the whole contig
    """
    df = (
        df.groupby(["id", "length", 'RF_decision'], sort=False)
        .size()
        .unstack(fill_value=0)
    )
    df = df.reset_index()
    df = df.reindex(['length', 'id', 'virus', 'plant', 'bacteria'], axis=1)
    conditions = [
        (df['virus'] > df['plant']) & (df['virus'] > df['bacteria']),
        (df['plant'] > df['virus']) & (df['plant'] > df['bacteria']),
        (df['bacteria'] >= df['plant']) & (df['bacteria'] >= df['virus']),
    ]
    choices = ['virus', 'plant', 'bacteria']
    df['decision'] = np.select(conditions, choices, default='bacteria')
    df = df.sort_values(by='length', ascending=False)
    df = df.loc[:, ['length', 'id', 'virus', 'plant', 'bacteria', 'decision']]
    df = df.rename(columns={'virus': '# viral fragments', 'bacteria': '# bacterial fragments', 'plant': '# plant fragments'})
    return df


def launch_predict(config):
    """
    Function for realizing full prediction pipeline
    """
    with open(config, "r") as yamlfile:
        cf = yaml.load(yamlfile, Loader=yaml.FullLoader)
    dfs_fr = []
    dfs_cont = []
    for l_ in 500, 1000:
        df = predict_nn(
            ds_path=cf[0]["predict"]["ds_path"],
            nn_weights_path=cf[0]["predict"][f"nn_weights_path_{l_}"],
            length=l_,
            n_cpus=cf[0]["predict"]["n_cpus"],
        )
        df = predict_rf(
            df=df,
            rf_weights_path=cf[0]["predict"][f"rf_weights_path_{l_}"],
        )
        dfs_fr.append(df)
        df = predict_contigs(df)
        dfs_cont.append(df)
    df_500 = dfs_fr[0][(dfs_fr[0]['length'] >= 750) & (dfs_fr[0]['length'] < 1500)]
    df_1000 = dfs_fr[1][(dfs_fr[1]['length'] >= 1500)]
    df = pd.concat([df_1000, df_500], ignore_index=True)
    pred_fr = Path(cf[0]["predict"]["out_path"], f"{Path(cf[0]['predict']['ds_path']).stem}_predicted_contig_fragments.csv")
    df.to_csv(pred_fr)

    df_500 = dfs_cont[0][(dfs_cont[0]['length'] >= 750) & (dfs_cont[0]['length'] < 1500)]
    df_1000 = dfs_cont[1][(dfs_cont[1]['length'] >= 1500)]
    df = pd.concat([df_1000, df_500], ignore_index=True)
    pred_contigs = Path(cf[0]["predict"]["out_path"], f"{Path(cf[0]['predict']['ds_path']).stem}_predicted_contigs.csv")
    df.to_csv(pred_contigs)


if __name__ == '__main__':
    fire.Fire(launch_predict)
