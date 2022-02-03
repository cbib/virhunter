#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Credits: Grigorii Sukhorukov, Macha Nikolski

import os
import tensorflow as tf
import numpy as np
from Bio import SeqIO
import pandas as pd
import ray
from virhunter.utils import preprocess as pp
from pathlib import Path
from virhunter.models import model_5, model_7, model_10
from joblib import load


def predict(ds_path, weights_path, out_path, length, n_cpus=3, batch_size=256):
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
    ray.init(num_cpus=n_cpus, num_gpus=0)
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
        model.load_weights(Path(weights_path, f"model_{s}.h5"))
        prediction = model.predict([test_encoded, test_encoded_rc], batch_size)
        out_table[f"pred_plant_{s}"].extend(list(prediction[..., 0]))
        out_table[f"pred_vir_{s}"].extend(list(prediction[..., 1]))
        out_table[f"pred_bact_{s}"].extend(list(prediction[..., 2]))
    print('Exporting predictions to csv file')
    df = pd.DataFrame(out_table)
    pred_file = Path(out_path, f"{Path(ds_path).stem}_predicted.csv")
    df.to_csv(pred_file)
