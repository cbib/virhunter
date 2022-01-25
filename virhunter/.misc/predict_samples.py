import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from Bio import SeqIO
import pandas as pd
import ray
import wandb
from utils import preprocess as pp
from predict import predict, ensemble_predict
from pathlib import Path

plant = 'peach-cds-g-chl_masked_bact'
exp_name = f"{plant}_samples"

run = wandb.init(
    project="virhunter",
    entity="gregoruar",
    job_type="predict",
    dir="/home/gsukhorukov/classifier",
    save_code=True,
)

conf = run.config
conf.exp_name = exp_name
conf.n_cpus = 4
conf.n_gpus = 0
conf.batch_size = 256
conf.n_classes = 3


model = "/home/gsukhorukov/classifier/wandb/run-20211001_002131-2k6soew3/files"
clf_weights = "/home/gsukhorukov/classifier/wandb/run-20211001_102246-11idd0ly/files"
weights = [
    # peach
    Path(model, "model_5.h5"),
    Path(model, "model_7.h5"),
    Path(model, "model_10.h5"),
]
test_ds_name = ["viral", "bact", "peach"]
conf.ds_path = "/home/gsukhorukov/classifier/data/sampled_test_fragments_3"
conf.length = 1000
conf.rundir = run.dir
mut_rate = 0

for organism in test_ds_name:
    for fr_length in [250, 500, 750, 1000, 1500, 2000, 3000, 6000]:
        print(f"starting predictions for {organism} {fr_length}")
        ds_path = Path(conf.ds_path, f"{organism}_{fr_length}_{str(int(mut_rate*100))}prc.fasta")
        df_out = predict(weights, conf.length, ds_path, n_cpus=conf.n_cpus, batch_size=conf.batch_size)
        df_out = ensemble_predict(df_out, Path(clf_weights, f"{plant}_RF.joblib"), aggregate=False)
        pred_file = Path(run.dir, f"{conf.exp_name}_{organism}_{fr_length}_{str(int(mut_rate*100))}prc.csv")
        df_out.to_csv(pred_file)
