import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import wandb
import pandas as pd
from pathlib import Path
from train import train
from predict import predict, ensemble_predict


# plant = "apple"
# model = "/home/gsukhorukov/classifier/wandb/run-20210901_175743-6h1sikfh/files"
# ensemble_model = "/home/gsukhorukov/classifier/wandb/run-20210831_155025-1z6bltt7/files"

plant = "peach"
model = "/home/gsukhorukov/classifier/wandb/run-20210901_175623-1nqs0yii/files"
ensemble_model = "/home/gsukhorukov/classifier/wandb/run-20210831_171513-lesy79h9/files"

# plant = "tomato"
# model = "/home/gsukhorukov/classifier/wandb/run-20210901_175336-21osg8kv/files"
# ensemble_model = "/home/gsukhorukov/classifier/wandb/run-20210902_091906-uin3pl02/files"

ds_path = "/home/gsukhorukov/classifier/data/chimera_seqs/"

exp_name = plant + "-chimeras_aggregated"

run = wandb.init(
    project="virhunter",
    entity="gregoruar",
    job_type="predict_chimeras",
    dir="/home/gsukhorukov/classifier",
    save_code=True,
)

cf = run.config
cf.rundir = run.dir
cf.host = plant
cf.exp_name = exp_name
cf.ds_path = ds_path
cf.n_cpus = 4
cf.batch_size = 256

weights = [
    Path(model, "model_4.h5"),
    Path(model, "model_7.h5"),
    Path(model, "model_10.h5"),
]

# # disaggregated
# for n_frag in range(9):
#     df = predict(
#         weights=weights,
#         length=1000,
#         ds=Path(ds_path, f"{plant}_n{n_frag}.fasta"),
#         n_cpus=cf.n_cpus,
#         batch_size=cf.batch_size
#     )
#     for clf_type in ["LR", "RF"]:
#         df = ensemble_predict(df, Path(ensemble_model, f"{plant}_{clf_type}.joblib"), aggregate=False)
#         name = f"{plant}_n{n_frag}_{clf_type}.csv"
#         saving_path = Path(run.dir, name)
#         df.to_csv(saving_path)

# aggregated
dfs = []
for n_frag in range(9):
    df = predict(
        weights=weights,
        length=1000,
        ds=Path(ds_path, f"{plant}_n{n_frag}.fasta"),
        n_cpus=cf.n_cpus,
        batch_size=cf.batch_size
    )
    for clf_type in ["LR", "RF"]:
        df_50 = ensemble_predict(df, Path(ensemble_model, f"{plant}_{clf_type}.joblib"), aggregate=True, quantile=0.5)
        df_75 = ensemble_predict(df, Path(ensemble_model, f"{plant}_{clf_type}.joblib"), aggregate=True, quantile=0.75)
        df_25 = ensemble_predict(df, Path(ensemble_model, f"{plant}_{clf_type}.joblib"), aggregate=True, quantile=0.25)

        df_ = df_50[["id"]]
        df_["clf_type"] = clf_type
        df_["n_frag"] = n_frag
        df_["25%"] = df_25.loc[:, ["predicted"]]
        df_["50%"] = df_50.loc[:, ["predicted"]]
        df_["75%"] = df_75.loc[:, ["predicted"]]
        dfs.append(df_)
df_merged = pd.concat(dfs)
df_merged.sort_values(by=['clf_type', 'id', 'n_frag'])
name = f"{plant}_aggregated.csv"
saving_path = Path(run.dir, name)
df_merged.to_csv(saving_path)
run.finish()
