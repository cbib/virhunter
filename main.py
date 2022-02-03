import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import fire
import yaml
from virhunter.train import train
from virhunter.predict import predict
from virhunter.prepare_ds import prepare_ds
from virhunter.utils.sample_test import sample_test


def main(task, config):
    with open(config, "r") as yamlfile:
        cf = yaml.load(yamlfile, Loader=yaml.FullLoader)
    if task == "prepare_ds":
        prepare_ds(
            path_virus=cf[0]["prepare_ds"]["path_virus"],
            path_plant=cf[0]["prepare_ds"]["path_plant"],
            path_bact=cf[0]["prepare_ds"]["path_bact"],
            out_path=cf[0]["prepare_ds"]["out_path"],
            fragment_length=cf[0]["prepare_ds"]["fragment_length"],
            n_cpus=cf[0]["prepare_ds"]["n_cpus"],
            random_seed=cf[0]["prepare_ds"]["random_seed"],
        )
    elif task == "sample_test":
        sample_test(
            path_virus=cf[1]["sample_test"]["path_virus"],
            path_plant=cf[1]["sample_test"]["path_plant"],
            path_bact=cf[1]["sample_test"]["path_bact"],
            out_path=cf[1]["sample_test"]["out_path"],
            fragment_length=cf[1]["sample_test"]["fragment_length"],
            n_cpus=cf[1]["sample_test"]["n_cpus"],
            random_seed=cf[1]["sample_test"]["random_seed"],
        )
    elif task == "train":
        train(
            ds_path=cf[2]["train"]["ds_path"],
            out_path=cf[2]["train"]["out_path"],
            test_nn_path=cf[2]["train"]["test_nn_path"],
            length=cf[2]["train"]["fragment_length"],
            n_cpus=cf[2]["train"]["n_cpus"],
            epochs=cf[2]["train"]["epochs"],
            random_seed=cf[2]["train"]["random_seed"],
        )
    elif task == "predict":
        predict(
            ds_path=cf[3]["predict"]["ds_path"],
            out_path=cf[3]["predict"]["out_path"],
            weights_path=cf[3]["predict"]["weights_path"],
            length=cf[3]["predict"]["fragment_length"],
            n_cpus=cf[3]["predict"]["n_cpus"],
        )

    elif task == "help":
        print("help")


if __name__ == '__main__':
    fire.Fire(main)

