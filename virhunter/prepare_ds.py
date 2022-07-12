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
from Bio import SeqIO
import random
import ray
import numpy as np
from utils import preprocess as pp
from sklearn.utils import shuffle
import h5py
from pathlib import Path


def prepare_ds_nn(
        path_virus,
        path_plant,
        path_bact,
        out_path,
        fragment_length=1000,
        n_cpus=1,
        random_seed=None,
):
    """
    This is a function for the example dataset preparation.
    If random seed is not specified it will be generated randomly.
    """

    if random_seed is None:
        random.seed(a=random_seed)
        random_seed = random.randrange(1000000)
    random.seed(a=random_seed)

    v_encoded, v_encoded_rc, v_labs, v_seqs, v_n_frags = pp.prepare_ds_fragmenting(
        in_seq=path_virus, label='virus', label_int=1, fragment_length=fragment_length,
        sl_wind_step=int(fragment_length / 2), max_gap=0.05, n_cpus=n_cpus)


    pl_encoded, pl_encoded_rc, pl_labs, _, pl_n_frags = pp.prepare_ds_sampling(
        in_seqs=path_plant, fragment_length=fragment_length, n_frags=v_n_frags, label='plant', label_int=0,
        random_seed=random.randrange(1000000))

    b_encoded, b_encoded_rc, b_labs, b_seqs, b_n_frags = pp.prepare_ds_sampling(
        in_seqs=path_bact, fragment_length=fragment_length,
        n_frags=v_n_frags, label='bacteria', label_int=2, random_seed=random.randrange(1000000))

    assert v_n_frags == b_n_frags
    assert v_n_frags == pl_n_frags

    all_encoded = np.concatenate([v_encoded, pl_encoded, b_encoded])
    all_encoded_rc = np.concatenate([v_encoded_rc, pl_encoded_rc, b_encoded_rc])
    all_labs = np.concatenate([v_labs, pl_labs, b_labs])

    # adding reverse complement
    all_encoded = np.concatenate((all_encoded, all_encoded_rc))
    all_encoded_rc = np.concatenate((all_encoded_rc, all_encoded))
    all_labs = np.concatenate((all_labs, all_labs))

    # saving one-hot encoded fragments
    pp.storing_encoded(all_encoded, all_encoded_rc, all_labs,
                       Path(out_path, f"encoded_train_{fragment_length}.hdf5"))


def prepare_ds_rf(
        path_virus,
        path_plant,
        path_bact,
        out_path,
        fragment_length=1000,
        n_cpus=1,
        random_seed=1,
):
    n_frags = 20000
    # sampling virus
    _, _, _, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=path_virus, fragment_length=fragment_length,
        n_frags=n_frags, label='virus', label_int=1, random_seed=random_seed, n_cpus=n_cpus)
    seqs = shuffle(seqs, random_state=random_seed, n_samples=n_frags)
    assert len(seqs) == n_frags
    out_path_seqs = Path(out_path, f"seqs_virus_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(seqs, out_path_seqs, "fasta")
    # sampling plant
    _, _, _, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=path_plant, fragment_length=fragment_length,
        n_frags=n_frags, label='plant', label_int=0, random_seed=random_seed, n_cpus=n_cpus)
    seqs = shuffle(seqs, random_state=random_seed, n_samples=n_frags)
    # force same number of fragments
    assert len(seqs) == n_frags
    out_path_seqs = Path(out_path, f"seqs_plant_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(seqs, out_path_seqs, "fasta")
    # sampling bacteria
    _, _, _, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=path_bact, fragment_length=fragment_length,
        n_frags=n_frags, label='bacteria', label_int=2, random_seed=random_seed, n_cpus=n_cpus)
    seqs = shuffle(seqs, random_state=random_seed, n_samples=n_frags)
    assert len(seqs) == n_frags
    out_path_seqs = Path(out_path, f"seqs_bacteria_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(seqs, out_path_seqs, "fasta")


def prepare_ds(config):
    with open(config, "r") as yamlfile:
        cf = yaml.load(yamlfile, Loader=yaml.FullLoader)

    assert Path(cf["prepare_ds"]["path_virus"]).exists(), f'{cf["prepare_ds"]["path_virus"]} does not exist'
    assert Path(cf["prepare_ds"]["path_plant"]).exists(), f'{cf["prepare_ds"]["path_plant"]} does not exist'
    assert Path(cf["prepare_ds"]["path_bact"]).exists(), f'{cf["prepare_ds"]["path_bact"]} does not exist'

    Path(cf["prepare_ds"]["out_path"], "500").mkdir(parents=True, exist_ok=True)
    Path(cf["prepare_ds"]["out_path"], "1000").mkdir(parents=True, exist_ok=True)

    for l_ in 500, 1000:
        prepare_ds_nn(
            path_virus=cf["prepare_ds"]["path_virus"],
            path_plant=cf["prepare_ds"]["path_plant"],
            path_bact=cf["prepare_ds"]["path_bact"],
            out_path=Path(cf["prepare_ds"]["out_path"], f"{l_}"),
            fragment_length=l_,
            n_cpus=cf["prepare_ds"]["n_cpus"],
            random_seed=cf["prepare_ds"]["random_seed"],
        )
        prepare_ds_rf(
            path_virus=cf["prepare_ds"]["path_virus"],
            path_plant=cf["prepare_ds"]["path_plant"],
            path_bact=cf["prepare_ds"]["path_bact"],
            out_path=Path(cf["prepare_ds"]["out_path"], f"{l_}"),
            fragment_length=l_,
            n_cpus=cf["prepare_ds"]["n_cpus"],
            random_seed=cf["prepare_ds"]["random_seed"],
        )
        print(f"finished dataset preparation for {l_} fragment size")
    print(f"NN and RF datasets are stored in {cf['prepare_ds']['out_path']}")


if __name__ == '__main__':
    fire.Fire(prepare_ds)
