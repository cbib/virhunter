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
from pathlib import Path
import ray
from sklearn.utils import shuffle
import utils.preprocess as pp
import numpy as np


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


def sample_test_complex(
        path_virus,
        path_plant_genome,
        path_plant_cds,
        path_plant_chl,
        path_bact,
        out_path,
        fragment_length=1000,
        n_frags = 10000,
        n_cpus=1,
        random_seed=1,
):
    path_plants = [
        path_plant_genome,
        path_plant_cds,
        path_plant_chl,
    ]
    plant_weights = [0.6, 0.3, 0.1]
    # sampling virus
    _, _, _, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=path_virus, fragment_length=fragment_length,
        n_frags=n_frags, label='virus', label_int=1, random_seed=random_seed, n_cpus=n_cpus)
    seqs = shuffle(seqs, random_state=random_seed, n_samples=n_frags)
    assert len(seqs) == n_frags
    out_path_seqs = Path(out_path, f"seqs_virus_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(seqs, out_path_seqs, "fasta")
    # sampling plant
    pl_seqs = []
    for path_plant, w in zip(path_plants, plant_weights):
        n_frags_1_pl = int(n_frags * w)
        _, _, _, seqs, _ = pp.prepare_ds_sampling(
            in_seqs=path_plant, fragment_length=fragment_length,
            n_frags=n_frags_1_pl, label='plant', label_int=0, random_seed=random_seed, n_cpus=n_cpus)
        seqs = shuffle(seqs, random_state=random_seed, n_samples=n_frags_1_pl)
        pl_seqs.extend(seqs)
    # force same number of fragments
    assert len(pl_seqs) == n_frags
    out_path_seqs = Path(out_path, f"seqs_plant_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(pl_seqs, out_path_seqs, "fasta")
    # sampling bacteria
    _, _, _, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=path_bact, fragment_length=fragment_length,
        n_frags=n_frags, label='bacteria', label_int=2, random_seed=random_seed, n_cpus=n_cpus)
    seqs = shuffle(seqs, random_state=random_seed, n_samples=n_frags)
    assert len(seqs) == n_frags
    out_path_seqs = Path(out_path, f"seqs_bacteria_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(seqs, out_path_seqs, "fasta")


def launch_prepare_ds_rf(config):
    with open(config, "r") as yamlfile:
        cf = yaml.load(yamlfile, Loader=yaml.FullLoader)
    prepare_ds_rf(
        path_virus=cf[2]["prepare_ds_rf"]["path_virus"],
        path_plant=cf[2]["prepare_ds_rf"]["path_plant"],
        path_bact=cf[2]["prepare_ds_rf"]["path_bact"],
        out_path=cf[2]["prepare_ds_rf"]["out_path"],
        fragment_length=cf[2]["prepare_ds_rf"]["fragment_length"],
        n_cpus=cf[2]["prepare_ds_rf"]["n_cpus"],
        random_seed=cf[2]["prepare_ds_rf"]["random_seed"],
    )


if __name__ == '__main__':
    fire.Fire(launch_prepare_ds_rf)
