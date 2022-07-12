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


def prepare_ds_nn_complex(
        path_virus,
        path_plant_genome,
        path_plant_cds,
        path_plant_chl,
        path_bact,
        out_path,
        fragment_length,
        n_cpus=1,
        random_seed=None,
):
    """
    This is a function for the dataset preparation used in the paper.
    Compared to prepare_ds function it requires paths to plant genome, cds and chloroplast sequences.
    It also uses fragmenting procedure for plant genome and cds as described in the paper.
    """

    if random_seed is None:
        random.seed(a=random_seed)
        random_seed = random.randrange(1000000)
    print(f'starting generation using random seed {random_seed}')
    random.seed(a=random_seed)

    v_encoded, v_encoded_rc, v_labs, _, v_n_frags = pp.prepare_ds_fragmenting(
        in_seq=path_virus, label='virus', label_int=1, fragment_length=fragment_length,
        sl_wind_step=int(fragment_length/2), max_gap=0.05, n_cpus=n_cpus)

    pl_g_encoded, pl_g_encoded_rc, pl_g_labs, _, _ = pp.prepare_ds_fragmenting(
        in_seq=path_plant_genome, label='plant', label_int=0, fragment_length=fragment_length,
        sl_wind_step=fragment_length, max_gap=0.05, n_cpus=n_cpus)
    pl_g_n_frags = int(v_n_frags * 0.6)
    pl_g_encoded, pl_g_encoded_rc, pl_g_labs = shuffle(
        pl_g_encoded, pl_g_encoded_rc, pl_g_labs, random_state=random_seed, n_samples=pl_g_n_frags)

    pl_cds_encoded, pl_cds_encoded_rc, pl_cds_labs, _, _ = pp.prepare_ds_fragmenting(
        in_seq=path_plant_cds, label='plant', label_int=0, fragment_length=fragment_length,
        sl_wind_step=int(fragment_length/2), max_gap=0.05, n_cpus=n_cpus)
    pl_cds_n_frags = int(v_n_frags * 0.3)
    pl_cds_encoded, pl_cds_encoded_rc, pl_cds_labs = shuffle(
        pl_cds_encoded, pl_cds_encoded_rc, pl_cds_labs, random_state=random_seed, n_samples=pl_cds_n_frags)

    pl_chl_n_frags = int(v_n_frags * 0.1)
    pl_chl_encoded, pl_chl_encoded_rc, pl_chl_labs, _, pl_chl_n_frags = pp.prepare_ds_sampling(
        in_seqs=path_plant_chl, fragment_length=fragment_length, n_frags=pl_chl_n_frags, label='plant', label_int=0,
        random_seed=random.randrange(1000000))

    pl_cds_extra_n_frags = v_n_frags - pl_cds_n_frags - pl_g_n_frags - pl_chl_n_frags
    if pl_cds_extra_n_frags > 0:
        pl_cds_extra_encoded, pl_cds_extra_encoded_rc, pl_cds_extra_labs, _, _ = \
            pp.prepare_ds_sampling(in_seqs=path_plant_cds, fragment_length=fragment_length,
                                   n_frags=pl_cds_extra_n_frags,
                                   label='plant', label_int=0, random_seed=random.randrange(1000000))
        pl_cds_encoded = np.concatenate([pl_cds_encoded, pl_cds_extra_encoded])
        pl_cds_encoded_rc = np.concatenate([pl_cds_encoded_rc, pl_cds_extra_encoded_rc])
        pl_cds_labs = np.concatenate([pl_cds_labs, pl_cds_labs])

    b_encoded, b_encoded_rc, b_labs, b_seqs, b_n_frags = pp.prepare_ds_sampling(
        in_seqs=path_bact, fragment_length=fragment_length,
        n_frags=v_n_frags, label='bacteria', label_int=2, random_seed=random.randrange(1000000))

    # assert v_n_frags == b_n_frags
    # assert v_n_frags == pl_cds_n_frags + pl_g_n_frags + pl_chl_n_frags + pl_cds_extra_n_frags

    all_encoded = np.concatenate([v_encoded, pl_g_encoded, pl_cds_encoded, pl_chl_encoded, b_encoded])
    all_encoded_rc = np.concatenate([v_encoded_rc, pl_g_encoded_rc, pl_cds_encoded_rc, pl_chl_encoded_rc, b_encoded_rc])
    all_labs = np.concatenate([v_labs, pl_g_labs, pl_cds_labs, pl_chl_labs, b_labs])

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


def prepare_ds_rf_complex(
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


def prepare_ds_complex(config):
    with open(config, "r") as yamlfile:
        cf = yaml.load(yamlfile, Loader=yaml.FullLoader)

    assert Path(cf["prepare_ds_complex"]["path_virus"]).exists(), f'{cf["prepare_ds_complex"]["path_virus"]} does not exist'
    assert Path(cf["prepare_ds_complex"]["path_plant_genome"]).exists(), f'{cf["prepare_ds_complex"]["path_plant_genome"]} does not exist'
    assert Path(cf["prepare_ds_complex"]["path_plant_chl"]).exists(), f'{cf["prepare_ds_complex"]["path_plant_chl"]} does not exist'
    assert Path(cf["prepare_ds_complex"]["path_plant_cds"]).exists(), f'{cf["prepare_ds_complex"]["path_plant_cds"]} does not exist'
    assert Path(cf["prepare_ds_complex"]["path_bact"]).exists(), f'{cf["prepare_ds_complex"]["path_bact"]} does not exist'

    Path(cf["prepare_ds_complex"]["out_path"]).mkdir(parents=True, exist_ok=True)

    for l_ in 500, 1000:
        prepare_ds_nn_complex(
            path_virus=cf["prepare_ds_complex"]["path_virus"],
            path_plant_genome=cf["prepare_ds_complex"]["path_plant_genome"],
            path_plant_cds=cf["prepare_ds_complex"]["path_plant_cds"],
            path_plant_chl=cf["prepare_ds_complex"]["path_plant_chl"],
            path_bact=cf["prepare_ds_complex"]["path_bact"],
            out_path=cf["prepare_ds_complex"]["out_path"],
            fragment_length=l_,
            n_cpus=cf["prepare_ds_complex"]["n_cpus"],
            random_seed=cf["prepare_ds_complex"]["random_seed"],
        )
        prepare_ds_rf_complex(
            path_virus=cf["prepare_ds_complex"]["path_virus"],
            path_plant_genome=cf["prepare_ds_complex"]["path_plant_genome"],
            path_plant_cds=cf["prepare_ds_complex"]["path_plant_cds"],
            path_plant_chl=cf["prepare_ds_complex"]["path_plant_chl"],
            path_bact=cf["prepare_ds_complex"]["path_bact"],
            out_path=cf["prepare_ds_complex"]["out_path"],
            fragment_length=l_,
            n_cpus=cf["prepare_ds_complex"]["n_cpus"],
            random_seed=cf["prepare_ds_complex"]["random_seed"],
        )
        print(f"finished dataset preparation for {l_} fragment size\n")
    print(f"NN and RF datasets are stored in {cf['prepare_ds']['out_path']}")


if __name__ == '__main__':
    fire.Fire(prepare_ds_complex)
