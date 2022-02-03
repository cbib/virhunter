import os
from Bio import SeqIO
import random
import ray
import numpy as np
from virhunter.utils import preprocess as pp
from sklearn.utils import shuffle
import h5py
from pathlib import Path


def prepare_ds(
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
    print(f'starting generation using random seed {random_seed}')
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


def prepare_ds_complex(
        path_virus,
        path_plant_genome,
        path_plant_cds,
        path_plant_chl,
        path_bact,
        out_path,
        fragment_length=1000,
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

    v_encoded, v_encoded_rc, v_labs, v_seqs, v_n_frags = pp.prepare_ds_fragmenting(
        in_seq=path_virus, label='virus', label_int=1, fragment_length=fragment_length,
        sl_wind_step=int(fragment_length/2), max_gap=0.05, n_cpus=n_cpus)

    pl_g_encoded, pl_g_encoded_rc, pl_g_labs, pl_g_seqs, pl_g_n_frags = pp.prepare_ds_fragmenting(
        in_seq=path_plant_genome, label='plant', label_int=0, fragment_length=fragment_length,
        sl_wind_step=fragment_length, max_gap=0.05, n_cpus=n_cpus)
    pl_g_n_frags = int(v_n_frags * 0.6)
    pl_g_encoded, pl_g_encoded_rc, pl_g_labs, pl_g_seqs = shuffle(
        pl_g_encoded, pl_g_encoded_rc, pl_g_labs, pl_g_seqs, random_state=random_seed, n_samples=pl_g_n_frags)

    pl_cds_encoded, pl_cds_encoded_rc, pl_cds_labs, pl_cds_seqs, pl_cds_n_frags = pp.prepare_ds_fragmenting(
        in_seq=path_plant_cds, label='plant', label_int=0, fragment_length=fragment_length,
        sl_wind_step=int(fragment_length/2), max_gap=0.05, n_cpus=n_cpus)
    print(len(pl_cds_seqs), pl_cds_n_frags)
    pl_chl_n_frags = int(v_n_frags * 0.1)

    pl_chl_encoded, pl_chl_encoded_rc, pl_chl_labs, pl_chl_seqs, pl_chl_n_frags = pp.prepare_ds_sampling(
        in_seqs=path_plant_chl, fragment_length=fragment_length, n_frags=pl_chl_n_frags, label='plant', label_int=0,
        random_seed=random.randrange(1000000))
    print(len(pl_chl_seqs), pl_chl_n_frags)

    pl_cds_extra_n_frags = v_n_frags - pl_cds_n_frags - pl_g_n_frags - pl_chl_n_frags

    pl_cds_extra_encoded, pl_cds_extra_encoded_rc, pl_cds_extra_labs, pl_cds_extra_seqs, pl_cds_extra_n_frags = \
        pp.prepare_ds_sampling(in_seqs=path_plant_cds, fragment_length=fragment_length,
                               n_frags=pl_cds_extra_n_frags,
                               label='plant', label_int=0, random_seed=random.randrange(1000000))
    print(len(pl_cds_extra_seqs), pl_cds_extra_n_frags)

    b_encoded, b_encoded_rc, b_labs, b_seqs, b_n_frags = pp.prepare_ds_sampling(
        in_seqs=path_bact, fragment_length=fragment_length,
        n_frags=v_n_frags, label='bacteria', label_int=2, random_seed=random.randrange(1000000))

    assert v_n_frags == b_n_frags
    assert v_n_frags == pl_cds_n_frags + pl_g_n_frags + pl_chl_n_frags + pl_cds_extra_n_frags

    all_encoded = np.concatenate([v_encoded, pl_g_encoded, pl_cds_encoded, pl_chl_encoded,
                                  pl_cds_extra_encoded, b_encoded])
    all_encoded_rc = np.concatenate([v_encoded_rc, pl_g_encoded_rc, pl_cds_encoded_rc, pl_chl_encoded_rc,
                                     pl_cds_extra_encoded_rc, b_encoded_rc])
    all_labs = np.concatenate([v_labs, pl_g_labs, pl_cds_labs, pl_chl_labs, pl_cds_extra_labs, b_labs])

    # adding reverse complement
    all_encoded = np.concatenate((all_encoded, all_encoded_rc))
    all_encoded_rc = np.concatenate((all_encoded_rc, all_encoded))
    all_labs = np.concatenate((all_labs, all_labs))

    # saving one-hot encoded fragments
    pp.storing_encoded(all_encoded, all_encoded_rc, all_labs,
                    Path(out_path, f"encoded_train_{name}_{fragment_length}.hdf5"))
