import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Bio import SeqIO
import random
import ray
import numpy as np
from utils import preprocess as pp
from sklearn.utils import shuffle
import h5py
import wandb
from pathlib import Path


def prepare_ds(
        path_virus, ds, plant, path_plant_genome, path_plant_cds, path_plant_chl, path_bact, out_path,
        fragment_length=1000,
        n_cpus=1,
        random_seed=None,
):
    # TODO: write docs
    # TODO: reformat arguments line

    run = wandb.init(
        project="virhunter",
        entity="gregoruar",
        job_type="prepare_ds",
        dir="/home/gsukhorukov/classifier",
        save_code=True,
    )
    cf = run.config
    # cf.ds_path = [path_virus, path_plant, path_bact, out_path]

    if random_seed is None:
        random.seed(a=random_seed)
        random_seed = random.randrange(1000000)
    print(f'starting generation using random seed {random_seed}')
    cf.rs = random_seed
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
    # print(len(pl_g_seqs), pl_g_n_frags)

    pl_cds_encoded, pl_cds_encoded_rc, pl_cds_labs, pl_cds_seqs, pl_cds_n_frags = pp.prepare_ds_fragmenting(
        in_seq=path_plant_cds, label='plant', label_int=0, fragment_length=fragment_length,
        sl_wind_step=int(fragment_length/2), max_gap=0.05, n_cpus=n_cpus)
    print(len(pl_cds_seqs), pl_cds_n_frags)

    # pl_chl_n_frags = (pl_cds_n_frags + pl_g_n_frags) * 0.1
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

    # all_seqs = [v_seqs, pl_g_seqs, pl_cds_seqs, pl_chl_seqs, pl_cds_seqs, b_seqs]
    pp.storing_encoded(all_encoded, all_encoded_rc, all_labs,
                    Path(out_path, f"encoded_train_{ds}_{fragment_length}.hdf5"))

    print('saving parts of the dataset as sequences and encoded')
    if 'reduced_viruses' in ds:
        v_path = Path(out_path, 'fixed_parts', f"encoded_{ds}_{fragment_length}.hdf5")
        v_path_seqs = Path(out_path, 'fixed_parts', f"seqs_{ds}_{fragment_length}.fasta")
    else:
        v_path = Path(out_path, 'fixed_parts', f"encoded_virus_{fragment_length}.hdf5")
        v_path_seqs = Path(out_path, 'fixed_parts', f"seqs_virus_{fragment_length}.fasta")
    if not (v_path.exists() and v_path_seqs.exists()):
        pp.storing_encoded(v_encoded, v_encoded_rc, v_labs, v_path)
        SeqIO.write(v_seqs, v_path_seqs, "fasta")
    pl_path = Path(out_path, 'fixed_parts', f"encoded_{plant}_{fragment_length}.hdf5")
    pl_path_seqs = Path(out_path, 'fixed_parts', f"seqs_{plant}_{fragment_length}.fasta")
    if not(pl_path.exists() and pl_path_seqs.exists()):
        pl_encoded = np.concatenate([pl_g_encoded, pl_cds_encoded, pl_chl_encoded, pl_cds_extra_encoded])
        pl_encoded_rc = np.concatenate([pl_g_encoded_rc, pl_cds_encoded_rc, pl_chl_encoded_rc, pl_cds_extra_encoded_rc])
        pl_labs = np.concatenate([pl_g_labs, pl_cds_labs, pl_chl_labs, pl_cds_extra_labs])
        pl_seqs = sum([pl_g_seqs, pl_cds_seqs, pl_chl_seqs, pl_cds_extra_seqs], [])
        pp.storing_encoded(pl_encoded, pl_encoded_rc, pl_labs, pl_path)
        SeqIO.write(pl_seqs, pl_path_seqs, "fasta")
    b_path = Path(out_path, 'fixed_parts', f"encoded_bacteria_{fragment_length}.hdf5")
    b_path_seqs = Path(out_path, 'fixed_parts', f"seqs_bacteria_{fragment_length}.fasta")
    if not(b_path.exists() and b_path_seqs.exists()):
        pp.storing_encoded(b_encoded, b_encoded_rc, b_labs, b_path)
        SeqIO.write(b_seqs, b_path_seqs, "fasta")
    run.finish()


if __name__ == "__main__":
    # path_viruses = '/home/gsukhorukov/vir_db/plant-virus_all_2021-10-26.fasta'
    path_viruses = '/home/gsukhorukov/vir_db/plant-virus_no-grapevine-viruses_2021-10-26.fasta'

    # plant = "peach"
    # path_plant_chl = "/home/gsukhorukov/plant_db/peach/peach_chl.fasta"
    # path_plant_genome = "/home/gsukhorukov/plant_db/peach/peach_genome.fasta"
    # path_plant_cds = "/home/gsukhorukov/plant_db/peach/peach_cds.fasta"
    # ds = "sugar_beet_reduced_viruses"
    # plant = "sugar_beet"
    # path_plant_chl = "/home/gsukhorukov/plant_db/sugar_beet/sugar_beet_chl.fasta"
    # path_plant_genome = "/home/gsukhorukov/plant_db/sugar_beet/sugar_beet_genome.fasta"
    # path_plant_cds = "/home/gsukhorukov/plant_db/sugar_beet/sugar_beet_cds.fasta"
    # plant = "tomato_reduced_viruses"
    # path_plant_chl = "/home/gsukhorukov/plant_db/tomato/tomato_chl.fasta"
    # path_plant_genome = "/home/gsukhorukov/plant_db/tomato/tomato_genome.fasta"
    # path_plant_cds = "/home/gsukhorukov/plant_db/tomato/tomato_cds.fasta"
    ds = "grapevine_reduced_viruses"
    plant = 'grapevine'
    path_plant_chl = "/home/gsukhorukov/plant_db/grapevine/grapevine_chl.fasta"
    path_plant_genome = "/home/gsukhorukov/plant_db/grapevine/grapevine_genome.fasta"
    path_plant_cds = "/home/gsukhorukov/plant_db/grapevine/grapevine_cds.fasta"

    path_bact = "/home/gsukhorukov/bact_db/refseq_2021-10-29"
    out_path = "/home/gsukhorukov/classifier/data/train"

    # fragment_length = 1000
    n_cpus = 8
    for fragment_length in 500, 1000:
        sl_wind_step = int(fragment_length / 2)
        prepare_ds(
            path_virus=path_viruses,
            ds=ds,
            plant=plant,
            path_plant_genome=path_plant_genome,
            path_plant_cds=path_plant_cds,
            path_plant_chl=path_plant_chl,
            out_path=out_path,
            fragment_length=fragment_length,
            path_bact=path_bact,
            n_cpus=n_cpus,
            random_seed=5,
        )
