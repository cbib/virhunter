import os
from pathlib import Path
from sklearn.utils import shuffle
import h5py
import numpy as np
import prepare_ds as pp_ds
from Bio import SeqIO
from utils import preprocess as pp


def load_ds(ds_path):
    data_path = ds_path
    f = h5py.File(data_path, "r")
    fragments = f["fragments"]
    fragments_rc = f["fragments_rc"]
    labels = f["labels"]
    return fragments, fragments_rc, labels


def prepare_ds_leave_out(plant, path_viruses, path_plant_bact, out_path, leave_out_ds_name, fragment_length=1000):
    n_cpus = 8
    random_seed = 6

    print(f'Generating leave-out train ds for {leave_out_ds_name}')
    v_path = Path(out_path, 'fixed_parts', f'encoded_virus_no-{leave_out_ds_name}_{fragment_length}.hdf5')
    v_path_seqs = Path(out_path, 'fixed_parts', f'seqs_virus_no-{leave_out_ds_name}_{fragment_length}.fasta')
    if v_path.exists() and v_path_seqs.exists():
        v_encoded, v_encoded_rc, v_labs = load_ds(ds_path=v_path)
        v_n_frags = v_encoded.shape[0]
    else:
        path_virus = Path(path_viruses, f'plant-virus_no-{leave_out_ds_name}_2021-10-26.fasta')
        v_encoded, v_encoded_rc, v_labs, v_seqs, v_n_frags = pp.prepare_ds_fragmenting(
            in_seq=path_virus, label='virus', label_int=1, fragment_length=fragment_length,
            sl_wind_step=500, max_gap=0.05, n_cpus=n_cpus)
        # saving virus dataset
        pp.storing_encoded(v_encoded, v_encoded_rc, v_labs, v_path)
        SeqIO.write(v_seqs, v_path_seqs, "fasta")

    pl_encoded, pl_encoded_rc, pl_labs = load_ds(ds_path=Path(path_plant_bact,
                                                              f"encoded_{plant}_{fragment_length}.hdf5"))

    pl_encoded, pl_encoded_rc, pl_labs = shuffle(pl_encoded[()], pl_encoded_rc[()], pl_labs[()],
                                                 random_state=random_seed, n_samples=v_n_frags)
    b_encoded, b_encoded_rc, b_labs = load_ds(ds_path=Path(path_plant_bact,
                                                           f"encoded_bacteria_{fragment_length}.hdf5"))
    b_encoded, b_encoded_rc, b_labs = shuffle(b_encoded[()], b_encoded_rc[()], b_labs[()],
                                              random_state=random_seed + 10, n_samples=v_n_frags)

    all_encoded = np.concatenate([v_encoded, pl_encoded, b_encoded])
    all_encoded_rc = np.concatenate([v_encoded_rc, pl_encoded_rc, b_encoded_rc])
    all_labs = np.concatenate([v_labs, pl_labs, b_labs])

    # adding reverse complement
    all_encoded = np.concatenate((all_encoded, all_encoded_rc))
    all_encoded_rc = np.concatenate((all_encoded_rc, all_encoded))
    all_labs = np.concatenate((all_labs, all_labs))

    ds_path = Path(out_path, f'encoded_{plant}_no-{leave_out_ds_name}_{fragment_length}.hdf5')
    if not ds_path.exists():
        pp.storing_encoded(all_encoded, all_encoded_rc, all_labs, ds_path)


if __name__ == "__main__":
    leave_out_ds_names = [
        'Caulimoviridae',
        'Closteroviridae',
        'Phenuiviridae',
        'Unclassified',
        'Genomoviridae',
        'Potyviridae',
        'Tolecusatellitidae',
        'Nanoviridae',
        'Species',
    ]

    path_viruses = "/home/gsukhorukov/vir_db/leave_out"
    plant = "grapevine"
    path_plant_bact = "/home/gsukhorukov/classifier/data/train_leave_out/fixed_parts"
    out_path = '/home/gsukhorukov/classifier/data/train_leave_out'

    for leave_out_ds_name in leave_out_ds_names:
        prepare_ds_leave_out(plant, path_viruses, path_plant_bact, out_path, leave_out_ds_names)