import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Bio import SeqIO
from pathlib import Path
import ray
from sklearn.utils import shuffle
import preprocess as pp
import numpy as np


path_virus = '/home/gsukhorukov/vir_db/plant-virus_all_2021-10-26.fasta'
out_path = "/home/gsukhorukov/classifier/data/train/samples_contigs"
random_seed = 6
n_frags = 10000
n_cpus = 4

# for fragment_length in [1000, 1500, 2000, 3000, 4500, 6000]:
#     # sampling virus
#     encoded, encoded_rc, labs, seqs, _ = pp.prepare_ds_sampling(
#         in_seqs=path_virus, fragment_length=fragment_length,
#         n_frags=n_frags, label='virus', label_int=1, random_seed=random_seed, n_cpus=n_cpus, limit=20)
#     encoded, encoded_rc, labs, seqs = shuffle(encoded, encoded_rc, labs, seqs,
#                                               random_state=random_seed, n_samples=n_frags)
#     assert len(seqs) == n_frags
#     out_path_seqs = Path(out_path, f"seqs_virus_sampled_{fragment_length}_{n_frags}.fasta")
#     SeqIO.write(seqs, out_path_seqs, "fasta")
#     for mut_rate in [0.05, 0.1, 0.15]:
#         seqs_ = seqs
#         seqs_ = pp.introduce_mutations(seqs_, mut_rate, rs=None)
#         out_path_seqs = Path(out_path, f"seqs_virus_sampled_mut_{mut_rate}_{fragment_length}_{n_frags}.fasta")
#         SeqIO.write(seqs_, out_path_seqs, "fasta")


# for fragment_length in [1000, 1500, 2000, 3000, 4500, 6000]:
#     # sampling virus
#     encoded, encoded_rc, labs, seqs, _ = pp.prepare_ds_sampling(
#         in_seqs=path_virus, fragment_length=fragment_length,
#         n_frags=n_frags, label='virus', label_int=1, random_seed=random_seed, n_cpus=n_cpus, limit=20)
#     encoded, encoded_rc, labs, seqs = shuffle(encoded, encoded_rc, labs, seqs,
#                                               random_state=random_seed, n_samples=n_frags)
#     assert len(seqs) == n_frags
#     out_path_seqs = Path(out_path, f"seqs_virus_sampled_{fragment_length}_{n_frags}.fasta")
#     SeqIO.write(seqs, out_path_seqs, "fasta")
#     for mut_rate in [0.05, 0.1, 0.15]:
#         seqs_ = seqs
#         seqs_ = pp.introduce_mutations(seqs_, mut_rate, rs=None)
#         out_path_seqs = Path(out_path, f"seqs_virus_sampled_mut_{mut_rate}_{fragment_length}_{n_frags}.fasta")
#         SeqIO.write(seqs_, out_path_seqs, "fasta")

# plant = "peach"
# path_plant = "/home/gsukhorukov/plant_db/peach/peach_genome.fasta"
# plant = 'grapevine'
# path_plant = "/home/gsukhorukov/plant_db/grapevine/grapevine_genome.fasta"
plant = 'sugar_beet'
path_plant = "/home/gsukhorukov/plant_db/sugar_beet/sugar_beet_genome.fasta"


for fragment_length in [1000, 1500, 2000, 3000, 4500, 6000]:
    print()
    print(f'fragment length {fragment_length}')
    # sampling plant
    encoded, encoded_rc, labs, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=path_plant, fragment_length=fragment_length,
        n_frags=n_frags, label='plant', label_int=0, random_seed=random_seed, n_cpus=n_cpus)
    encoded, encoded_rc, labs, seqs = shuffle(encoded, encoded_rc, labs, seqs,
                                              random_state=random_seed, n_samples=n_frags)

    out_path_seqs = Path(out_path, f"seqs_{plant}_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(seqs, out_path_seqs, "fasta")
    for mut_rate in [0.05, 0.1, 0.15]:
        seqs_ = seqs
        seqs_ = pp.introduce_mutations(seqs_, mut_rate, rs=None)
        out_path_seqs = Path(out_path, f"seqs_{plant}_sampled_mut_{mut_rate}_{fragment_length}_{n_frags}.fasta")
        SeqIO.write(seqs_, out_path_seqs, "fasta")

# path_bact = "/home/gsukhorukov/bact_db/refseq_2021-10-29"
# for fragment_length in [1000, 1500, 2000, 3000, 4500, 6000]:
#     # sampling bacteria
#     encoded, encoded_rc, labs, seqs, _ = pp.prepare_ds_sampling(
#         in_seqs=path_bact, fragment_length=fragment_length,
#         n_frags=n_frags, label='bacteria', label_int=2, random_seed=random_seed, n_cpus=n_cpus)
#     encoded, encoded_rc, labs, seqs = shuffle(encoded, encoded_rc, labs, seqs,
#                                               random_state=random_seed, n_samples=n_frags)
#     assert len(seqs) == n_frags
#     out_path_seqs = Path(out_path, f"seqs_bacteria_sampled_{fragment_length}_{n_frags}.fasta")
#     SeqIO.write(seqs, out_path_seqs, "fasta")
#     for mut_rate in [0.05, 0.1, 0.15]:
#         seqs_ = seqs
#         seqs_ = pp.introduce_mutations(seqs_, mut_rate, rs=None)
#         out_path_seqs = Path(out_path, f"seqs_bacteria_sampled_mut_{mut_rate}_{fragment_length}_{n_frags}.fasta")
#         SeqIO.write(seqs_, out_path_seqs, "fasta")