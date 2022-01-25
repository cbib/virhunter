import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Bio import SeqIO
from pathlib import Path
import ray
from sklearn.utils import shuffle
import preprocess as pp
import numpy as np

# def sampling_test(in_path, n_frags, fragment_length, random_seed):
#     ray.init(num_cpus=n_cpus, num_gpus=0)
#     seqs_list = pp.prepare_seq_lists(in_path, n_frags)
#     it = pp.chunks(seqs_list, int(len(seqs_list) / n_cpus + 1))
#     frs_ = ray.get(
#         [pp.sample_fragments.remote(s, fragment_length, random_seed, limit=10, max_gap=0.05) for s in it])
#     seqs = sum([i[2] for i in frs_], [])
#     ray.shutdown()
#     return seqs


# path_virus = '/home/gsukhorukov/vir_db/plant-virus_all_2021-10-26.fasta'
# path_bact = "/home/gsukhorukov/bact_db/refseq_2021-10-29"
# plant = "peach"
# path_plants = [
#     "/home/gsukhorukov/plant_db/peach/peach_chl.fasta",
#     "/home/gsukhorukov/plant_db/peach/peach_genome.fasta",
#     "/home/gsukhorukov/plant_db/peach/peach_cds.fasta",
# ]
# plant = 'tomato'
# path_plants = [
#     "/home/gsukhorukov/plant_db/tomato/tomato_chl.fasta",
#     "/home/gsukhorukov/plant_db/tomato/tomato_genome.fasta",
#     "/home/gsukhorukov/plant_db/tomato/tomato_cds.fasta",
# ]
# plant = 'grapevine'
# path_plants = [
#     "/home/gsukhorukov/plant_db/grapevine/grapevine_chl.fasta",
#     "/home/gsukhorukov/plant_db/grapevine/grapevine_genome.fasta",
#     "/home/gsukhorukov/plant_db/grapevine/grapevine_cds.fasta",
# ]
plant = 'sugar_beet'
path_plants = [
    "/home/gsukhorukov/plant_db/sugar_beet/sugar_beet_chl.fasta",
    "/home/gsukhorukov/plant_db/sugar_beet/sugar_beet_genome.fasta",
    "/home/gsukhorukov/plant_db/sugar_beet/sugar_beet_cds.fasta",
]
plant_weights = [0.1, 0.6, 0.3]

out_path = "/home/gsukhorukov/classifier/data/train/samples_test_rfc"
random_seed = 6
fragment_length = 500
n_frags = 10000
n_cpus = 4

# # sampling virus
# encoded, encoded_rc, labs, seqs, _ = pp.prepare_ds_sampling(
#     in_seqs=path_virus, fragment_length=fragment_length,
#     n_frags=n_frags, label='virus', label_int=1, random_seed=random_seed, n_cpus=n_cpus)
# encoded, encoded_rc, labs, seqs = shuffle(encoded, encoded_rc, labs, seqs,
#                                           random_state=random_seed, n_samples=n_frags)
# assert len(seqs) == n_frags
# out_path_encoded = Path(out_path, f"encoded_virus_sampled_{fragment_length}_{n_frags}.hdf5")
# pp.storing_encoded(encoded, encoded_rc, labs, out_path_encoded)
# out_path_seqs = Path(out_path, f"seqs_virus_sampled_{fragment_length}_{n_frags}.fasta")
# SeqIO.write(seqs, out_path_seqs, "fasta")

# sampling plant
pl_encoded = []
pl_encoded_rc = []
pl_labels = []
pl_seqs = []
for path_plant, w in zip(path_plants, plant_weights):
    n_frags_1_pl = int(n_frags * w)
    encoded, encoded_rc, labs, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=path_plant, fragment_length=fragment_length,
        n_frags=n_frags_1_pl, label='plant', label_int=0, random_seed=random_seed, n_cpus=n_cpus)
    encoded, encoded_rc, labs, seqs = shuffle(encoded, encoded_rc, labs, seqs,
                                              random_state=random_seed, n_samples=n_frags_1_pl)
    pl_encoded.append(encoded)
    pl_encoded_rc.append(encoded_rc)
    pl_labels.append(labs)
    pl_seqs.extend(seqs)
# force same number of fragments
assert len(pl_seqs) == n_frags
pl_encoded = np.concatenate(pl_encoded)
pl_encoded_rc = np.concatenate(pl_encoded_rc)
pl_labs = np.concatenate(pl_labels)
out_path_encoded = Path(out_path, f"encoded_{plant}_sampled_{fragment_length}_{n_frags}.hdf5")
pp.storing_encoded(pl_encoded, pl_encoded_rc, pl_labs, out_path_encoded)
out_path_seqs = Path(out_path, f"seqs_{plant}_sampled_{fragment_length}_{n_frags}.fasta")
SeqIO.write(pl_seqs, out_path_seqs, "fasta")

# sampling bacteria
# encoded, encoded_rc, labs, seqs, _ = pp.prepare_ds_sampling(
#     in_seqs=path_bact, fragment_length=fragment_length,
#     n_frags=n_frags, label='bacteria', label_int=2, random_seed=random_seed, n_cpus=n_cpus)
# encoded, encoded_rc, labs, seqs = shuffle(encoded, encoded_rc, labs, seqs,
#                                           random_state=random_seed, n_samples=n_frags)
# assert len(seqs) == n_frags
# out_path_encoded = Path(out_path, f"encoded_bacteria_sampled_{fragment_length}_{n_frags}.hdf5")
# pp.storing_encoded(encoded, encoded_rc, labs, out_path_encoded)
# out_path_seqs = Path(out_path, f"seqs_bacteria_sampled_{fragment_length}_{n_frags}.fasta")
# SeqIO.write(seqs, out_path_seqs, "fasta")

# # sampling viruses for leave-out
# leave_out_ds_names = '''Alphaflexiviridae
# Alphasatellitidae
# Amalgaviridae
# Aspiviridae
# Benyviridae
# Betaflexiviridae
# Bromoviridae
# Caulimoviridae
# Closteroviridae
# Endornaviridae
# Fimoviridae
# Geminiviridae
# Genomoviridae
# Kitaviridae
# Mayoviridae
# Nanoviridae
# Partitiviridae
# Phenuiviridae
# Phycodnaviridae
# Potyviridae
# Reoviridae
# Rhabdoviridae
# Secoviridae
# Small_families
# Solemoviridae
# Species
# Tolecusatellitidae
# Tombusviridae
# Tospoviridae
# Tymoviridae
# Unclassified
# Virgaviridae'''.split('\n')
# random_seed = 5
# fragment_length = 1000
# n_frags = 10000
# n_cpus = 4
# path_viruses = "/home/gsukhorukov/vir_db/leave_out"
#
# for n_frags in [10000, 100000]:
#     if n_frags == 10000:
#         out_path = '/home/gsukhorukov/classifier/data/train_leave_out/samples_test_rfc'
#     else:
#         out_path = '/home/gsukhorukov/classifier/data/train_leave_out/samples_train_rfc'
#     for leave_out_ds_name in leave_out_ds_names:
#         print(leave_out_ds_name)
#         if n_frags == 10000:
#             path_virus = str(Path(path_viruses, f'plant-virus_{leave_out_ds_name}_2021-10-26.fasta'))
#         else:
#             path_virus = str(Path(path_viruses, f'plant-virus_no-{leave_out_ds_name}_2021-10-26.fasta'))
#         v_encoded, v_encoded_rc, v_labs, v_seqs, _ = pp.prepare_ds_sampling(
#             in_seqs=path_virus, fragment_length=fragment_length,
#             n_frags=n_frags, label='virus', label_int=1, random_seed=random_seed, n_cpus=n_cpus)
#         v_encoded, v_encoded_rc, v_labs, v_seqs = shuffle(v_encoded, v_encoded_rc, v_labs, v_seqs,
#                                                           random_state=random_seed, n_samples=n_frags)
#         assert len(v_seqs) == n_frags
#         # out_path_encoded = Path(out_path, f"encoded_virus_{leave_out_ds_name}_sampled_{fragment_length}_{n_frags}.hdf5")
#         # pp.storing_encoded(v_encoded, v_encoded_rc, v_labs, out_path_encoded)
#         if n_frags == 10000:
#             out_path_seqs = Path(out_path, f"seqs_virus_{leave_out_ds_name}_sampled_{fragment_length}_{n_frags}.fasta")
#         else:
#             out_path_seqs = Path(out_path, f"seqs_virus_no- {leave_out_ds_name}_sampled_{fragment_length}_{n_frags}.fasta")
#         SeqIO.write(v_seqs, out_path_seqs, "fasta")
