import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import ray
import os
import random
import utils.preprocess as pp
from sklearn.utils import shuffle


random_seed = 1
n_fragments = 10000
n_cpus = 8
path_virus = "/home/gsukhorukov/plant_vir_all_2020-01-04.fasta"
# path_phage = "/mnt/cbib/classifier/vir_db/phages_2020-02-08.fasta"
path_bact = "/home/gsukhorukov/masked_bacteria"
plant1 = "peach"
path_plant1 = "/mnt/cbib/baracuda/gsukhorukov/plant_db/data/GCF_000346465.2",
# plant2 = "apple"
# path_plant2 = "/mnt/cbib/inextvir/workspace/gregoruar/deepvirfinder/dvf_train/plant_db/data/GCF_002114115.1"
out_path = "/home/gsukhorukov/classifier/data/sampled_test_fragments_3"
mut = True
mut_rate = 0.0

for fragment_length in [250, 500, 750, 1000, 1500, 2000, 3000, 6000]:
    ray.init(num_cpus=n_cpus, num_gpus=0)
    # sampling virus fragments
    _, _, viral_seqs = ray.get(pp.sample_fragments.remote(
        [[path_virus, n_fragments]], fragment_length, random_seed, max_gap=0.05
    ))
    if mut:
        viral_seqs = pp.introduce_mutations(viral_seqs, mut_rate)
        out_path_seqs = os.path.join(out_path, f"viral_{fragment_length}_{str(int(mut_rate*100))}prc.fasta")
    else:
        out_path_seqs = os.path.join(out_path, f"viral_{fragment_length}.fasta")
    SeqIO.write(viral_seqs, out_path_seqs, "fasta")
    print(f"Sampling {n_fragments} viral sequences with length {fragment_length} finished")
    ray.shutdown()

    # ray.init(num_cpus=n_cpus, num_gpus=0)
    # # sampling phage fragments
    # _, _, phage_seqs = ray.get(pp.sample_fragments.remote(
    #     [[path_phage, n_fragments]], fragment_length, random_seed, max_gap=0.05
    # ))
    # if mut:
    #     phage_seqs = pp.introduce_mutations(phage_seqs, mut_rate)
    #     out_path_seqs = os.path.join(out_path, f"phage_{fragment_length}_{str(int(mut_rate*100))}prc.fasta")
    # else:
    #     out_path_seqs = os.path.join(out_path, f"phage_{fragment_length}.fasta")
    # SeqIO.write(phage_seqs, out_path_seqs, "fasta")
    # print(f"Sampling {n_fragments} phage sequences with length {fragment_length} finished")
    # ray.shutdown()

    ray.init(num_cpus=n_cpus, num_gpus=0)
    # sampling plant1 fragments
    plant_seqs_list = pp.prepare_seq_lists(path_plant1, n_fragments)
    it = pp.chunks(plant_seqs_list, int(len(plant_seqs_list)/n_cpus+1))
    plant_frs_ = ray.get([pp.sample_fragments.remote(s, fragment_length, random_seed, max_gap=0.05) for s in it])
    plant_seqs = sum([i[2] for i in plant_frs_], [])
    # have n_fragments seqs
    plant_seqs = random.sample(plant_seqs, n_fragments)
    if mut:
        plant_seqs = pp.introduce_mutations(plant_seqs, mut_rate)
        out_path_seqs = os.path.join(out_path, f"{plant1}_{fragment_length}_{str(int(mut_rate*100))}prc.fasta")
    else:
        out_path_seqs = os.path.join(out_path, f"{plant1}_{fragment_length}.fasta")
    SeqIO.write(plant_seqs, out_path_seqs, "fasta")
    print(f"Sampling {n_fragments} {plant1} fragments with length {fragment_length} finished")
    ray.shutdown()

    # ray.init(num_cpus=n_cpus, num_gpus=0)
    # # sampling plant2 fragments
    # plant_seqs_list = pp.prepare_seq_lists(path_plant2, n_fragments)
    # it = pp.chunks(plant_seqs_list, int(len(plant_seqs_list)/n_cpus+1))
    # plant_frs_ = ray.get([pp.sample_fragments.remote(s, fragment_length, random_seed, max_gap=0.05) for s in it])
    # plant_seqs = sum([i[2] for i in plant_frs_], [])
    # # have n_fragments seqs
    # plant_seqs = random.sample(plant_seqs, n_fragments)
    # if mut:
    #     plant_seqs = pp.introduce_mutations(plant_seqs, mut_rate)
    #     out_path_seqs = os.path.join(out_path, f"{plant2}_{fragment_length}_{str(int(mut_rate*100))}prc.fasta")
    # else:
    #     out_path_seqs = os.path.join(out_path, f"{plant2}_{fragment_length}.fasta")
    # SeqIO.write(plant_seqs, out_path_seqs, "fasta")
    # print(f"Sampling {n_fragments} {plant2} fragments with length {fragment_length} finished")
    # ray.shutdown()

    ray.init(num_cpus=n_cpus, num_gpus=0)
    # generating bacterial fragments and labels
    bact_seqs_list = pp.prepare_seq_lists(path_bact, n_fragments)
    it = pp.chunks(bact_seqs_list, int(len(bact_seqs_list)/n_cpus+1))
    bact_frs_ = ray.get([pp.sample_fragments.remote(s, fragment_length, random_seed, max_gap=0.05) for s in it])
    bact_seqs = sum([i[2] for i in bact_frs_], [])
    # have n_fragments seqs
    bact_seqs = random.sample(bact_seqs, n_fragments)
    if mut:
        bact_seqs = pp.introduce_mutations(bact_seqs, mut_rate)
        out_path_seqs = os.path.join(out_path, f"bact_{fragment_length}_{str(int(mut_rate*100))}prc.fasta")
    else:
        out_path_seqs = os.path.join(out_path, f"bact_{fragment_length}.fasta")
    SeqIO.write(bact_seqs, out_path_seqs, "fasta")
    print(f"Sampling {n_fragments} bacterial sequences with length {fragment_length} finished")
    ray.shutdown()
