from Bio import SeqIO
from pathlib import Path


def masking_leave_out(leave_out_seqs_p, masked_seqs_p, out_file_p):
    lo_ids = [seq.id for seq in list(SeqIO.parse(leave_out_seqs_p, "fasta"))]
    masked_seqs = list(SeqIO.parse(masked_seqs_p, "fasta"))
    container = []
    for seq in masked_seqs:
        if seq.id in lo_ids:
            container.append(seq)
    SeqIO.write(container, out_file_p, "fasta")
    print("finished selecting sequences")


if __name__ == '__main__':
    path_viruses = [
        # "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_NO-Caulimoviridae_2020-01-04.fasta",
        # "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_NO-Closteroviridae_2020-01-04.fasta",
        # "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_NO-Phenuiviridae_2020-01-04.fasta",
        # "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_NO-Unclassified_2020-01-04.fasta",
        # "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_NO-Genomoviridae_2020-01-04.fasta",
        # "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_NO-Potyviridae_2020-01-04.fasta",
        # "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_NO-Tolecusatellitidae_2020-01-04.fasta",
        # "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_NO-Nanoviridae_2020-01-04.fasta",
        # "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Caulimoviridae_2020-01-04.fasta",
        # "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Closteroviridae_2020-01-04.fasta",
        # "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Phenuiviridae_2020-01-04.fasta",
        "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Unclassified_2020-01-04.fasta",
        "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Genomoviridae_2020-01-04.fasta",
        "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Potyviridae_2020-01-04.fasta",
        "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Tolecusatellitidae_2020-01-04.fasta",
        "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Nanoviridae_2020-01-04.fasta",
    ]

    for p_ in path_viruses:
        p_ = Path(p_)
        masking_leave_out(
            p_,
            '/home/gsukhorukov/plant_vir_all_2020-01-04_masked.fna',
            p_.parent / f'masked_{p_.name}'
        )
