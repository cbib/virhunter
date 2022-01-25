from Bio import SeqIO
from pathlib import Path
import preprocess as pp
import prepare_ds_new as pp_ds


l = [
    '/home/gsukhorukov/vir_db/leave_out/plant-virus_Caulimoviridae_2021-10-26_fragmented-1000-500.fasta'
]

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

n_cpus = 4
fragment_length = 1000
sl_wind_step = 500
for leave_out_ds_name in leave_out_ds_names:
    encoded, encoded_rc, labs, seqs, _ = prepare_ds_fragmenting(
        in_seq=in_path, label=label, label_int=label_int, fragment_length=fragment_length,
        sl_wind_step=sl_wind_step, max_gap=0.05, n_cpus=4)
    pp_ds.storing_encoded(v_encoded, v_encoded_rc, v_labs,
                    out_path=Path(out_path, ))
    SeqIO.write(pl_seqs, Path(out_path, 'fixed_parts', f"seqs_bacteria_{fragment_length}.fasta"), "fasta")
