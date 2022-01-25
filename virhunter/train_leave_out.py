import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# special lifehack to detect gpus
print(tf.config.list_physical_devices('GPU'))
import numpy as np
import wandb
from pathlib import Path
from train import train, subset_df
from predict import predict
import train_ml
from joblib import load
import pandas as pd
from prepare_ds_leave_out import prepare_ds_leave_out

# limiting cpus usage. x2 multiplier to go from cpus to threads
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(6)

if __name__ == "__main__":
    plant = "peach"
    length = 1000
    path_viruses = "/home/gsukhorukov/vir_db/leave_out"
    path_plant_bact = "/home/gsukhorukov/classifier/data/train_leave_out/fixed_parts"
    out_path = '/home/gsukhorukov/classifier/data/train_leave_out'
    leave_out_ds_names = [
        # 'Alphaflexiviridae',
        # 'Alphasatellitidae',
        # 'Amalgaviridae',
        # 'Aspiviridae',
        # 'Benyviridae',
        # 'Betaflexiviridae',
        # 'Bromoviridae',
        # 'Endornaviridae',
        # 'Caulimoviridae',
        # 'Closteroviridae',
        # 'Fimoviridae',
        # 'Geminiviridae',
        # 'Genomoviridae',
        # 'Kitaviridae',
        # 'Mayoviridae',
        # 'Nanoviridae',
        # 'Partitiviridae',
        # 'Phenuiviridae',
        # 'Phycodnaviridae',
        # 'Potyviridae',
        'Reoviridae',
        'Rhabdoviridae',
        'Secoviridae',
        # 'Small_families',
        # 'Solemoviridae',
        # 'Species',
        # 'Tolecusatellitidae',
        # 'Tombusviridae',
        # 'Tospoviridae',
        # 'Tymoviridae',
        # 'Unclassified',
        # 'Virgaviridae',
        ]
    random_seed = 23
    fragment_length = 1000
    for leave_out_ds_name in leave_out_ds_names:
        print(leave_out_ds_name)
        print('preparing dataset')
        prepare_ds_leave_out(plant, path_viruses, path_plant_bact, out_path, leave_out_ds_name, fragment_length)
        print('starting training')
        test_ds_paths = [
            f'/home/gsukhorukov/classifier/data/train_leave_out/samples_train_rfc/seqs_virus_no-{leave_out_ds_name}_sampled_1000_100000.fasta',
            f'/home/gsukhorukov/classifier/data/train_leave_out/samples_train_rfc/seqs_{plant}_sampled_1000_100000.fasta',
            '/home/gsukhorukov/classifier/data/train_leave_out/samples_train_rfc/seqs_bacteria_sampled_1000_100000.fasta',
        ]
        real_ds_paths = [
            f'/home/gsukhorukov/classifier/data/train_leave_out/samples_test_rfc/seqs_virus_{leave_out_ds_name}_sampled_1000_10000.fasta',
            f'/home/gsukhorukov/classifier/data/train_leave_out/samples_test_rfc/seqs_{plant}_sampled_1000_10000.fasta',
            '/home/gsukhorukov/classifier/data/train_leave_out/samples_test_rfc/seqs_bacteria_sampled_1000_10000.fasta',
        ]
        job_type = "train_leave-out"
        train_ds_path = f'/home/gsukhorukov/classifier/data/train_leave_out/encoded_{plant}_no-{leave_out_ds_name}_1000.hdf5'
        n_runs = 3
        run = train(
            ds_path=train_ds_path,
            out_path="/home/gsukhorukov/classifier",
            plant=plant,
            length=length,
            exp_name=f'{length}_{plant}_{leave_out_ds_name}',
            n_cpus=4,
            epochs=10,
            batch_size=256,
            n_runs=n_runs,
            test_nn_paths=test_ds_paths,
            test_ml_paths=real_ds_paths,
            job_type=job_type,
            run_finish=False
        )
        print(plant)
        print(leave_out_ds_name)
        confms = []
        for n_ in range(n_runs):
            df_, _ = train_ml.merge_ds(
                path_ds_v=Path(run.dir, f"seqs_virus_{leave_out_ds_name}_sampled_1000_10000_n{n_}.csv"),
                path_ds_pl=Path(run.dir, f"seqs_{plant}_sampled_1000_10000_n{n_}.csv"),
                path_ds_b=Path(run.dir, f"seqs_bacteria_sampled_1000_10000_n{n_}.csv"),
                fract=0.2,
                rs=random_seed, )
            df_test = df_.append(_, sort=False)
            clf = load(Path(run.dir, f"RF_n{n_}.joblib"))
            train_ml.plot_confm(df_test, clf, Path(run.dir, f"confm_{leave_out_ds_name}_n{n_}.png"),
                                random_seed, )
            confm = train_ml.calc_confm(df_test, clf, random_seed)
            confms.append(confm)
            print(n_)
            print(confm)
            random_seed += 1
        print('average')
        print(np.mean(np.array(confms), axis=0).round(3))
        run.finish()
        print('removing leave out ds')
        ds_path = Path(out_path, f'encoded_{plant}_no-{leave_out_ds_name}_{fragment_length}.hdf5')
        ds_path.unlink()
    