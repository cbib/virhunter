import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
    # loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))
import numpy as np
from sklearn.utils import shuffle
import wandb
import h5py
import random
from pathlib import Path
from utils.batch_loader import BatchLoader, BatchGenerator
import pandas as pd
from models import model_5, model_7, model_10
import predict_new as pr
import train_ml


# for offline wandb work
# os.environ["WANDB_MODE"] = "offline"


def fetch_batches(fragments, fragments_rc, labels, random_seed, batch_size, train_fr):
    # for reverse contigs we use half a batch size, as the second half is filled with reverse fragments
    batch_size_ = int(batch_size / 2)
    # shuffle and separate into train and validation
    # complicated approach to prepare balanced batches
    random.seed(a=random_seed)
    n_seqs = int(fragments.shape[0] / 3)
    ind_0 = shuffle([i for i in range(0, n_seqs)], random_state=random.randrange(1000000))
    ind_1 = shuffle([i for i in range(n_seqs, 2 * n_seqs)], random_state=random.randrange(1000000))
    ind_2 = shuffle([i for i in range(2 * n_seqs, 3 * n_seqs)], random_state=random.randrange(1000000))
    batches = []
    # memory of what class was used in the last batch
    last_class = [0, 0, 0]
    # how many do we need to add to the batch
    to_add = batch_size_ % 3
    # how many to take from each class
    ind_n_ = batch_size_ // 3
    # how many times batches we do we have
    for i in range(n_seqs * 3 // batch_size_):
        batch_idx = []
        # trick to reduce batches in turn, should work for any batch size
        # based on modulo from batch number divided by 3
        # we balance batches in turns (batches are even, and we have 3 classes)
        # class to add at the end of the batch
        class_to_add = i % 3
        # depending on batch size we change ind_n vector
        if to_add == 2:
            ind_n = [ind_n_ + 1] * 3
            ind_n[class_to_add] -= 1
        elif to_add == 1:
            ind_n = [ind_n_] * 3
            ind_n[class_to_add] += 1
        else:
            ind_n = [ind_n_] * 3
        # creating a batch from earlier coordinates and ind_n vector
        batch_idx.extend(ind_0[last_class[0]:last_class[0] + ind_n[0]])
        batch_idx.extend(ind_1[last_class[1]:last_class[1] + ind_n[1]])
        batch_idx.extend(ind_2[last_class[2]:last_class[2] + ind_n[2]])
        batches.append(batch_idx)
        # updating starting coordinate for the i array, so that we do not lose any entries
        last_class = [x + y for x, y in zip(last_class, ind_n)]
    print("Finished preparation of balanced batches")

    assert (1.0 - train_fr) > 0.0

    train_batches = batches[:int(len(batches) * train_fr)]
    val_batches = batches[int(len(batches) * train_fr):]
    # train_batches = batches[:60]
    # val_batches = batches[60:80]

    train_gen = BatchLoader(fragments, fragments_rc, labels, train_batches, rc=True,
                            random_seed=random.randrange(1000000))
    val_gen = BatchLoader(fragments, fragments_rc, labels, val_batches, rc=True, random_seed=random.randrange(1000000))
    return train_gen, val_gen


def subset_df(df_path, org, thr=0.8, final_df_size=20000):
    if thr == 1.0:
        df = pd.read_csv(df_path, sep=',')
        df_1 = df.sample(n=final_df_size)
    else:
        df = pd.read_csv(df_path, sep=',')
        df_1 = df.query(f'pred_{org}_5 <= {thr} | pred_{org}_7 <= {thr} | pred_{org}_10 <= {thr}')
        print(df_1.shape[0])
        if df_1.shape[0] < int(final_df_size/2):
            print('too little bad predictions')
            df_1 = df
        df_1 = df_1.sample(n=int(final_df_size/2))
        df_2 = df.query(f'pred_{org}_5 > {thr} & pred_{org}_7 > {thr} & pred_{org}_10 > {thr}')
        df_2 = df_2.sample(n=int(final_df_size/2))
        df_1 = df_1.append(df_2, sort=False)
    return df_1


def train(
        ds_path,
        out_path,
        length,
        test_nn_paths,
        test_ml_paths,
        n_cpus=1,
        epochs=10,
        batch_size=256,
        n_runs=1,
        random_seed=None,
        train_only_clf=False,
        train_only_clf_nn_path=None,
):

    assert Path(out_path).is_dir(), 'out_path was not provided correctly'

    # initializing random generator with the random seed
    random.seed(a=random_seed)

    # callbacks for training of models
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto',
            min_delta=0.000001, cooldown=3, ),
        wandb.keras.WandbCallback(),
    ]

    for n_ in range(n_runs):
        # if we want just to train classifier
        if train_only_clf:
            weights = [
                Path(train_only_clf_nn_path, f"model_5_it_{n_}.h5"),
                Path(train_only_clf_nn_path, f"model_7_it_{n_}.h5"),
                Path(train_only_clf_nn_path, f"model_10_it_{n_}.h5"),
            ]
        else:
            try:
                f = h5py.File(ds_path, "r")
                fragments = f["fragments"]
                fragments_rc = f["fragments_rc"]
                labels = f["labels"]
            except FileNotFoundError:
                raise Exception("dataset was not found. Change ds_path or launch prepare_ds script")
            print(f'using {random_seed} random_seed in batch generation')
            train_gen, val_gen = fetch_batches(fragments,
                                               fragments_rc,
                                               labels,
                                               random_seed=random_seed,
                                               batch_size=batch_size,
                                               train_fr=0.9)

            models_list = zip(["model_5", "model_7", "model_10"], [model_5, model_7, model_10])

            for model_, model_obj in models_list:
                model = model_obj.model(length)
                model.summary()
                model.fit(x=train_gen,
                          validation_data=val_gen,
                          epochs=epochs,
                          callbacks=callbacks,
                          batch_size=batch_size,
                          verbose=2)
                # taking into account validation data
                model.fit(x=val_gen,
                          epochs=1,
                          callbacks=[wandb.keras.WandbCallback()],
                          batch_size=batch_size,
                          verbose=2)
                model.save_weights(Path(out_path, f"{model_}_it_{n_}.h5"))
                print(f'finished training of {n_} iteration of {model_} network')

            weights = [
                Path(out_path, f"model_5_it_{n_}.h5"),
                Path(out_path, f"model_7_it_{n_}.h5"),
                Path(out_path, f"model_10_it_{n_}.h5"),
            ]
        print('predictions for test dataset')
        for org_lab, org_ds in zip(['virus', 'plant', 'bacteria'], test_nn_paths):
            df = pr.predict(
                weights=weights,
                length=length,
                ds=org_ds,
                n_cpus=4,
                batch_size=256
            )
            name = f"pred-{org_lab}_it_{n_}.csv"
            saving_path = Path(out_path, name)
            df.to_csv(saving_path)

        # subsetting predictions
        df_v = subset_df(Path(out_path, f"pred-virus_it_{n_}.csv"), 'vir', thr=0.8)
        df_pl = subset_df(Path(out_path, f"pred-plant_it_{n_}.csv"), 'plant', thr=1.0)
        df_b = subset_df(Path(out_path, f"pred-bacteria_it_{n_}.csv"), 'bact', thr=1.0)

        print('training ml classifier')
        df_train, _ = train_ml.merge_ds(path_ds_v=df_v,
                                        path_ds_pl=df_pl,
                                        path_ds_b=df_b,
                                        fract=0.2,
                                        rs=random_seed, loaded=True)
        df_train = df_train.append(_, sort=False)
        _ = train_ml.fit_clf(df_train, Path(out_path, f"RF_n{n_}.joblib"), random_seed)

        print('predictions on real datasets')
        for ds_p in test_ml_paths:
            df_out = pr.predict(weights, length, ds_p, n_cpus=n_cpus, batch_size=batch_size)
            df_out = pr.ensemble_predict(df_out, Path(out_path, f"RF_n{n_}.joblib"), aggregate=False)
            pred_file = Path(out_path, f"{Path(ds_p).stem}_n{n_}.csv")
            df_out.to_csv(pred_file)
        # changing random seed for the next run
        random_seed += 1
    return out_path



if __name__ == "__main__":
    # plant = "peach"
    # plant = 'tomato'
    plant = 'peach'
    plant_samples = 'peach'
    for length in [500, 1000]:
        train_ds_path = f"/home/gsukhorukov/classifier/data/train/encoded_train_{plant}_{length}.hdf5"
        test_ml_paths = [
            '/home/gsukhorukov/real_test/n_r_rc/LIB1_TGCGGCGT-TACCGAGG-AH2C7TDRXY_L001 (paired, trimmed pairs) contig list.fa',
            '/home/gsukhorukov/real_test/n_r_rc/LIB2_CATAATAC-CGTTAGAA-AH2C7TDRXY_L001 (paired, trimmed pairs) contig list.fa',
            '/home/gsukhorukov/real_test/n_r_rc/LIB3_GATCTATC-AGCCTCAT-AH2C7TDRXY_L001 (paired, trimmed pairs) contig list.fa',
        ]
        test_nn_paths = [
            f'/home/gsukhorukov/classifier/data/train/samples_train_rfc/seqs_virus_sampled_{length}_100000.fasta',
            f'/home/gsukhorukov/classifier/data/train/samples_train_rfc/seqs_{plant_samples}_sampled_{length}_100000.fasta',
            f'/home/gsukhorukov/classifier/data/train/samples_train_rfc/seqs_bacteria_sampled_{length}_100000.fasta',
        ]
        train(
            ds_path=train_ds_path,
            out_path="",
            length=length,
            n_cpus=4,
            epochs=10,
            batch_size=256,
            n_runs=1,
            test_nn_paths=test_nn_paths,
            test_ml_paths=test_ml_paths,
        )
