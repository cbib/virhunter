import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from Bio import SeqIO
import pandas as pd
import ray
import wandb
from virhunter.utils import preprocess as pp
from pathlib import Path
from virhunter.models import model_5, model_7, model_10
from joblib import load


def predict_one(model, length, ds, n_cpus=4, batch_size=256):
    """
    this function does prediction from one specified model
    model - tf object with already loaded weights
    may not work correctly, should be rewritten
    """
    print(f"loading sequences")
    try:
        test_seqs = list(SeqIO.parse(ds, "fasta"))
    except FileNotFoundError:
        raise Exception("dataset was not found. Change ds variable")

    print(f"Separating sequences according to length")
    seqs_, _ = pp.separate_by_length(length, test_seqs, fold=None, )

    print(f"generating viral fragments and labels")
    out_table = {
        "id": [],
        "length": [],
        "fragment": [],
        "pred_plant": [],
        "pred_vir": [],
        "pred_bact": [],
    }
    if seqs_:
        test_fragments = []
        test_fragments_rc = []
        ray.init(num_cpus=n_cpus, num_gpus=0)
        for seq in seqs_:
            fragments_, fragments_rc, _ = pp.fragmenting([seq], length, max_gap=0.05,
                                                         sl_wind_step=int(length / 2))
            test_fragments.extend(fragments_)
            test_fragments_rc.extend(fragments_rc)
            for j in range(len(fragments_)):
                out_table["id"].append(seq.id)
                out_table["length"].append(len(seq.seq))
                out_table["fragment"].append(j)
        it = pp.chunks(test_fragments, int(len(test_fragments) / n_cpus + 1))
        test_encoded = np.concatenate(ray.get([pp.one_hot_encode.remote(s) for s in it]))
        it = pp.chunks(test_fragments_rc, int(len(test_fragments_rc) / n_cpus + 1))
        test_encoded_rc = np.concatenate(ray.get([pp.one_hot_encode.remote(s) for s in it]))
        print(f"Encoding sequences finished")
        print(
            f"{np.shape(test_encoded)[0]} + {np.shape(test_encoded_rc)[0]} fragments generated")
        ray.shutdown()

        print(f"Starting sequence prediction")
        prediction = model.predict([test_encoded, test_encoded_rc], batch_size)
        out_table["pred_plant"].extend(list(prediction[..., 0]))
        out_table["pred_vir"].extend(list(prediction[..., 1]))
        out_table["pred_bact"].extend(list(prediction[..., 2]))
    else:
        raise ValueError("All sequences were smaller than length of the model")
    print(f"Exporting predictions to csv file")
    df_out = pd.DataFrame(out_table)
    return df_out


def predict_family(model, length, ds, multi_out=False, rc=True, n_cpus=4, batch_size=256):
    ds_dict = {
        "Caulimoviridae": "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Caulimoviridae_2020-01-04_fragmented-1000-500.fasta",
        "Closteroviridae": "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Closteroviridae_2020-01-04_fragmented-1000-500.fasta",
        "Phenuiviridae": "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Phenuiviridae_2020-01-04_fragmented-1000-500.fasta",
        "Unclassified": "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Unclassified_2020-01-04_fragmented-1000-500.fasta",
        "Genomoviridae": "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Genomoviridae_2020-01-04_fragmented-1000-500.fasta",
        "Potyviridae": "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Potyviridae_2020-01-04_fragmented-1000-500.fasta",
        "Tolecusatellitidae": "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Tolecusatellitidae_2020-01-04_fragmented-1000-500.fasta",
        "Nanoviridae": "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Nanoviridae_2020-01-04_fragmented-1000-500.fasta",
        "Species": "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Species_2020-01-04_fragmented-1000-500.fasta",
        "Sequences": "/home/gsukhorukov/classifier/data/families/_seqs/plant-vir_Sequences_2020-01-04_fragmented-1000-500.fasta",
        "tomato": "/home/gsukhorukov/classifier/data/families/_seqs/tomato_fragmented-1000_500000.fasta",
        "bacteria": "/home/gsukhorukov/classifier/data/families/_seqs/bacteria_fragmented-1000_500000.fasta",
        "apple": "/home/gsukhorukov/classifier/data/families/_seqs/apple_fragmented-1000_500000.fasta",
    }

    print(f"loading sequences")
    try:
        test_seqs = list(SeqIO.parse(ds_dict[ds], "fasta"))
    except FileNotFoundError:
        raise Exception("dataset was not found. Change ds variable")

    print(f"Separating sequences according to length")
    seqs_, _ = pp.separate_by_length(length, test_seqs, fold=None, )

    print(f"generating viral fragments and labels")
    if multi_out:
        out_table = {
            "id": [],
            "length": [],
            "fragment": [],
            "pred_plant": [],
            "pred_vir": [],
            "pred_bact": [],
            "pred_plant_1": [],
            "pred_vir_1": [],
            "pred_bact_1": [],
            "pred_plant_2": [],
            "pred_vir_2": [],
            "pred_bact_2": [],
        }
    else:
        out_table = {
            "id": [],
            "length": [],
            "fragment": [],
            "pred_plant": [],
            "pred_vir": [],
            "pred_bact": [],
        }
    if seqs_:
        test_fragments = []
        test_fragments_rc = []
        ray.init(num_cpus=n_cpus, num_gpus=0)
        for seq in seqs_:
            fragments_, fragments_rc, _ = pp.fragmenting([seq], length, max_gap=0.05,
                                                         sl_wind_step=int(length / 2))
            test_fragments.extend(fragments_)
            test_fragments_rc.extend(fragments_rc)
            for j in range(len(fragments_)):
                out_table["id"].append(seq.id)
                out_table["length"].append(len(seq.seq))
                out_table["fragment"].append(j)
        it = pp.chunks(test_fragments, int(len(test_fragments) / n_cpus + 1))
        test_encoded = np.concatenate(ray.get([pp.one_hot_encode.remote(s) for s in it]))
        it = pp.chunks(test_fragments_rc, int(len(test_fragments_rc) / n_cpus + 1))
        test_encoded_rc = np.concatenate(ray.get([pp.one_hot_encode.remote(s) for s in it]))
        print(f"Encoding sequences finished")
        print(
            f"{np.shape(test_encoded)[0]} + {np.shape(test_encoded_rc)[0]} fragments generated")
        ray.shutdown()

        print(f"Starting sequence prediction")
        if rc:
            prediction = model.predict([test_encoded, test_encoded_rc], batch_size)
        else:
            prediction = model.predict(test_encoded, batch_size)
        if multi_out:
            out_table["pred_plant"].extend(list(prediction[0][..., 0]))
            out_table["pred_vir"].extend(list(prediction[0][..., 1]))
            out_table["pred_bact"].extend(list(prediction[0][..., 2]))
            out_table["pred_plant_1"].extend(list(prediction[1][..., 0]))
            out_table["pred_vir_1"].extend(list(prediction[1][..., 1]))
            out_table["pred_bact_1"].extend(list(prediction[1][..., 2]))
            out_table["pred_plant_2"].extend(list(prediction[2][..., 0]))
            out_table["pred_vir_2"].extend(list(prediction[2][..., 1]))
            out_table["pred_bact_2"].extend(list(prediction[2][..., 2]))
        else:
            out_table["pred_plant"].extend(list(prediction[..., 0]))
            out_table["pred_vir"].extend(list(prediction[..., 1]))
            out_table["pred_bact"].extend(list(prediction[..., 2]))
    else:
        raise ValueError("All sequences were smaller than length of the model")
    print(f"Exporting predictions to csv file")
    df_out = pd.DataFrame(out_table)
    return df_out


def predict_samples(weights, length, ds, n_cpus=4, batch_size=256):
    """
    weights should correspond to models
    """
    print(f"loading sequences")
    try:
        seqs_ = list(SeqIO.parse(ds, "fasta"))
    except FileNotFoundError:
        raise Exception("dataset was not found. Change ds variable")

    # print(f"Separating sequences according to length")
    # seqs_, _ = pp.separate_by_length(length, test_seqs, fold=None, )

    print(f"generating viral fragments and labels")
    out_table = {
        "id": [],
        "length": [],
        "fragment": [],
        "pred_plant_5": [],
        "pred_vir_5": [],
        "pred_bact_5": [],
        "pred_plant_7": [],
        "pred_vir_7": [],
        "pred_bact_7": [],
        "pred_plant_10": [],
        "pred_vir_10": [],
        "pred_bact_10": [],
    }
    if seqs_:
        test_fragments = []
        test_fragments_rc = []
        ray.init(num_cpus=n_cpus, num_gpus=0)
        for seq in seqs_:
            fragments_, fragments_rc, _ = pp.fragmenting([seq], length, max_gap=0.8,
                                                         sl_wind_step=int(length / 2))
            test_fragments.extend(fragments_)
            test_fragments_rc.extend(fragments_rc)
            for j in range(len(fragments_)):
                out_table["id"].append(seq.id)
                out_table["length"].append(len(seq.seq))
                out_table["fragment"].append(j)
        it = pp.chunks(test_fragments, int(len(test_fragments) / n_cpus + 1))
        test_encoded = np.concatenate(ray.get([pp.one_hot_encode.remote(s) for s in it]))
        it = pp.chunks(test_fragments_rc, int(len(test_fragments_rc) / n_cpus + 1))
        test_encoded_rc = np.concatenate(ray.get([pp.one_hot_encode.remote(s) for s in it]))
        print(f"Encoding sequences finished")
        print(
            f"{np.shape(test_encoded)[0]} + {np.shape(test_encoded_rc)[0]} fragments generated")
        ray.shutdown()

        print(f"Starting sequence prediction")
        for model, w, s in zip([model_5.model(length), model_7.model(length), model_10.model(length)], weights,
                               [5, 7, 10]):
            model.load_weights(w)
            prediction = model.predict([test_encoded, test_encoded_rc], batch_size)
            out_table[f"pred_plant_{s}"].extend(list(prediction[..., 0]))
            out_table[f"pred_vir_{s}"].extend(list(prediction[..., 1]))
            out_table[f"pred_bact_{s}"].extend(list(prediction[..., 2]))
    else:
        raise ValueError("All sequences were smaller than length of the model")
    print(f"Exporting predictions to csv file")
    df_out = pd.DataFrame(out_table)
    return df_out


def predict(weights, length, ds, n_cpus=3, batch_size=256):
    """
    weights - list with paths to trained model weights
    they should go in order: 4, 7, 10
    """
    print(f"loading sequences")
    try:
        seqs_ = list(SeqIO.parse(ds, "fasta"))
    except FileNotFoundError:
        raise Exception("test dataset was not found. Change ds variable")

    # print(f"Separating sequences according to length")
    # seqs_, _ = pp.separate_by_length(length, test_seqs, fold=None, )

    print(f"generating viral fragments and labels")
    out_table = {
        "id": [],
        "length": [],
        "fragment": [],
        "pred_plant_5": [],
        "pred_vir_5": [],
        "pred_bact_5": [],
        "pred_plant_7": [],
        "pred_vir_7": [],
        "pred_bact_7": [],
        "pred_plant_10": [],
        "pred_vir_10": [],
        "pred_bact_10": [],
    }
    if seqs_:
        test_fragments = []
        test_fragments_rc = []
        ray.init(num_cpus=n_cpus, num_gpus=0)
        for seq in seqs_:
            fragments_, fragments_rc, _ = pp.fragmenting([seq], length, max_gap=0.8,
                                                         sl_wind_step=int(length / 2))
            test_fragments.extend(fragments_)
            test_fragments_rc.extend(fragments_rc)
            for j in range(len(fragments_)):
                out_table["id"].append(seq.id)
                out_table["length"].append(len(seq.seq))
                out_table["fragment"].append(j)
        it = pp.chunks(test_fragments, int(len(test_fragments) / n_cpus + 1))
        test_encoded = np.concatenate(ray.get([pp.one_hot_encode.remote(s) for s in it]))
        it = pp.chunks(test_fragments_rc, int(len(test_fragments_rc) / n_cpus + 1))
        test_encoded_rc = np.concatenate(ray.get([pp.one_hot_encode.remote(s) for s in it]))
        print(f"Encoding sequences finished")
        print(
            f"{np.shape(test_encoded)[0]} + {np.shape(test_encoded_rc)[0]} fragments generated")
        ray.shutdown()

        print(f"Starting sequence prediction")
        for model, w, s in zip([model_5.model(length), model_7.model(length), model_10.model(length)], weights,
                               [5, 7, 10]):
            # for model, w, s in zip([model_4.model(length),], weights,
            #                        [4,]):
            model.load_weights(w)
            prediction = model.predict([test_encoded, test_encoded_rc], batch_size)
            out_table[f"pred_plant_{s}"].extend(list(prediction[..., 0]))
            out_table[f"pred_vir_{s}"].extend(list(prediction[..., 1]))
            out_table[f"pred_bact_{s}"].extend(list(prediction[..., 2]))
    else:
        raise ValueError("All sequences were smaller than length of the model")
    print(f"Exporting predictions to csv file")
    return pd.DataFrame(out_table)


def ensemble_predict(df, weights, aggregate=True, quantile=0.5):
    clf = load(weights)
    if aggregate:
        df = df.groupby(["id"], sort=False).quantile(q=quantile)
        df["id"] = df.index
        df = df.reset_index(drop=True)
    X = df[
        ["pred_plant_5", "pred_vir_5", "pred_plant_7", "pred_vir_7", "pred_plant_10", "pred_vir_10", ]]
    y_pred = np.array(clf.predict(X))
    mapping = {0: "plant", 1: "virus", 2: "bacteria"}
    df["predicted"] = np.vectorize(mapping.get)(y_pred)
    return df


if __name__ == '__main__':

    plant = "grapevine"
    model = "/home/gsukhorukov/classifier/wandb/run-20211122_104208-eexc9jk6/files"
    # clf_weights = "/home/gsukhorukov/classifier/wandb/run-20211122_104208-eexc9jk6/files"
    clf_weights = '/home/gsukhorukov/classifier/wandb/run-20211209_175146-27hc4wzz/files'

    ds_paths = [
        '/home/gsukhorukov/classifier/data/train_leave_out/samples_test_rfc/seqs_peach_sampled_1000_10000.fasta',
        '/home/gsukhorukov/classifier/data/train_leave_out/samples_test_rfc/seqs_grapevine_sampled_1000_10000.fasta',
        '/home/gsukhorukov/classifier/data/train_leave_out/samples_test_rfc/seqs_tomato_sampled_1000_10000.fasta',
    ]

    length = 1000
    exp_name = f"{plant}_{length}_predictions"
    n_cpus = 4
    batch_size = 256

    run = wandb.init(
        project="virhunter",
        entity="gregoruar",
        job_type="predict",
        dir="/home/gsukhorukov/classifier",
        save_code=True,
    )

    cf = run.config
    cf.rundir = run.dir

    weights = [
        Path(clf_weights, "model_5_it_0.h5"),
        Path(clf_weights, "model_7_it_0.h5"),
        Path(clf_weights, "model_10_it_0.h5"),
    ]

    print(f"starting predictions")
    for ds_path in ds_paths:
        df_out = predict(weights, length, ds_path, n_cpus=n_cpus, batch_size=batch_size)
        df_out = ensemble_predict(df_out, Path(clf_weights, f"RF_n0.joblib"), aggregate=False)
        s = df_out.shape[0]
        print(df_out.query('predicted == "plant"').shape[0]/s)
        print(df_out.query('predicted == "virus"').shape[0] / s)
        print(df_out.query('predicted == "bacteria"').shape[0] / s)
        print()

        pred_file = Path(run.dir, f"{Path(ds_path).stem}_pred.csv")
        df_out.to_csv(pred_file)
    #
    # for contig_length in [1000, 1500, 2000, 3000, 4500, 6000]:
    #     ds_p = f"/home/gsukhorukov/classifier/data/train/samples_contigs/seqs_virus_sampled_{contig_length}_10000.fasta"
    #     df_out = predict(weights, length, ds_p, n_cpus=n_cpus, batch_size=batch_size)
    #     df_out = ensemble_predict(df_out, Path(clf_weights, f"RF_n0.joblib"), aggregate=False)
    #     pred_file = Path(run.dir, f"{Path(ds_p).stem}.csv")
    #     df_out.to_csv(pred_file)
    #     for mut_rate in [0.05, 0.1, 0.15]:
    #         ds_p = f"/home/gsukhorukov/classifier/data/train/samples_contigs/seqs_virus_sampled_mut_{mut_rate}_{contig_length}_10000.fasta"
    #         df_out = predict(weights, length, ds_p, n_cpus=n_cpus, batch_size=batch_size)
    #         df_out = ensemble_predict(df_out, Path(clf_weights, f"RF_n0.joblib"), aggregate=False)
    #         pred_file = Path(run.dir, f"{Path(ds_p).stem}.csv")
    #         df_out.to_csv(pred_file)
