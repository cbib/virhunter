import os
import pandas as pd
import numpy as np
from pathlib import Path
import wandb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from joblib import dump, load
from sklearn.base import BaseEstimator, ClassifierMixin

plt.rcParams.update({'font.size': 20,
                     'figure.figsize': (12, 8)})


def load_ds(path_ds, label, family=None, loaded=False):
    if loaded:
        df = path_ds
    else:
        df = pd.read_csv(path_ds, sep=',')
    df = df.drop(['length', 'fragment'], 1)
    df["label"] = label
    if family is not None:
        df["family"] = family
    return df


def load_ds_old(path_ds, plant, fam, org, label, n):
    if fam is None:
        df = pd.read_csv(Path(path_ds, f"pred-{org}_it_{n}.csv"), sep=',')
    else:
        df = pd.read_csv(Path(path_ds, f"{plant}_{fam}_pred-{org}_it_{n}.csv"), sep=',')
    df = df.drop(['length', 'fragment'], 1)
    df["label"] = label
    df["family"] = fam
    return df


def merge_ds(path_ds_v, path_ds_pl, path_ds_b, fract, rs, family=None, loaded=False):
    df_vir = load_ds(path_ds_v, label=1, family=family, loaded=loaded)
    df_plant = load_ds(path_ds_pl, label=0, family=family, loaded=loaded)
    df_bact = load_ds(path_ds_b, label=2, family=family, loaded=loaded)
    # balancing datasets by downsampling to the size of the smallest dataset
    l_ = min(df_vir.shape[0], df_bact.shape[0], df_plant.shape[0])
    df_vir = df_vir.sample(frac=l_ / df_vir.shape[0], random_state=rs)
    df_plant = df_plant.sample(frac=l_ / df_plant.shape[0], random_state=rs)
    df_bact = df_bact.sample(frac=l_ / df_bact.shape[0], random_state=rs)
    # splitting on train and test
    df_vir_train, df_vir_test = train_test_split(df_vir, test_size=fract, shuffle=False)
    df_plant_train, df_plant_test = train_test_split(df_plant, test_size=fract, shuffle=False)
    df_bact_train, df_bact_test = train_test_split(df_bact, test_size=fract, shuffle=False)
    # merging dfs
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    df_train = df_train.append([df_vir_train, df_plant_train, df_bact_train], ignore_index=True)
    df_test = df_test.append([df_vir_test, df_plant_test, df_bact_test], ignore_index=True)
    return df_train, df_test


def merge_ds_old(path_ds, plant, fam, fract, rs, n=0):
    df_vir = load_ds_old(path_ds, plant, fam, org="virus", label=1, n=n)
    df_plant = load_ds_old(path_ds, plant, fam, org="plant", label=0, n=n)
    df_bact = load_ds_old(path_ds, plant, fam, org="bacteria", label=2, n=n)
    # balancing datasets by downsampling to the size of the smallest dataset
    l_ = min(df_vir.shape[0], df_bact.shape[0], df_plant.shape[0])
    df_vir = df_vir.sample(frac=l_ / df_vir.shape[0], random_state=rs)
    df_plant = df_plant.sample(frac=l_ / df_plant.shape[0], random_state=rs)
    df_bact = df_bact.sample(frac=l_ / df_bact.shape[0], random_state=rs)
    # splitting on train and test
    df_vir_train, df_vir_test = train_test_split(df_vir, test_size=fract, shuffle=False)
    df_plant_train, df_plant_test = train_test_split(df_plant, test_size=fract, shuffle=False)
    df_bact_train, df_bact_test = train_test_split(df_bact, test_size=fract, shuffle=False)
    # merging dfs
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    df_train = df_train.append([df_vir_train, df_plant_train, df_bact_train], ignore_index=True)
    df_test = df_test.append([df_vir_test, df_plant_test, df_bact_test], ignore_index=True)
    return df_train, df_test


def merge_ds_leave_out(families_dict, plant, fract, rs, n=0):
    fam_dfs_train = []
    fam_dfs_test = []
    for fam in sorted(families_dict.keys()):
        # adding together prediction for plant, virus, bacteria
        df_train, df_test = merge_ds_old(families_dict[fam], plant, fam, fract, rs, n=n)
        fam_dfs_train.append(df_train)
        fam_dfs_test.append(df_test)
        # precaution to sample different seqs from plant and bact dataset
        rs += 1
    fam_df_train = pd.DataFrame()
    fam_df_test = pd.DataFrame()
    fam_df_train = fam_df_train.append(fam_dfs_train, ignore_index=True)
    fam_df_test = fam_df_test.append(fam_dfs_test, ignore_index=True)
    return fam_df_train, fam_df_test


def fit_clf(df, save_path, rs):
    df_reshuffled = df.sample(frac=1, random_state=rs)
    X = df_reshuffled[["pred_plant_5", "pred_vir_5", "pred_plant_7", "pred_vir_7", "pred_plant_10", "pred_vir_10", ]]
    y = df_reshuffled["label"]
    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, max_samples=0.3)
    clf.fit(X, y)
    dump(clf, save_path)
    return clf


def plot_confm(df, clf, save_path, rs):
    df_reshuffled = df.sample(frac=1, random_state=rs)
    X_test = df_reshuffled[["pred_plant_5", "pred_vir_5", "pred_plant_7", "pred_vir_7", "pred_plant_10", "pred_vir_10", ]]
    y_test = df_reshuffled["label"]
    plot_confusion_matrix(
        clf, X_test, y_test,
        display_labels=["plant", "virus", "bacteria"],
        cmap=plt.cm.Blues,
        normalize='true',
    )
    plt.savefig(save_path, dpi=300)


def calc_confm(df, clf, rs):
    df_reshuffled = df.sample(frac=1, random_state=rs)
    X_test = df_reshuffled[["pred_plant_5", "pred_vir_5", "pred_plant_7", "pred_vir_7", "pred_plant_10", "pred_vir_10", ]]
    y_pred = np.array(clf.predict(X_test))
    y_pred = np.append(y_pred, [0, 1, 2])
    # mapping = {0: "plant", 1: "virus", 2: "bacteria"}
    # y_pred = np.vectorize(mapping.get)(y_pred)
    y_test = np.array(df_reshuffled["label"])
    y_test = np.append(y_test, [0, 1, 2])
    confm = confusion_matrix(
        y_test, y_pred,
        labels = [0, 1, 2],
        normalize = 'true'
    ).round(decimals=3)
    return confm


def plot_confm_fams(df, families, clf, save_path, rs, ):
    # hardcoded grid with 9 families
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(24, 18))
    for j in range(len(families)):
        ax = axes[j // 3][j % 3]
        df_ = df[df["family"] == families[j]]
        df_reshuffled = df_.sample(frac=1, random_state=rs)
        X_test = df_reshuffled[
            ["pred_plant_5", "pred_vir_5", "pred_plant_7", "pred_vir_7", "pred_plant_10", "pred_vir_10", ]]
        y_test = df_reshuffled["label"]
        plot_confusion_matrix(
            clf, X_test, y_test, ax=ax,
            display_labels=["plant", "virus", "bacteria"],
            cmap=plt.cm.Blues,
            normalize='true',
        )
        ax.set_title(f"tested on {families[j]} from test")
    # fig.delaxes(axes[2][2])
    fig.tight_layout()
    plt.savefig(save_path, dpi=100)


class TemplateClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
        self.classes_ = np.array([0, 1, 2])

    def predict(self, X):
        X = X.to_numpy()
        return self.classes_[np.argmax(X, axis=1)]


def plot_confm_one_mod(df, mod, save_path, plant, rs, n=0):
    df_reshuffled = df.sample(frac=1, random_state=rs)
    X_test = df_reshuffled[
        [f"pred_plant_{mod}", f"pred_vir_{mod}", f"pred_bact_{mod}"]]
    y_test = df_reshuffled["label"]
    clf = TemplateClassifier()
    plot_confusion_matrix(
        clf, X_test, y_test,
        display_labels=["plant", "virus", "bacteria"],
        cmap=plt.cm.Blues,
        normalize='true',
    )
    plt.savefig(Path(save_path, f"confm_{plant}_model-{mod}_n{n}.png"), dpi=300)


def plot_confm_fams_one_mod(df, mod, families, save_path, plant, rs, n=0):
    # hardcoded grid with 8 families
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(24, 18))
    for j in range(len(families)):
        ax = axes[j // 3][j % 3]
        df_ = df[df["family"] == families[j]]
        df_reshuffled = df_.sample(frac=1, random_state=rs)
        X_test = df_reshuffled[
            [f"pred_plant_{mod}", f"pred_vir_{mod}", f"pred_bact_{mod}"]]
        y_test = df_reshuffled["label"]
        clf = TemplateClassifier()
        plot_confusion_matrix(
            clf, X_test, y_test, ax=ax,
            display_labels=["plant", "virus", "bacteria"],
            cmap=plt.cm.Blues,
            normalize='true',
        )
        ax.set_title(f"tested on {families[j]} from test")
    fig.delaxes(axes[2][2])
    plt.savefig(Path(save_path, f"confm-separate_{plant}_model-{mod}_n{n}.png"), dpi=100)


if __name__ == "__main__":

    # plant = "1000_peach-cds-g-chl"
    # families_dict = {
    #     "Closteroviridae": "/home/gsukhorukov/classifier/wandb/run-20211103_023525-1sxvtr30/files",
    #     "Caulimoviridae": "/home/gsukhorukov/classifier/wandb/run-20211102_200729-23g1zraf/files",
    #     "Phenuiviridae": "/home/gsukhorukov/classifier/wandb/run-20211103_084827-2su9dxag/files",
    #     "Unclassified": "/home/gsukhorukov/classifier/wandb/run-20211102_200807-3idmwiyr/files",
    #     "Potyviridae": "/home/gsukhorukov/classifier/wandb/run-20211103_091705-2tv1vfyy/files",
    #     "Tolecusatellitidae": "/home/gsukhorukov/classifier/wandb/run-20211102_194619-hbmcfenr/files",
    #     "Genomoviridae": "/home/gsukhorukov/classifier/wandb/run-20211103_023944-3vespe21/files",
    #     "Nanoviridae": "/home/gsukhorukov/classifier/wandb/run-20211103_021807-9398oroq/files",
    #     # "Species": "/home/gsukhorukov/classifier/wandb/run-20211103_084405-1iu1fvxo/files"
    # }

    # plant = "1000_peach-cds-g-chl"
    # families_dict = {
    #     "Closteroviridae": "/home/gsukhorukov/classifier/wandb/run-20211111_193304-2kye15es/files",
    #     "Caulimoviridae": "/home/gsukhorukov/classifier/wandb/run-20211111_114432-a0truanj/files",
    #     "Phenuiviridae": "/home/gsukhorukov/classifier/wandb/run-20211111_114522-2r7owmzc/files",
    #     "Unclassified": "/home/gsukhorukov/classifier/wandb/run-20211111_194404-3qj8co0q/files",
    #     "Potyviridae": "/home/gsukhorukov/classifier/wandb/run-20211110_233728-2j1qgp2i/files",
    #     "Tolecusatellitidae": "/home/gsukhorukov/classifier/wandb/run-20211111_194930-jkc9aeuw/files",
    #     "Genomoviridae": "/home/gsukhorukov/classifier/wandb/run-20211111_114616-1fvzoo0d/files",
    #     "Nanoviridae": "/home/gsukhorukov/classifier/wandb/run-20211111_114718-y01x2b34/files",
    # }
    #
    # out_path = "/home/gsukhorukov/classifier"
    # exp_name = f"{plant}"
    # random_seed = 3
    # run = wandb.init(
    #     project="virhunter",
    #     entity="gregoruar",
    #     job_type="train-ml",
    #     dir=out_path,
    #     save_code=True,
    # )
    # cf = run.config
    # # variable parameters
    # cf.exp_name = exp_name
    # cf.host = plant
    # # constant parameters, not advised to be changed
    # cf.rundir = run.dir
    # cf.random_seed = random_seed
    # n_runs = 3
    # for n_ in range(n_runs):
    #     for fam in sorted(families_dict.keys()):
    #         df = pd.read_csv(Path(families_dict[fam], f"pred-virus_it_{n_}.csv"), sep=',')
    #         df = df.query('pred_vir_5 < 0.8 | pred_vir_7 < 0.8 | pred_vir_10 < 0.8')
    #         df.to_csv(Path(families_dict[fam], f"pred-virus_it_{n_}_reduced.csv"))
    #
    #         df_train, _ = merge_ds(path_ds_v=Path(families_dict[fam], f"pred-virus_it_{n_}_reduced.csv"),
    #                                path_ds_pl=Path(families_dict[fam], f"pred-plant_it_{n_}.csv"),
    #                                path_ds_b=Path(families_dict[fam], f"pred-bacteria_it_{n_}.csv"),
    #                                fract=0.2,
    #                                rs=random_seed, )
    #         df_train = df_train.append(_, sort=False)
    #         clf = fit_clf(df_train, Path(run.dir, f"RF_{fam}_n{n_}.joblib"), random_seed)
    #         print('predictions on real datasets')
    #         df_test, _ = merge_ds(
    #             path_ds_v=Path(families_dict[fam], f"virus_{fam}_sampled_1000_10000_{n_}.csv"),
    #             path_ds_pl=Path(families_dict[fam], f"peach-cds-g-chl_sampled_1000_10000_{n_}.csv"),
    #             path_ds_b=Path(families_dict[fam], f"bacteria_sampled_1000_10000_{n_}.csv"),
    #             fract=0.2,
    #             rs=random_seed, )
    #         df_test = df_test.append(_, sort=False)
    #         plot_confm(df_test, clf, Path(run.dir, f"confm_{fam}_n{n_}.png"), random_seed, )
    #         random_seed += 1

    plant = "1000_peach-cds-g-chl"
    train_nn_path = '/home/gsukhorukov/classifier/wandb/run-20211120_135013-2dh1v56p/files'
    out_path = "/home/gsukhorukov/classifier"
    exp_name = f"{plant}"
    random_seed = 3
    run = wandb.init(
        project="virhunter",
        entity="gregoruar",
        job_type="train-ml",
        dir=out_path,
        save_code=True,
    )
    cf = run.config
    # variable parameters
    cf.exp_name = exp_name
    cf.host = plant
    # constant parameters, not advised to be changed
    cf.rundir = run.dir
    cf.random_seed = random_seed
    n_runs = 1
    for n_ in range(n_runs):
        thr = 0.8
        adding = '_adding'
        df = pd.read_csv(Path(train_nn_path, f"pred-virus_it_{n_}.csv"), sep=',')
        df_1 = df.query(f'pred_vir_5 <= {thr} | pred_vir_7 <= {thr} | pred_vir_10 <= {thr}')
        if adding == '_adding':
            n_rows = df_1.shape[0]
            df_2 = df.query(f'pred_vir_5 > {thr} & pred_vir_7 > {thr} & pred_vir_10 > {thr}')
            df_2 = df_2.sample(n_rows, replace=False)
            df_1 = df_1.append(df_2, sort=False)
        df_1.to_csv(Path(train_nn_path, f"pred-virus_it_{n_}_reduced_thr_{thr}{adding}.csv"))

        df_train, _ = merge_ds(path_ds_v=Path(train_nn_path, f"pred-virus_it_{n_}_reduced_thr_{thr}{adding}.csv"),
                               path_ds_pl=Path(train_nn_path, f"pred-plant_it_{n_}.csv"),
                               path_ds_b=Path(train_nn_path, f"pred-bacteria_it_{n_}.csv"),
                               fract=0.2,
                               rs=random_seed, )
        df_train = df_train.append(_, sort=False)
        clf = fit_clf(df_train, Path(run.dir, f"RF_n{n_}.joblib"), random_seed)
        random_seed += 1

    # Part with merged classifier, trained on all families
    # n_runs = 3
    # for n_ in range(n_runs):
    #     fam_dfs_train = []
    #     fam_dfs_test = []
    #     for fam in sorted(families_dict.keys()):
    #         # adding together prediction for plant, virus, bacteria
    #         df_train, df_test = merge_ds(
    #             path_ds_v=Path(families_dict[fam], f"virus_{fam}_sampled_1000_10000_{n_}.csv"),
    #             path_ds_pl=Path(families_dict[fam], f"peach-cds-g-chl_sampled_1000_10000_{n_}.csv"),
    #             path_ds_b=Path(families_dict[fam], f"bacteria_sampled_1000_10000_{n_}.csv"),
    #             fract=0.33, rs=random_seed, family=fam)
    #         fam_dfs_train.append(df_train)
    #         fam_dfs_test.append(df_test)
    #         # precaution to sample different seqs from plant and bact dataset
    #         random_seed += 1
    #     fam_df_train = pd.DataFrame()
    #     fam_df_test = pd.DataFrame()
    #     fam_df_train = fam_df_train.append(fam_dfs_train, ignore_index=True)
    #     fam_df_test = fam_df_test.append(fam_dfs_test, ignore_index=True)
    #     clf = fit_clf(fam_df_train, Path(run.dir, f"RF_n{n_}.joblib"), random_seed)
    #     plot_confm(fam_df_test, clf, Path(run.dir, f"confm_test_n{n_}.png"), random_seed,)
    #     plot_confm_fams(fam_df_test, sorted(families_dict.keys()), clf, Path(run.dir, f"confm-separate_n{n_}.png"), random_seed,)
    #     random_seed += 1
    # run.finish()
