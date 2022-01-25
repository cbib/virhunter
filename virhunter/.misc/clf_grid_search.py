import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from joblib import dump, load
import fire
import train_ml


def one_run(
        v_train_in,
        pl_train_in,
        b_train_in,
        v_test_in,
        pl_test_in,
        b_test_in,
        clf_out,
        pred_out_all,
        pred_out,
        max_features,
        n_estimators,
        max_samples,
        max_depth,
        random_seed=5,
):
    df_train, _ = train_ml.merge_ds(path_ds_v=v_train_in,
                                    path_ds_pl=pl_train_in,
                                    path_ds_b=b_train_in,
                                    fract=0.2,
                                    rs=random_seed, )
    df_train = df_train.append(_, sort=False)
    df_reshuffled = df_train.sample(frac=1, random_state=random_seed)
    X = df_reshuffled[["pred_plant_5", "pred_vir_5", "pred_bact_5",
                       "pred_plant_7", "pred_vir_7", "pred_bact_7",
                       "pred_plant_10", "pred_vir_10", "pred_bact_10",
                       ]]
    y = df_reshuffled["label"]
    clf = RandomForestClassifier(max_depth=int(max_depth), n_estimators=int(n_estimators), max_features=int(max_features),
                                 max_samples=float(max_samples))
    clf.fit(X, y)
    dump(clf, Path(clf_out,
                   f"max_features_{max_features}_n_estimators_{n_estimators}_max_samples_{max_samples}_max_depth_{max_depth}.joblib"))
    df_test, _ = train_ml.merge_ds(
        path_ds_v=v_test_in,
        path_ds_pl=pl_test_in,
        path_ds_b=b_test_in,
        fract=0.2,
        rs=random_seed, )
    df_test = df_test.append(_, sort=False)
    X_test = df_test[["pred_plant_5", "pred_vir_5", "pred_bact_5",
                      "pred_plant_7", "pred_vir_7", "pred_bact_7",
                      "pred_plant_10", "pred_vir_10", "pred_bact_10",
                      ]]
    y_test = df_test["label"]
    y_pred = np.array(clf.predict(X_test))
    conf_matrix = np.round((confusion_matrix(y_test, y_pred) / 10000).flatten(), 3)
    with open(pred_out_all, 'a') as f:
        s_1 = f"max_features_{max_features}_n_estimators_{n_estimators}_max_samples_{max_samples}_max_depth_{max_depth}\t"
        s_2 = '\t'.join(str(x) for x in conf_matrix)
        f.write(s_1 + s_2 + '\n')
    with open(Path(pred_out, f"max_features_{max_features}_n_estimators_{n_estimators}_max_samples_{max_samples}_max_depth_{max_depth}.csv"), 'w') as f:
        s_1 = f"max_features_{max_features}_n_estimators_{n_estimators}_max_samples_{max_samples}_max_depth_{max_depth}\t"
        s_2 = '\t'.join(str(x) for x in conf_matrix)
        f.write(s_1 + s_2)


if __name__ == '__main__':
    fire.Fire(one_run)
    # one_run(
    #     v_train_in='/mnt/cbib/virhunter/grid_search/in_Caulimoviridae/pred-virus_it_0.csv',
    #     pl_train_in='/mnt/cbib/virhunter/grid_search/in_Caulimoviridae/pred-plant_it_0.csv',
    #     b_train_in='/mnt/cbib/virhunter/grid_search/in_Caulimoviridae/pred-bacteria_it_0.csv',
    #     v_test_in='/mnt/cbib/virhunter/grid_search/in_Caulimoviridae/virus_Caulimoviridae_sampled_1000_10000_0.csv',
    #     pl_test_in='/mnt/cbib/virhunter/grid_search/in_Caulimoviridae/peach-cds-g-chl_sampled_1000_10000_0.csv',
    #     b_test_in='/mnt/cbib/virhunter/grid_search/in_Caulimoviridae/bacteria_sampled_1000_10000_0.csv',
    #     clf_out='/mnt/cbib/virhunter/grid_search/clf_out',
    #     pred_out_all='/mnt/cbib/virhunter/grid_search/pred_out_all_Caulimoviridae.csv',
    #     pred_out='/mnt/cbib/virhunter/grid_search/pred_out',
    #     max_features=5,
    #     n_estimators=10,
    #     max_samples=10,
    #     max_depth=10,
    #     random_seed=5,
    # )