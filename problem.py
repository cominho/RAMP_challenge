import os
import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import GroupShuffleSplit

problem_title = "Ranking des assets par return"

_target_column_name = "target"
_ignore_column_names = ["date"]

Predictions = rw.prediction_types.make_regression()

workflow = rw.workflows.Estimator()

# Fonction NDCG pour un groupe

def ndcg_group(y_true, y_pred):
    """
    Calcule le NDCG pour un groupe.
    Décale les valeurs si des retours négatifs sont présents.
    """
    offset = 0
    if np.min(y_true) < 0:
        offset = -np.min(y_true)
    y_true_adj = y_true + offset

    #ordre décroissant selon y_pred
    order = np.argsort(-y_pred)
    dcg = np.sum((2 ** y_true_adj[order] - 1) / np.log2(np.arange(2, len(y_true_adj) + 2)))
    
    # DCG idéal (IDCG)
    ideal_order = np.argsort(-y_true_adj)
    idcg = np.sum((2 ** y_true_adj[ideal_order] - 1) / np.log2(np.arange(2, len(y_true_adj) + 2)))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


class NDCGScore(rw.score_types.BaseScoreType):
    name = "ndcg"
    precision = 4
    greater_is_better = True

    def __call__(self, y_true, y_pred):
        # on suppose que chaque groupe contient exactement 39 observations
        group_size = 39
        n = len(y_true)
        ndcg_vals = []
        for i in range(0, n, group_size):
            yt = y_true[i:i+group_size]
            yp = y_pred[i:i+group_size]
            ndcg_vals.append(ndcg_group(yt, yp))
        return np.mean(ndcg_vals)

score_types = [NDCGScore()]


def get_cv(X, y):
    groups = np.arange(len(y)) // 39
    splitter = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    return splitter.split(X, y, groups)


def _read_data(path, file_name):
    """
    Lit un fichier CSV depuis le dossier 'data', sépare la cible (return) des features,
    et supprime les colonnes à ignorer.
    """
    df = pd.read_csv(os.path.join(path, "data", file_name))
    y = df[_target_column_name].values

    columns_to_drop = [_target_column_name] + _ignore_column_names
    X = df.drop(columns=columns_to_drop, errors='ignore')
    return X, y

def get_train_data(path="."):
    return _read_data(path, "train.csv")

def get_test_data(path="."):
    return _read_data(path, "test.csv")
