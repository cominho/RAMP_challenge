import numpy as np
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

class XGBRankerWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper pour xgboost.XGBRanker.
    Transforme automatiquement les returns continus en scores de pertinence entiers par groupe.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.kwargs = kwargs

    def _transform_labels(self, y, group):
        """
        Pour chaque groupe (de taille donnée dans 'group'), transforme les valeurs continues de y
        en scores de pertinence entiers.
        Le meilleur return reçoit le score max (groupe - 1), le second reçoit (groupe - 2), etc...
        """
        y_transformed = np.empty_like(y, dtype=int)
        start = 0
        for g in group:
            group_indices = np.arange(start, start+g)
            group_y = y[group_indices]
            #on trie par ordre décroissant : le plus grand return en premier
            order = np.argsort(group_y)[::-1]
            ranks = np.empty(g, dtype=int)
            for rank, idx in enumerate(order):
                ranks[idx] = g - 1 - rank
            y_transformed[group_indices] = ranks
            start += g
        return y_transformed

    def fit(self, X, y, group=None):
        if group is None:
            group_size = 39
            n_groups = len(y) // group_size
            group = [group_size] * n_groups

        y_transformed = self._transform_labels(y, group)

        self.model_ = xgb.XGBRanker(
            objective="rank:ndcg",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            ndcg_exp_gain=False,  
            **self.kwargs
        )
        self.model_.fit(X, y_transformed, group=group)
        return self

    def predict(self, X):
        return self.model_.predict(X)

def get_estimator():
    """
    Retourne un pipeline scikit-learn comprenant :
      - une imputation médiane,
      - une standardisation des features,
      - et notre modèle XGBRankerWrapper.
    """
    pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        XGBRankerWrapper(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    )
    return pipeline
