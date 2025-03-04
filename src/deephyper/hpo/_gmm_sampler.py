import numpy as np
import pandas as pd

from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    IntegerHyperparameter,
    FloatHyperparameter,
    OrdinalHyperparameter,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder


class GMMSampler:
    def __init__(self, config_space):
        self.config_space = config_space

        # Check variable types
        self.categorical_cols = []
        self.ordinal_cols = []
        self.integer_cols = []
        self.float_cols = []
        for hp_name in self.config_space.keys():
            hp = self.config_space[hp_name]
            if isinstance(hp, CategoricalHyperparameter):
                self.categorical_cols.append(hp_name)
            elif isinstance(hp, OrdinalHyperparameter):
                self.ordinal_cols.append(hp_name)
            elif isinstance(hp, IntegerHyperparameter):
                self.integer_cols.append(hp_name)
            elif isinstance(hp, FloatHyperparameter):
                self.float_cols.append(hp_name)
            else:
                raise ValueError(f"Incompatible hyperparameter {hp}")
        self.numerical_cols = self.integer_cols + self.float_cols

        self.categorical_encoder = None
        self.ordinal_encoder = None
        self.numerical_encoder = None
        self.gmm = None

    def fit(self, df: pd.DataFrame):
        categorical_categories = []
        for hp_name in self.categorical_cols:
            hp = self.config_space[hp_name]
            categorical_categories.append(list(hp.choices))
        self.categorical_encoder = OneHotEncoder(
            categories=categorical_categories, sparse_output=False
        )

        ordinal_categories = []
        for hp_name in self.ordinal_cols:
            hp = self.config_space[hp_name]
            ordinal_categories.append(list(hp.sequence))
        self.ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)

        self.numerical_encoder = StandardScaler()

        X_cat = self.categorical_encoder.fit_transform(df[self.categorical_cols].values)
        X_ord = self.ordinal_encoder.fit_transform(df[self.ordinal_cols].values)
        X_num = self.numerical_encoder.fit_transform(df[self.numerical_cols].values)

        self.n_X_cat = X_cat.shape[1]
        self.n_X_ord = X_ord.shape[1]
        self.n_X_num = X_num.shape[1]

        X = np.hstack([X_cat, X_ord, X_num])

        self.gmm = GaussianMixture(n_components=5)
        self.gmm.fit(X)

    def sample(self, n_samples: int) -> pd.DataFrame:
        X = self.gmm.sample(n_samples)[0]

        # Enforce constraints for each variable

        # Categorical
        X_cat = self.categorical_encoder.inverse_transform(X[:, : self.n_X_cat])

        # Ordinal
        X_ord = X[:, self.n_X_cat : self.n_X_cat + self.n_X_ord]
        for i, hp_name in enumerate(self.ordinal_cols):
            categories = self.ordinal_encoder.categories_[i]
            X_ord[:, i] = np.clip(X_ord[:, i], a_min=0, a_max=len(categories) - 1).astype(int)
        X_ord = self.ordinal_encoder.inverse_transform(X_ord)

        # Numerical
        X_num = self.numerical_encoder.inverse_transform(X[:, self.n_X_cat + self.n_X_ord :])
        for i, hp_name in enumerate(self.numerical_cols):
            hp = self.config_space[hp_name]
            X_num[:, i] = np.clip(X_num[:, i], a_min=hp.lower, a_max=hp.upper)

        # Integer
        X_num[:, : len(self.integer_cols)] = np.round(X_num[:, : len(self.integer_cols)]).astype(
            int
        )

        X = np.hstack([X_cat, X_ord, X_num])
        df = (
            pd.DataFrame(
                data=X,
                columns=self.categorical_cols + self.ordinal_cols + self.numerical_cols,
            )
            .astype({k: int for k in self.integer_cols})
            .astype({k: float for k in self.float_cols})
        )

        return df
