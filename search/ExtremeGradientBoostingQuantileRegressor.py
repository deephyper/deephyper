import numpy as np
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import check_random_state
from sklearn.externals.joblib import Parallel, delayed
import numpy as np
from scipy.stats import binom_test
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost.sklearn import XGBRegressor
from functools import partial

def _parallel_fit(regressor, X, y):
    return regressor.fit(X, y)

def quantile_loss(y_true, y_pred,_alpha,_delta,_threshold,_var): 
    x = y_true - y_pred 
    grad = (x<(_alpha-1.0)*_delta)*(1.0-_alpha)- ((x>=(_alpha-1.0)*_delta)& (x<_alpha*_delta) )*x/_delta-_alpha*(x>_alpha*_delta) 
    hess = ((x>=(_alpha-1.0)*_delta)& (x<_alpha*_delta) )/_delta 
    _len = np.array([y_true]).size 
    var = (2*np.random.randint(2, size=_len)-1.0)*_var 
    grad = (np.abs(x)<_threshold )*grad - (np.abs(x)>=_threshold )*var 
    hess = (np.abs(x)<_threshold )*hess + (np.abs(x)>=_threshold ) 
    return grad, hess

class ExtremeGradientBoostingQuantileRegressor(BaseEstimator, RegressorMixin):
    """Predict several quantiles with one estimator.

    This is a wrapper around `GradientBoostingRegressor`'s quantile
    regression that allows you to predict several `quantiles` in
    one go.

    Parameters
    ----------
    * `quantiles` [array-like]:
        Quantiles to predict. By default the 16, 50 and 84%
        quantiles are predicted.

    * `base_estimator` [GradientBoostingRegressor instance or None (default)]:
        Quantile regressor used to make predictions. Only instances
        of `GradientBoostingRegressor` are supported. Use this to change
        the hyper-parameters of the estimator.

    * `n_jobs` [int, default=1]:
        The number of jobs to run in parallel for `fit`.
        If -1, then the number of jobs is set to the number of cores.

    * `random_state` [int, RandomState instance, or None (default)]:
        Set random state to something other than None for reproducible
        results.
    """

    def __init__(self, quantiles=[0.05, 0.16, 0.5, 0.84, 0.95], base_estimator=None, n_jobs=1, random_state=None):
        self.quantiles = quantiles
        self.random_state = random_state
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
        
        self.quant_alpha_lower = 0.16
        self.quant_delta_lower = 1.0 
        self.quant_thres_lower = 5.0 
        self.quant_var_lower = 4.2
 
        self.quant_alpha_upper = 0.84
        self.quant_delta_upper = 1.0 
        self.quant_thres_upper = 6.0 
        self.quant_var_upper = 3.2
        
        #xgboost parameters
        n_estimators = 100
        max_depth = 5
        reg_alpha = 5.
        reg_lambda=1.0
        gamma=0.5
        self.n_estimators = n_estimators 
        self.max_depth = max_depth 
        self.reg_alpha= reg_alpha 
        self.reg_lambda = reg_lambda 
        self.gamma = gamma 

        
    def fit(self, X, y): 
        self.clf_lower = XGBRegressor(objective=partial(quantile_loss,_alpha = self.quant_alpha_lower,_delta = self.quant_delta_lower,_threshold = self.quant_thres_lower,_var = self.quant_var_lower),
                                n_estimators = self.n_estimators, max_depth = self.max_depth, reg_alpha =self.reg_alpha, reg_lambda = self.reg_lambda, gamma = self.gamma)
        self.clf_upper = XGBRegressor(objective=partial(quantile_loss,_alpha = self.quant_alpha_upper,_delta = self.quant_delta_upper,_threshold = self.quant_thres_upper,_var = self.quant_var_upper),
                                n_estimators = self.n_estimators, max_depth = self.max_depth, reg_alpha =self.reg_alpha, reg_lambda = self.reg_lambda, gamma = self.gamma)
        self.clf = XGBRegressor(n_estimators = self.n_estimators, max_depth = self.max_depth, reg_alpha =self.reg_alpha, reg_lambda = self.reg_lambda, gamma = self.gamma)
        self.clf_lower.fit(X,y) 
        self.clf_upper.fit(X,y)
        self.clf.fit(X,y)
        return self        
        
    def fit(self, X, y):
        """Fit one regressor for each quantile.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        * `y` [array-like, shape=(n_samples,)]:
            Target values (real numbers in regression)
        """
        rng = check_random_state(self.random_state)

        if self.base_estimator is None:
            base_estimator = XGBRegressor(n_estimators = self.n_estimators, max_depth = self.max_depth, reg_alpha =self.reg_alpha, reg_lambda = self.reg_lambda, gamma = self.gamma)
        else:
            base_estimator = self.base_estimator

            if not isinstance(base_estimator, XGBRegressor):
                raise ValueError('base_estimator has to be of type'
                                 ' XGBRegressor.')
        # The predictions for different quantiles should be sorted.
        # Therefore each of the regressors need the same seed.
        #print(base_estimator.get_params().keys())
        #base_estimator.set_params(random_state=rng)
        regressors = []
        for q in self.quantiles:
            if q == 0.05 or q == 0.16:
                regressor = XGBRegressor(objective=partial(quantile_loss,_alpha = q,_delta = self.quant_delta_lower,_threshold = self.quant_thres_lower,_var = self.quant_var_lower), 
                                         n_estimators = self.n_estimators, max_depth = self.max_depth, reg_alpha =self.reg_alpha, reg_lambda = self.reg_lambda, gamma = self.gamma)
            elif q == 0.84 or q == 0.95:
                regressor = XGBRegressor(objective=partial(quantile_loss,_alpha = q,_delta = self.quant_delta_upper,_threshold = self.quant_thres_upper,_var = self.quant_var_upper),
                                         n_estimators = self.n_estimators, max_depth = self.max_depth, reg_alpha =self.reg_alpha, reg_lambda = self.reg_lambda, gamma = self.gamma)
            
            elif q == 0.50:
                regressor = clone(base_estimator)
            
            regressors.append(regressor)

        self.regressors_ = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(_parallel_fit)(regressor, X, y)
            for regressor in regressors)

        return self
    
    def feature_importances_(self):
        model = self.regressors_[self.quantiles.index(0.5)]
        score = model.booster().get_score()
        return score
    
    def predict(self, X, return_std=False, return_quantiles=False):
        """Predict.
        Predict `X` at every quantile if `return_std` is set to False.
        If `return_std` is set to True, then return the mean
        and the predicted standard deviation, which is approximated as
        the (0.84th quantile - 0.16th quantile) divided by 2.0

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            where `n_samples` is the number of samples
            and `n_features` is the number of features.
        """
        predicted_quantiles = np.asarray(
            [rgr.predict(X) for rgr in self.regressors_])
        if return_quantiles:
            return np.asarray(predicted_quantiles.T)

        elif return_std:
            std_quantiles = [0.16, 0.5, 0.84] #[0.05, 0.5, 0.95] #
            is_present_mask = np.in1d(std_quantiles, self.quantiles)
            if not np.all(is_present_mask):
                raise ValueError(
                    "return_std works only if the quantiles during "
                    "instantiation include 0.16, 0.5 and 0.84")
            low = self.regressors_[self.quantiles.index(0.16)].predict(X)
            high = self.regressors_[self.quantiles.index(0.84)].predict(X)
            mean = self.regressors_[self.quantiles.index(0.5)].predict(X)
            #print('===>{}; {}; {}'.format(low, mean, high))
            return mean, ((high - low) / 2.0)

        # return the mean
        return self.regressors_[self.quantiles.index(0.5)].predict(X)
