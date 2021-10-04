from deephyper.core.exceptions import DeephyperRuntimeError
import math
import logging

import ConfigSpace as CS
import ConfigSpace.hyperparameters as csh
import numpy as np
import pandas as pd
import skopt
from deephyper.core.logs.logging import JsonMessage as jm
from deephyper.search.search import Search


class AMBS(Search):
    """Asynchronous Model-Based Search.

    Args:
        problem (HpProblem): [description]
        evaluator (Evaluator): [description]
        random_state ([type], optional): [description]. Defaults to None.
        log_dir (str, optional): [description]. Defaults to ".".
        verbose (int, optional): [description]. Defaults to 0.
        surrogate_model (str, optional): [description]. Defaults to "RF".
        acq_func (str, optional): [description]. Defaults to "LCB".
        kappa (float, optional): [description]. Defaults to 1.96.
        xi (float, optional): [description]. Defaults to 0.001.
        liar_strategy (str, optional): [description]. Defaults to "cl_min".
        n_jobs (int, optional): [description]. Defaults to 1.
    """
    def __init__(
        self,
        problem,
        evaluator,
        random_state: int=None,
        log_dir: str=".",
        verbose: int=0,
        surrogate_model: str="RF",
        acq_func: str="LCB",
        kappa: float=1.96,
        xi: float=0.001,
        liar_strategy: str="cl_min",
        n_jobs: int=1,  # 32 is good for Theta
        **kwargs
    ):

        super().__init__(problem, evaluator, random_state, log_dir, verbose)

        self._n_initial_points = self._evaluator.num_workers
        self._liar_strategy = liar_strategy
        self._fitted = False

        self._opt = None
        self._opt_kwargs = dict(
            dimensions=self._problem.space,
            base_estimator=self.get_surrogate_model(surrogate_model, n_jobs),
            acq_func=acq_func,
            acq_optimizer="sampling",
            acq_func_kwargs={"xi": xi, "kappa": kappa},
            n_initial_points=self._n_initial_points,
            random_state=self._random_state,
        )

    def _setup_optimizer(self):
        if self._fitted:
            self._opt_kwargs["n_initial_points"] = 0
        self._opt = skopt.Optimizer(**self._opt_kwargs)

    def _search(self, max_evals, timeout):

        if self._opt is None:
            self._setup_optimizer()

        num_evals_done = 0

        # Filling available nodes at start
        logging.info(f"Generating {self._evaluator.num_workers} initial points...")
        self._evaluator.submit(self.get_random_batch(size=self._n_initial_points))

        # Main loop
        while max_evals < 0 or num_evals_done < max_evals:
            # Collecting finished evaluations
            new_results = self._evaluator.gather("BATCH", size=1)

            if len(new_results) > 0:

                self._evaluator.dump_evals(log_dir=self._log_dir)

                num_received = len(new_results)
                num_evals_done += num_received

                # Transform configurations to list to fit optimizer
                opt_X = []
                opt_y = []
                for cfg, obj in new_results:
                    x = replace_nan(cfg.values())
                    opt_X.append(x)
                    opt_y.append(-obj)  #! maximizing
                self._opt.tell(opt_X, opt_y)  #! fit: costly
                new_X = self._opt.ask(
                    n_points=len(new_results), strategy=self._liar_strategy
                )

                new_batch = []
                for x in new_X:
                    new_cfg = self.to_dict(x)
                    new_batch.append(new_cfg)

                # submit_childs
                if len(new_results) > 0:
                    self._evaluator.submit(new_batch)

    def get_surrogate_model(self, name: str, n_jobs: int = None):
        """Get a surrogate model from Scikit-Optimize.

        Args:
            name (str): name of the surrogate model.
            n_jobs (int): number of parallel processes to distribute the computation of the surrogate model.

        Raises:
            ValueError: when the name of the surrogate model is unknown.
        """
        accepted_names = ["RF", "ET", "GBRT", "GP", "DUMMY"]
        if not (name in accepted_names):
            raise ValueError(
                f"Unknown surrogate model {name}, please choose among {accepted_names}."
            )

        if name == "RF":
            surrogate = skopt.learning.RandomForestRegressor(n_jobs=n_jobs)
        elif name == "ET":
            surrogate = skopt.learning.ExtraTreesRegressor(n_jobs=n_jobs)
        elif name == "GBRT":
            surrogate = skopt.learning.GradientBoostingQuantileRegressor(n_jobs=n_jobs)
        else:  # for DUMMY and GP
            surrogate = name

        return surrogate

    def return_cond(self, cond, cst_new):
        parent = cst_new.get_hyperparameter(cond.parent.name)
        child = cst_new.get_hyperparameter(cond.child.name)
        if type(cond) == CS.EqualsCondition:
            value = cond.value
            cond_new = CS.EqualsCondition(child, parent, cond.value)
        elif type(cond) == CS.GreaterThanCondition:
            value = cond.value
            cond_new = CS.GreaterThanCondition(child, parent, value)
        elif type(cond) == CS.NotEqualsCondition:
            value = cond.value
            cond_new = CS.GreaterThanCondition(child, parent, value)
        elif type(cond) == CS.LessThanCondition:
            value = cond.value
            cond_new = CS.GreaterThanCondition(child, parent, value)
        elif type(cond) == CS.InCondition:
            values = cond.values
            cond_new = CS.GreaterThanCondition(child, parent, values)
        else:
            print("Not supported type" + str(type(cond)))
        return cond_new

    def return_forbid(self, cond, cst_new):
        if type(cond) == CS.ForbiddenEqualsClause or type(cond) == CS.ForbiddenInClause:
            hp = cst_new.get_hyperparameter(cond.hyperparameter.name)
            if type(cond) == CS.ForbiddenEqualsClause:
                value = cond.value
                cond_new = CS.ForbiddenEqualsClause(hp, value)
            elif type(cond) == CS.ForbiddenInClause:
                values = cond.values
                cond_new = CS.ForbiddenInClause(hp, values)
            else:
                print("Not supported type" + str(type(cond)))
        return cond_new

    def fit_surrogate(self, df):
        """Fit the surrogate model of the search from a checkpoint.

        Args:
            df (str|DataFrame): a checkpoint from a previous search.
        """
        if type(df) is str and df[-4:] == ".csv":
            df = pd.read_csv(df)
        assert isinstance(df, pd.DataFrame)

        self._fitted = True

        if self._opt is None:
            self._setup_optimizer()

        hp_names = self._problem.space.get_hyperparameter_names()
        try:
            x = df[hp_names].values.tolist()
            y = df.objective.tolist()
        except KeyError:
            raise DeephyperRuntimeError("Incompatible dataframe to fit surrogate model of AMBS.")

        self._opt.tell(x, [-yi for yi in y])

    def fit_search_space(self, df):

        if type(df) is str and df[-4:] == ".csv":
            df = pd.read_csv(df)
        assert isinstance(df, pd.DataFrame)

        cst = self._problem.space
        if type(cst) != CS.ConfigurationSpace:
            logging.error(f"{type(cst)}: not supported for trainsfer learning")

        res_df = df
        res_df_names = res_df.columns.values
        best_index = np.argmax(res_df["objective"].values)
        best_param = res_df.iloc[best_index]

        fac_numeric = 8.0
        fac_categorical = 10.0

        cst_new = CS.ConfigurationSpace(seed=1234)
        hp_names = cst.get_hyperparameter_names()
        for hp_name in hp_names:
            hp = cst.get_hyperparameter(hp_name)
            if hp_name in res_df_names:
                if (
                    type(hp) is csh.UniformIntegerHyperparameter
                    or type(hp) is csh.UniformFloatHyperparameter
                ):
                    mu = best_param[hp.name]
                    lower = hp.lower
                    upper = hp.upper
                    sigma = max(1.0, (upper - lower) / fac_numeric)
                    if type(hp) is csh.UniformIntegerHyperparameter:
                        param_new = csh.NormalIntegerHyperparameter(
                            name=hp.name,
                            default_value=mu,
                            mu=mu,
                            sigma=sigma,
                            lower=lower,
                            upper=upper,
                        )
                    else:  # type is csh.UniformFloatHyperparameter:
                        param_new = csh.NormalFloatHyperparameter(
                            name=hp.name,
                            default_value=mu,
                            mu=mu,
                            sigma=sigma,
                            lower=lower,
                            upper=upper,
                        )
                    cst_new.add_hyperparameter(param_new)
                elif type(hp) is csh.CategoricalHyperparameter:
                    choices = hp.choices
                    weights = len(hp.choices) * [1.0]
                    index = choices.index(best_param[hp.name])
                    weights[index] = fac_categorical
                    norm_weights = [float(i) / sum(weights) for i in weights]
                    param_new = csh.CategoricalHyperparameter(
                        name=hp.name, choices=choices, weights=norm_weights
                    )
                    cst_new.add_hyperparameter(param_new)
                else:
                    logging.warning("Not fitting {hp} because it is not supported!")
                    cst_new.add_hyperparameter(hp)
            else:
                logging.warning(
                    "Not fitting {hp} because it was not found in the dataframe!"
                )
                cst_new.add_hyperparameter(hp)

        # For conditions
        for cond in cst.get_conditions():
            if type(cond) == CS.AndConjunction or type(cond) == CS.OrConjunction:
                cond_list = []
                for comp in cond.components:
                    cond_list.append(self.return_cond(comp, cst_new))
                if type(cond) is CS.AndConjunction:
                    cond_new = CS.AndConjunction(*cond_list)
                elif type(cond) is CS.OrConjunction:
                    cond_new = CS.OrConjunction(*cond_list)
                else:
                    logging.warning(f"Condition {type(cond)} is not implemented!")
            else:
                cond_new = self.return_cond(cond, cst_new)
            cst_new.add_condition(cond_new)

        # For forbiddens
        for cond in cst.get_forbiddens():
            if type(cond) is CS.ForbiddenAndConjunction:
                cond_list = []
                for comp in cond.components:
                    cond_list.append(self.return_forbid(comp, cst_new))
                cond_new = CS.ForbiddenAndConjunction(*cond_list)
            elif (
                type(cond) is CS.ForbiddenEqualsClause
                or type(cond) is CS.ForbiddenInClause
            ):
                cond_new = self.return_forbid(cond, cst_new)
            else:
                logging.warning(f"Forbidden {type(cond)} is not implemented!")
            cst_new.add_forbidden_clause(cond_new)

        self._opt_kwargs["dimensions"] = cst_new

    def get_random_batch(self, size: int) -> list:

        if self._fitted:  # for the surrogate or search space
            batch = []
        else:
            batch = self._problem.starting_point_asdict
            # Replace None by "nan"
            for point in batch:
                for (k, v), hp in zip(
                    point.items(), self._problem.space.get_hyperparameters()
                ):
                    if v is None:
                        if (
                            type(hp) is csh.UniformIntegerHyperparameter
                            or type(hp) is csh.UniformFloatHyperparameter
                        ):
                            point[k] = np.nan
                        elif (
                            type(hp) is csh.CategoricalHyperparameter
                            or type(hp) is csh.OrdinalHyperparameter
                        ):
                            point[k] = "NA"

        # Add more starting points
        n_points = max(0, size - len(batch))
        if n_points > 0:
            points = self._opt.ask(n_points=n_points)
            for point in points:
                point_as_dict = self.to_dict(point)
                batch.append(point_as_dict)
        return batch

    def to_dict(self, x: list) -> dict:
        res = {}
        hps_names = self._problem.space.get_hyperparameter_names()
        for i in range(len(x)):
            res[hps_names[i]] = "nan" if isnan(x[i]) else x[i]
        return res


def isnan(x) -> bool:
    """Check if a value is NaN."""
    if isinstance(x, float):
        return math.isnan(x)
    elif isinstance(x, np.float64):
        return np.isnan(x)
    else:
        return False


def replace_nan(x):
    return [np.nan if x_i == "nan" else x_i for x_i in x]
