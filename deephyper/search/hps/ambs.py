"""Asynchronous Model-Based Search.

Arguments of AMBS:

* ``surrogate-model``

    * ``RF`` : Random Forest (default)
    * ``ET`` : Extra Trees
    * ``GBRT`` : Gradient Boosting Regression Trees
    * ``DUMMY`` :
    * ``GP`` : Gaussian process

* ``liar-strategy``

    * ``cl_max`` : (default)
    * ``cl_min`` :
    * ``cl_mean`` :

* ``acq-func`` : Acquisition function

    * ``LCB`` :
    * ``EI`` :
    * ``PI`` :
    * ``gp_hedge`` : (default)
"""


import math
import signal


import ConfigSpace as CS
import ConfigSpace.hyperparameters as csh
import numpy as np
import pandas as pd
import skopt
from deephyper.core.logs.logging import JsonMessage as jm
from deephyper.search import Search, util

dhlogger = util.conf_logger("deephyper.search.hps.ambs")

SERVICE_PERIOD = 2  # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 1  # How many jobs to complete between optimizer checkpoints
EXIT_FLAG = False


def on_exit(signum, stack):
    global EXIT_FLAG
    EXIT_FLAG = True


class AMBS(Search):
    def __init__(
        self,
        problem,
        run,
        evaluator,
        surrogate_model="RF",
        acq_func="LCB",
        kappa=1.96,
        xi=0.001,
        liar_strategy="cl_min",
        n_jobs=1,  # 32 is good for Theta
        checkpoint="",
        transfer_learning="",
        **kwargs,
    ):
        kwargs["cache_key"] = "to_dict"
        super().__init__(problem, run, evaluator, **kwargs)
        dhlogger.info("Initializing AMBS")
        dhlogger.info(f"kappa={kappa}, xi={xi}")

        self.checkpoint = pd.read_csv(checkpoint) if checkpoint != "" else None
        self.transfer_learning = pd.read_csv(transfer_learning) if transfer_learning != "" else None

        self.n_initial_points = self.evaluator.num_workers
        self.liar_strategy = liar_strategy

        self.opt = skopt.Optimizer(
            dimensions=self.problem.space,
            base_estimator=self.get_surrogate_model(surrogate_model, n_jobs),
            acq_func=acq_func,
            acq_optimizer="sampling",
            acq_func_kwargs={"xi": xi, "kappa": kappa},
            n_initial_points=self.n_initial_points if self.checkpoint is None else 0,
            random_state=self.problem.seed,
        )


    @staticmethod
    def _extend_parser(parser):
        parser.add_argument(
            "--surrogate-model",
            default="RF",
            choices=["RF", "ET", "GBRT", "DUMMY", "GP"],
            help="Type of surrogate model (learner).",
        )
        parser.add_argument(
            "--liar-strategy",
            default="cl_min",
            choices=["cl_min", "cl_mean", "cl_max"],
            help="Constant liar strategy",
        )
        parser.add_argument(
            "--acq-func",
            default="gp_hedge",
            choices=["LCB", "EI", "PI", "gp_hedge"],
            help="Acquisition function type",
        )
        parser.add_argument(
            "--kappa",
            type=float,
            default=1.96,
            help='Controls how much of the variance in the predicted values should be taken into account. If set to be very high, then we are favouring exploration over exploitation and vice versa. Used when the acquisition is "LCB".',
        )

        parser.add_argument(
            "--xi",
            type=float,
            default=0.01,
            help='Controls how much improvement one wants over the previous best values. If set to be very high, then we are favouring exploration over exploitation and vice versa. Used when the acquisition is "EI", "PI".',
        )

        parser.add_argument(
            "--n-jobs",
            type=int,
            default=1,
            help="number of cores to use for the 'surrogate model' (learner), if n_jobs=-1 then it will use all cores available.",
        )
        parser.add_argument(
            "--checkpoint",
            type=str,
            default="",
            help="Path to a CSV file which will be used to restart the search.",
        )
        parser.add_argument(
            "--transfer-learning",
            type=str,
            default="",
            help="Path to a CSV file which will be used for transfer learning.",
        )
        return parser

    def main(self):
        # timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
        # chkpoint_counter = 0

        num_evals_done = 0

        # Filling available nodes at start
        dhlogger.info(f"Generating {self.evaluator.num_workers} initial points...")
        self.evaluator.add_eval_batch(self.get_random_batch(size=self.n_initial_points))

        # Main loop
        while num_evals_done < self.max_evals:

            # Collecting finished evaluations
            new_results = list(self.evaluator.get_finished_evals())

            if len(new_results) > 0:
                stats = {"num_cache_used": self.evaluator.stats["num_cache_used"]}
                dhlogger.info(jm(type="env_stats", **stats))
                self.evaluator.dump_evals()

                num_received = len(new_results)
                num_evals_done += num_received

                # Transform configurations to list to fit optimizer
                opt_X = []
                opt_y = []
                for cfg, obj in new_results:
                    x = replace_nan(cfg.values())
                    opt_X.append(x)
                    opt_y.append(-obj)  #! maximizing
                self.opt.tell(opt_X, opt_y)  #! fit: costly
                new_X = self.opt.ask(
                    n_points=len(new_results), strategy=self.liar_strategy
                )

                new_batch = []
                for x in new_X:
                    new_cfg = self.to_dict(x)
                    new_batch.append(new_cfg)

                # submit_childs
                if len(new_results) > 0:
                    self.evaluator.add_eval_batch(new_batch)

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

    def fit_checkpoint(self):
        hp_names = self.problem.space.get_hyperparameter_names()
        for hp in self.problem.space.get_hyperparameters():
        x = self.checkpoint[hp_names].values.tolist()
        y = self.checkpoint.objective.tolist()

        self.opt.tell(x, [yi * -1.0 for yi in y])


    def return_cond(self, cond, cst_new):
        parent = cst_new.get_hyperparameter(cond.parent.name)
        child = cst_new.get_hyperparameter(cond.child.name)
        if type(cond) == CS.EqualsCondition:
            value = cond.value
            cond_new = CS.EqualsCondition(child,parent,cond.value)
        elif type(cond) == CS.GreaterThanCondition:
            value = cond.value
            cond_new = CS.GreaterThanCondition(child,parent,value)
        elif type(cond) == CS.NotEqualsCondition:
            value = cond.value
            cond_new = CS.GreaterThanCondition(child,parent,value)
        elif type(cond) == CS.LessThanCondition:
            value = cond.value
            cond_new = CS.GreaterThanCondition(child,parent,value)
        elif type(cond) == CS.InCondition:
            values = cond.values
            cond_new = CS.GreaterThanCondition(child,parent,values)
        else:
            print('Not supported type'+str(type(cond)))
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
                print('Not supported type'+str(type(cond)))
        return cond_new

    def fit_transfer_learning(self):
        cst =  self.problem.space
        if type(cst) != CS.ConfigurationSpace:
            print('%s: not supported for trainsfer learning'%type(cst))

        res_df = self.transfer_learning
        res_df_names = res_df.columns.values
        best_index = np.argmax(res_df['objective'].values)
        best_param = res_df.iloc[best_index]

        fac_numeric = 8.0
        fac_categorical = 10.0

        cst_new = CS.ConfigurationSpace(seed=1234)
        hp_names = cst.get_hyperparameter_names()
        for hp_name in hp_names:
            hp = cst.get_hyperparameter(hp_name)
            if hp_name in res_df_names:
                if type(hp) == csh.UniformIntegerHyperparameter or type(hp) == csh.UniformFloatHyperparameter:
                    mu = best_param[hp.name]
                    lower = hp.lower
                    upper = hp.upper
                    sigma = max(1.0, (upper - lower)/fac_numeric)
                    if type(hp) == csh.UniformIntegerHyperparameter:
                        param_new = csh.NormalIntegerHyperparameter(name=hp.name, default_value=mu, mu=mu, sigma=sigma, lower=lower, upper=upper)
                    elif type(hp) == csh.UniformFloatHyperparameter:
                        param_new = csh.NormalFloatHyperparameter(name=hp.name, default_value=mu, mu=mu, sigma=sigma, lower=lower, upper=upper)
                    else:
                        pass
                    cst_new.add_hyperparameter(param_new)
                elif type(hp) == csh.CategoricalHyperparameter:
                    choices = hp.choices
                    weights = len(hp.choices)*[1.0]
                    index = choices.index(best_param[hp.name])
                    weights[index] = fac_categorical
                    norm_weights = [float(i)/sum(weights) for i in weights]
                    param_new = csh.CategoricalHyperparameter(name=hp.name, choices=choices,weights=norm_weights)
                    cst_new.add_hyperparameter(param_new)
                else:
                    cst_new.add_hyperparameter(hp)
            else:
                cst_new.add_hyperparameter(hp)


        for cond in cst.get_conditions():
            if type(cond) == CS.AndConjunction or type(cond) == CS.OrConjunction:
                cond_list = []
                for comp in cond.components:
                    cond_list.append(self.return_cond(comp, cst_new))
                if type(cond) == CS.AndConjunction:
                    cond_new = CS.AndConjunction(*cond_list)
                elif type(cond) == CS.OrConjunction:
                    cond_new = CS.OrConjunction(*cond_list)
                else:
                    print('Not implemented')
            else:
                cond_new = self.return_cond(cond, cst_new)
            cst_new.add_condition(cond_new)

        for cond in cst.get_forbiddens():
            if type(cond) == CS.ForbiddenAndConjunction:
                cond_list = []
                for comp in cond.components:
                    cond_list.append(self.return_forbid(comp, cst_new))
                cond_new = CS.ForbiddenAndConjunction(*cond_list)
            elif type(cond) == CS.ForbiddenEqualsClause or type(cond) == CS.ForbiddenInClause:
                cond_new = self.return_forbid(cond, cst_new)
            else:
                print('Not supported type'+str(type(cond)))
            cst_new.add_forbidden_clause(cond_new)


    def get_random_batch(self, size: int) -> list:

        if self.checkpoint is not None:
            batch = []
            self.fit_checkpoint()
        elif self.transfer_learning is not None:
            batch = []
            self.fit_transfer_learning()
        else:
            batch = self.problem.starting_point_asdict
            # Replace None by "nan"
            for point in batch:
                for (k, v), hp in zip(
                    point.items(), self.problem.space.get_hyperparameters()
                ):
                    if v is None:
                        if (
                            type(hp) == csh.UniformIntegerHyperparameter
                            or type(hp) == csh.UniformFloatHyperparameter
                        ):
                            point[k] = np.nan
                        elif (
                            type(hp) == csh.CategoricalHyperparameter
                            or type(hp) == csh.OrdinalHyperparameter
                        ):
                            point[k] = "NA"

        # Add more starting points
        n_points = max(0, size - len(batch))
        if n_points > 0:
            points = self.opt.ask(n_points=n_points)
            for point in points:
                point_as_dict = self.to_dict(point)
                batch.append(point_as_dict)
        return batch

    def to_dict(self, x: list) -> dict:
        res = {}
        hps_names = self.problem.space.get_hyperparameter_names()
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


if __name__ == "__main__":
    args = AMBS.parse_args()
    search = AMBS(**vars(args))
    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)
    search.main()
