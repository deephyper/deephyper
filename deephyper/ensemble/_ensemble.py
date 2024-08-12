import functools
from typing import Dict, Sequence

import numpy as np

from deephyper.ensemble.aggregator import Aggregator
from deephyper.evaluator import Evaluator, RunningJob
from deephyper.evaluator.storage import NullStorage
from deephyper.predictor import Predictor


def predict_with_predictor(predictor: Predictor, X: np.ndarray):
    return predictor.predict(X)


def _wrapper_predict_with_predictor(job: RunningJob):
    return predict_with_predictor(**job.parameters)


class EnsemblePredictor(Predictor):

    def __init__(
        self,
        predictors: Sequence[Predictor],
        aggregator: Aggregator,
        evaluator: str | Dict = None,
    ):
        self.predictors = predictors
        self.aggregator = aggregator
        self._evaluator = None

        if evaluator is None:
            self.evaluator_method = "serial"
            self.evaluator_method_kwargs = {}
        elif isinstance(evaluator, str):
            self.evaluator_method = evaluator
            self.evaluator_method_kwargs = {}
        elif isinstance(evaluator, dict):
            self.evaluator_method = evaluator.get("method", "serial")
            self.evaluator_method_kwargs = evaluator.get("method_kwargs", {})
        else:
            raise ValueError(
                f"evaluator must be either None or str or dict, got {type(evaluator)}"
            )
        self.init_evaluator()

    def init_evaluator(self):
        """Initialize an evaluator for the ensemble.

        Returns:
            Evaluator: An evaluator instance.
        """
        method_kwargs = {
            "storage": NullStorage(),
            "run_function_kwargs": {},
        }
        method_kwargs.update(self.evaluator_method_kwargs)
        self._evaluator = Evaluator.create(
            run_function=_wrapper_predict_with_predictor,
            method=self.evaluator_method,
            method_kwargs=method_kwargs,
        )

    def predict(self, X: np.ndarray):
        y_predictors = self.predictions_from_predictors(X, self.predictors)

        y = self.aggregator.aggregate(y_predictors)
        return y

    def predictions_from_predictors(
        self, X: np.ndarray, predictors: Sequence[Predictor]
    ):
        self._evaluator.submit(
            [
                {
                    "predictor": predictor,
                    "X": X,
                }
                for predictor in predictors
            ]
        )

        jobs_done = self._evaluator.gather("ALL")
        jobs_done = list(sorted(jobs_done, key=lambda j: int(j.id.split(".")[-1])))

        y_pred = [job.result for job in jobs_done]
        return y_pred
