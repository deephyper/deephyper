import copy

import numpy as np

from typing import List, Callable

from deephyper.ensemble import EnsemblePredictor
from deephyper.ensemble.selector._selector import Selector
from deephyper.evaluator.callback import Callback


class OnlineSelector(Callback):
    """This class performs ensemble selection after each hyperparameter optimization job completion.

    The ``run``-function passed to the ``Evaluator`` should return in its
    output the ``"online_selector"`` key. This key has for value a
    dictionnary that includes both the ``"y_pred"`` key (i.e., predictions of
    the predictor on which the selection algorithm is applied) and the
    ``"y_pred_idx"`` key (i.e., indexes of the considered sampled in ``y``
    used to score the selection):

    .. code-block:: python

        def run(job):
            ...
            return {
                "objective": objective,
                "online_selector": {"y_pred": y_pred, "y_pred_idx": idx},
            }

    the ``y_pred`` and ``y_pred_idx`` have same first dimension.

    Then, we can create an instance of ``OnlineSelector``:

    .. code-block:: python

        from deephyper.ensemble.aggregator import MeanAggregator
        from deephyper.ensemble.loss import SquaredError
        from deephyper.ensemble.selector import GreedySelector

        online_selector = OnlineSelector(
            y=valid_y,
            selector=GreedySelector(
                loss_func=SquaredError(),
                aggregator=MeanAggregator(),
                k=20,
            ),
        )

    Winally pass this callback to the ``Evaluator`` used for hyperparameter optimization:

    .. code-block:: python

        evaluator = Evaluator.create(
            run,
            method_kwargs={
                "callbacks": [
                    online_selector,
                ],
            },
        )


    Args:
        y (np.ndarray): the data to use for the selector.
        selector (Selector): the selection strategy to use.
    """

    def __init__(
        self,
        y: np.ndarray,
        selector: Selector,
        ensemble: EnsemblePredictor,
        load_predictor_func: Callable,
    ):
        #: the data to use for the ``selector``.
        self.y: np.ndarray = y
        #: the ensemble selection algorithm.
        self.selector: Selector = selector
        #: the list of received job.id from completed hyperparameters optimization jobs.
        self.y_predictors_job_ids: List[str] = []
        #: the list of received predictions mapped to the same shape as ``y``.
        self.y_predictors: List[np.ma.MaskedArray] = []
        #: the list of indexes of the first dimension of ``y_predictors`` from the ``selector``.
        self.selected_predictors_indexes: List[int] = []
        #: the weights of selected predictors.
        self.selected_predictors_weights: List[float] = []

        self._ensemble = ensemble
        self._load_predictor_func = load_predictor_func

    def on_done(self, job):
        if type(job.output["objective"]) is str:
            return

        self.y_predictors_job_ids.append(job.id)

        # All mask entries set to 1 represent invalid values in a MaskedArray
        # The most import part of this class is to build the MaskedArray
        y_pred_mask = np.ones_like(self.y, dtype=int)
        y_pred_mask[job.output["online_selector"]["y_pred_idx"]] = 0

        y_pred = np.zeros_like(self.y, dtype=float)
        y_pred[job.output["online_selector"]["y_pred_idx"]] = job.output["online_selector"][
            "y_pred"
        ]

        m_y_pred = np.ma.masked_array(y_pred, mask=y_pred_mask)
        self.y_predictors.append(m_y_pred)

        # Ensemble
        (
            self.selected_predictors_indexes,
            self.selected_predictors_weights,
        ) = self.selector.select(self.y, self.y_predictors)

    def on_done_other(self, job):
        return self.on_done(job)

    @property
    def selected_predictors_job_ids(self) -> List[str]:
        """List of ``job.id`` corresponding to the selected set of predictors."""
        return [self.y_predictors_job_ids[idx] for idx in self.selected_predictors_indexes]

    @property
    def ensemble(self):
        """The ensemble with its weights.

        It will provide the ``ensemble`` with adapted ``.predictors`` and ``.weights`` from the
        latest selection.
        """
        # TODO: the following deepcopy can create issues with the evaluator used by the ensemble
        # ensemble = copy.deepcopy(self._ensemble)
        ensemble = copy.copy(self._ensemble)
        ensemble.predictors = [
            self._load_predictor_func(job_id) for job_id in self.selected_predictors_job_ids
        ]
        ensemble.weights = self.selected_predictors_weights
        return ensemble
