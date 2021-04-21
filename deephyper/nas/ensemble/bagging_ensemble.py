import argparse
import os
from re import I, VERBOSE
from numpy.core.defchararray import add

import tensorflow as tf
import numpy as np

from deephyper.search import util as search_util
from deephyper.nas.run import util as run_util
from deephyper.nas.metrics import selectMetric
from deephyper.core.parser import add_arguments_from_signature
from deephyper.nas.ensemble import BaseEnsemble


class BaggingEnsemble(BaseEnsemble):
    def __init__(
        self,
        problem,
        model_dir,
        members_files=None,
        size=5,
        history_dir=None,
        output_members_files=None,
        verbose=True,
        reduce="majority",
    ):
        super().__init__(
            problem,
            model_dir,
            members_files,
            size,
            history_dir,
            output_members_files,
            verbose,
        )
        self.reduce = reduce

    @staticmethod
    def _extend_parser(parser) -> argparse.ArgumentParser:
        add_arguments_from_signature(parser, BaggingEnsemble)
        return parser

    def main(self):
        self.build_ensemble()

    def build_ensemble(self):
        _, _, data = run_util.setup_data(self.problem.space, add_to_config=False)
        _, (valid_X, valid_y) = data

        if len(self.members_models) == 0:
            self.find_members(valid_X, valid_y)

            if len(self.members_models) < self.size and self.verbose:
                print(
                    f"Found only {len(self.members_models)} members to improve the ensemble"
                )

        scores = self.evaluate(valid_X, valid_y)
        if self.verbose:
            print(scores)

    def find_members(self, X, y) -> None:
        assert len(self.members_models) == 0

        model_files = [f for f in os.listdir(self.model_dir) if f[-2:] == "h5"]

        best_objective = 0

        if self.verbose:
            print("Starting Greedy selection of members of the ensemble.")

        # gready selection of members from the ensemble
        for model_file, model in zip(model_files, self.load_models(model_files)):

            self.members_models.append(model)
            scores = self.evaluate(X, y)

            if scores["r2"] > best_objective:
                best_objective = scores["r2"]
                self.members_files.append(model_file)

                if self.verbose:
                    print(
                        f"New objective: {best_objective} with ensemble size {len(self.members_models)}"
                    )
            else:
                self.members_models.pop()

            if len(self.members_models) == self.size:
                break

        self.save_members_files()

    def predict(self, X) -> np.ndarray:
        # make predictions
        yhats = [model.predict(X) for model in self.members_models]
        yhats = np.array(yhats)

        # sum across ensemble members
        summed = np.sum(yhats, axis=0)
        if self.reduce == "majority":
            # argmax across classes
            y = np.argmax(summed, axis=1)
        elif self.reduce == "mean":
            y = summed / len(self.members_models)

        return y

    def evaluate(self, X, y):
        scores = {}

        y_pred = self.predict(X)

        for metric_name in self.problem.space["metrics"]:
            scores[metric_name] = apply_metric(metric_name, y, y_pred)

        return scores


def apply_metric(metric_name, y_true, y_pred) -> float:
    metric_func = selectMetric(metric_name)
    metric = metric_func(
        tf.convert_to_tensor(y_true, dtype=np.float64),
        tf.convert_to_tensor(y_pred, dtype=np.float64),
    ).numpy()
    return metric


if __name__ == "__main__":
    args = BaggingEnsemble.parse_args()
    ensemble = BaggingEnsemble(**vars(args))
    ensemble.main()