import argparse
import os

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from deephyper.search import util as search_util
from deephyper.nas.run import util as run_util
from deephyper.nas.metrics import selectMetric
from deephyper.core.parser import add_arguments_from_signature
from deephyper.core.exceptions import DeephyperRuntimeError
from deephyper.nas.ensemble import BaseEnsemble


class UQBaggingEnsemble(BaseEnsemble):
    def __init__(
        self,
        model_dir,
        loss,
        batch_size=None,
        size=5,
        verbose=True,
        mode="classification",  # or "regression"
        selection="greedy",
    ):
        super().__init__(
            model_dir,
            loss,
            size,
            verbose,
        )
        self.mode = mode
        self.batch_size = batch_size
        self.model = None
        self.selection = selection

    @staticmethod
    def _extend_parser(parser) -> argparse.ArgumentParser:
        add_arguments_from_signature(parser, UQBaggingEnsemble)
        return parser

    def sort_models_by_min_loss(self, model_files: list, X, y) -> tuple:
        model_losses = []

        #! TODO: parallelize this loop
        for i, model in enumerate(self.load_models(model_files[:])):
            if self.verbose:
                print(f"Loading model {i}    ", end="\r", flush=True)
            if model is None:
                model_files.pop(i)
                continue
            self.compile_model(model)

            loss = model.evaluate(X, y, batch_size=self.batch_size, verbose=0)
            model_losses.append(loss)

        if self.verbose:
            print()

        model_files, model_losses = list(
            zip(*sorted(zip(model_files, model_losses), key=lambda t: t[1]))
        )
        model_files = list(model_files)
        model_losses = list(model_losses)

        return model_files, model_losses

    def compile_model(self, model):
        model.compile(loss=self.loss)

    def fit(self, X, y) -> None:

        if self.selection == "greedy":
            self.greedy_selection(X, y)
        elif self.selection == "topk":
            self.topk_selection(X, y)
        else:
            raise DeephyperRuntimeError(f"Selection '{self.selection}' is not valid!")

    def greedy_selection(self, X, y):
        if self.verbose:
            print("Starting Greedy selection of members of the ensemble.")

        model_files = self.list_all_model_files()
        model_files, model_losses = self.sort_models_by_min_loss(model_files, X, y)

        min_loss = model_losses[0]
        model_files_ens = [model_files[0]]

        if self.verbose:
            print(f"Initial loss is {min_loss:.5f}")

        # gready selection of members from the ensemble
        for i, model_file in enumerate(model_files[1:]):

            model_files_ens.append(model_file)

            self.model = self.build_ensemble_model(
                list(self.load_models(model_files_ens))
            )
            self.compile_model(self.model)
            loss = self.evaluate(X, y)

            if self.verbose:
                print(
                    f"Step {i+1:04} - Best Loss: {min_loss:.5f} | Current Try: {loss:.5f}",
                    end="\r",
                    flush=True,
                )

            if loss < min_loss:
                min_loss = loss

                if self.verbose:
                    print(
                        f"Adding Model {i+1:04} - New loss: {min_loss:.5f} with ensemble size {len(model_files_ens)}           "
                    )
            else:
                model_files_ens.pop()

            if len(model_files_ens) == self.size:
                break

        if self.verbose:
            print()

        self.members_files = model_files_ens
        self.model = self.build_ensemble_model(list(self.load_models(self.members_files)))
        self.compile_model(self.model)

        if len(self.members_files) < self.size and self.verbose:
            print(
                f"Found only {len(self.members_files)} members to improve the ensemble",
                flush=True,
            )

    def topk_selection(self, X, y):
        if self.verbose:
            print("Starting Top-K selection of members of the ensemble.")

        model_files = self.list_all_model_files()
        model_files, model_losses = self.sort_models_by_min_loss(model_files, X, y)

        self.members_files = model_files[: self.size]

        self.model = self.build_ensemble_model(list(self.load_models(self.members_files)))
        self.compile_model(self.model)
        loss = self.evaluate(X, y)

        if self.verbose:
            print(f"Top-K Loss: {loss:.5f}")

    def build_ensemble_model(self, models):

        if len(models) == 1:
            return models[0]

        models_input_shapes = [m.input.shape for m in models]
        assert all(
            [
                tuple(models_input_shapes[0]) == tuple(shape)
                for shape in models_input_shapes
            ]
        ), "all models of the ensemble must have equal input shapes"

        input_ensemble = tf.keras.Input(
            shape=tuple(models_input_shapes[0])[1:], name="input_ensemble"
        )

        locs, scales = [], []
        for model in models:
            normal_dist = model(input_ensemble)

            loc = tf.keras.layers.Lambda(lambda t: t.loc)(normal_dist)
            scale = tf.keras.layers.Lambda(lambda t: t.scale)(normal_dist)

            locs.append(loc)
            scales.append(scale)

        mean_locs = tf.keras.layers.Average()(locs)

        sum_loc_scale = [
            tf.math.square(loc) + tf.math.square(scale)
            for loc, scale in zip(locs, scales)
        ]
        mean_scales = tf.math.sqrt(
            tf.math.reduce_mean(sum_loc_scale, axis=0) - tf.math.square(mean_locs)
        )

        output_ensemble = tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Normal(
                loc=t[0],
                scale=t[1],
            )
        )([mean_locs, mean_scales])

        model_ensemble = tf.keras.Model(
            input_ensemble, output_ensemble, name="ensemble_model"
        )

        return model_ensemble

    def predict(self, X) -> tfp.distributions.Normal:
        return self.model(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, batch_size=self.batch_size, verbose=0)

    def load(self, file):
        self.load_members_files(file)
        self.model = self.build_ensemble_model(list(self.load_models(self.members_files)))
        self.compile_model(self.model)

    def save(self, file):
        self.save_members_files(file)


if __name__ == "__main__":
    args = UQBaggingEnsemble.parse_args()
    ensemble = UQBaggingEnsemble(**vars(args))
    ensemble.main()