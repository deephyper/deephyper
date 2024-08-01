import functools
import os
import traceback

import numpy as np
import torch
from deephyper.ensemble import BaseEnsemble

# TODO: selectMetric are metrics coded with tensorflow
from deephyper.nn.torch.metrics import selectMetric

# TODO: set_memory_... is using tensorflow
# from deephyper.nas.run._util import set_memory_growth_for_visible_gpus
from deephyper.core.exceptions import DeephyperRuntimeError


def nll(y, rv_y):
    """Negative log likelihood loss for Tensorflow probability."""
    return -rv_y.log_prob(y)


cce_obj = torch.nn.CrossEntropyLoss(reduction=False)


def cce(y_true, y_pred):
    """Categorical cross-entropy loss."""
    return cce_obj(torch.broadcast_to(y_true, y_pred.shape), y_pred)


LOSSES = {"nll": nll, "cce": cce}


def model_predict(model_path, X, batch_size=32, verbose=0, load_model_func=None):
    """Perform an inference of the model located at ``model_path``.

    :meta private:

    Args:
        model_path (str): Path to the ``h5`` file to load to perform the inferencec.
        X (array): array of input data for which we perform the inference.
        batch_size (int, optional): Batch size used to perform the inferencec. Defaults to 32.
        verbose (int, optional): Verbose option. Defaults to 0.
        load_model_func (callable, optional): Function to load the model. It takes as input the path to the model file and return the loaded model. Defaults to ``None`` for default model loading strategy.

    Returns:
        array: The prediction based on the provided input data.
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    # GPU Configuration if available
    # TODO: check if the following tensorflow function should be replaced
    # set_memory_growth_for_visible_gpus(True)
    # tfk.backend.clear_session()
    model_file = os.path.basename(model_path)

    try:
        if verbose:
            print(f"Loading model {model_file}", end="\n", flush=True)
        if load_model_func is None:
            model = torch.load(model_path, weights_only=False)
        else:
            model = load_model_func(model_path)
        model.eval()
    except Exception:
        if verbose:
            print(f"Could not load model {model_file}", flush=True)
            traceback.print_exc()
        model = None

    if model is None:
        return None

    # Create the dataset
    if type(X) is list:
        # Multiple input arrays
        dataset = TensorDataset(*(torch.from_numpy(Xi).float() for Xi in X))
    else:
        # Single input array
        dataset = TensorDataset(torch.from_numpy(X).float())
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def batch_predict(dataset, convert_func=lambda x: x):
        y_list = []
        for (batch,) in dataset:
            y = model(batch)
            y_list.append(convert_func(y))
        y = np.concatenate(y_list, axis=0)
        return y

    # Perform an inference to test the type of the output (Distribution or Tensor)
    y_dist = model(next(iter(dataset))[0])
    if isinstance(y_dist, torch.distributions.Distribution):
        if hasattr(y_dist, "loc") and hasattr(y_dist, "scale"):

            def convert_func(y_dist):
                return np.concatenate(
                    [
                        y_dist.loc.detach().numpy(),
                        y_dist.scale.detach().numpy(),
                    ],
                    axis=-1,
                )

            y = batch_predict(dataset, convert_func)
        else:
            raise DeephyperRuntimeError(
                "Distribution doesn't have 'loc' or 'scale' attributes!"
            )
    else:
        y = batch_predict(X, lambda x: x.detach().numpy())

    return y


class _TorchUQEnsemble(BaseEnsemble):
    """Ensemble with uncertainty quantification based on uniform averaging of the predictions of each members.

    :meta private:

    Args:
        model_dir (str): Path to directory containing saved Keras models in .h5 format.
        loss (callable): a callable taking (y_true, y_pred) as input.
        size (int, optional): Number of unique models used in the ensemble. Defaults to 5.
        verbose (bool, optional): Verbose mode. Defaults to True.
        ray_address (str, optional): Address of the Ray cluster. If "auto" it will try to connect to an existing cluster. If "" it will start a local Ray cluster. Defaults to "".
        num_cpus (int, optional): Number of CPUs allocated to load one model and predict. Defaults to 1.
        num_gpus (int, optional): Number of GPUs allocated to load one model and predict. Defaults to None.
        batch_size (int, optional): Batch size used batchify the inference of loaded models. Defaults to 32.
        selection (str, optional): Selection strategy to build the ensemble. Value in ``["topk", "caruana"]``. Default to ``topk``.
        mode (str, optional): Value in ``["regression", "classification"]``. Default to ``"regression"``.
        load_model_func (callable, optional): Function to load checkpointed models. It takes as input the path to the model file and return the loaded model. Defaults to ``None`` for default model loading strategy.
    """

    def __init__(
        self,
        model_dir,
        loss,
        size=5,
        verbose=True,
        batch_size=32,
        selection="topk",
        mode="regression",
        load_model_func=None,
        evaluator_method="serial",
        evaluator_method_kwargs=None,
    ):
        if type(loss) is str and loss in LOSSES:
            loss = LOSSES[loss]
        elif callable(loss):
            pass
        else:
            raise ValueError(
                f"loss={loss} is not a valid loss function. It should be a callable or a value in {list(LOSSES.keys())}."
            )
        super().__init__(
            model_dir,
            loss,
            size,
            verbose,
            batch_size,
            evaluator_method=evaluator_method,
            evaluator_method_kwargs=evaluator_method_kwargs,
        )
        assert selection in ["topk", "caruana"]
        self.selection = selection
        assert mode in ["regression", "classification"]
        self.mode = mode
        self._model_predict_func = functools.partial(
            model_predict, load_model_func=load_model_func
        )
        self.init_evaluator()

    def _list_files_in_model_dir(self):
        return [
            f
            for f in os.listdir(self.model_dir)
            if f.endswith("pth") or f.endswith("pt")
        ]

    def _select_members(self, loss_func, y_true, y_pred, k=2, verbose=0):
        if self.selection == "topk":
            func = topk
        elif self.selection == "caruana":
            func = greedy_caruana
        else:
            raise NotImplementedError
        return func(loss_func, y_true, y_pred, k, verbose)

    def fit(self, X, y):

        model_files = self._list_files_in_model_dir()
        if self.verbose:
            print(f"Found {len(model_files)} possible models to build the ensemble.")

        if len(model_files) == 0:
            raise ValueError(f"No '*.torch' files found in {self.model_dir}")

        def model_path(f):
            return os.path.join(self.model_dir, f)

        y_pred = self.get_predictions_from_models(
            X, (model_path(f) for f in model_files)
        )

        self._members_indexes = self._select_members(
            self.loss, y_true=y, y_pred=y_pred, k=self.size
        )
        self.members_files = [model_files[i] for i in self._members_indexes]

    def predict(self, X) -> np.ndarray:
        def model_path(f):
            return os.path.join(self.model_dir, f)

        y_pred = self.get_predictions_from_models(
            X, (model_path(f) for f in self.members_files)
        )

        y = aggregate_predictions(y_pred, regression=(self.mode == "regression"))

        return y

    def evaluate(self, X, y, metrics=None, scaler_y=None):
        scores = {}

        y_pred = self.predict(X)

        if scaler_y:
            y_pred = scaler_y(y_pred)
            y = scaler_y(y)

        scores["loss"] = torch.mean(self.loss(y, y_pred)).detach().numpy()
        if metrics:
            if type(metrics) is list:
                for metric in metrics:
                    if callable(metric):
                        metric_name = metric.__name__
                    else:
                        metric_name = metric
                    scores[metric_name] = apply_metric(metric, y, y_pred)
            elif type(metrics) is dict:
                for metric_name, metric in metrics.items():
                    scores[metric_name] = apply_metric(metric, y, y_pred)
            else:
                raise ValueError("Metrics should be of type list or dict.")

        return scores


class _TorchUQEnsembleRegressor(_TorchUQEnsemble):
    """Ensemble with uncertainty quantification for regression based on uniform averaging of the predictions of each members.

    Args:
        model_dir (str): Path to directory containing saved Keras models in .h5 format.
        loss (callable): a callable taking (y_true, y_pred) as input.
        size (int, optional): Number of unique models used in the ensemble. Defaults to 5.
        verbose (bool, optional): Verbose mode. Defaults to True.
        ray_address (str, optional): Address of the Ray cluster. If "auto" it will try to connect to an existing cluster. If "" it will start a local Ray cluster. Defaults to "".
        num_cpus (int, optional): Number of CPUs allocated to load one model and predict. Defaults to 1.
        num_gpus (int, optional): Number of GPUs allocated to load one model and predict. Defaults to None.
        batch_size (int, optional): Batch size used batchify the inference of loaded models. Defaults to 32.
        selection (str, optional): Selection strategy to build the ensemble. Value in ``[["topk", "caruana"]``. Default to ``topk``.
        load_model_func (callable, optional): Function to load checkpointed models. It takes as input the path to the model file and return the loaded model. Defaults to ``None`` for default model loading strategy.
    """

    def __init__(
        self,
        model_dir,
        loss=nll,
        size=5,
        verbose=True,
        batch_size=32,
        selection="topk",
        load_model_func=None,
        evaluator_method="serial",
        evaluator_method_kwargs=None,
    ):
        super().__init__(
            model_dir,
            loss,
            size,
            verbose,
            batch_size,
            selection,
            mode="regression",
            load_model_func=load_model_func,
            evaluator_method=evaluator_method,
            evaluator_method_kwargs=evaluator_method_kwargs,
        )

    def predict_var_decomposition(self, X):
        r"""Execute an inference of the ensemble for the provided data with uncertainty quantification estimates. The **aleatoric uncertainty** corresponds to the expected value of learned variance of each model composing the ensemble :math:`\mathbf{E}[\sigma_\\theta^2(\mathbf{x})]`. The **epistemic uncertainty** corresponds to the variance of learned mean estimates of each model composing the ensemble :math:`\mathbf{V}[\mu_\\theta(\mathbf{x})]`.

        Args:
            X (array): An array of input data.

        Returns:
            y, u1, u2: where ``y`` is the mixture distribution, ``u1`` is the aleatoric component of the variance of ``y`` and ``u2`` is the epistemic component of the variance of ``y``.
        """
        # make predictions

        def model_path(f):
            return os.path.join(self.model_dir, f)

        y_pred = y_pred = self.get_predictions_from_models(
            X, (model_path(f) for f in self.members_files)
        )

        y = aggregate_predictions(y_pred, regression=(self.mode == "regression"))

        # variance decomposition
        mid = np.shape(y_pred)[-1] // 2
        selection = [slice(0, s) for s in np.shape(y_pred)]
        selection_loc = selection[:]
        selection_std = selection[:]
        selection_loc[-1] = slice(0, mid)
        selection_std[-1] = slice(mid, np.shape(y_pred)[-1])
        loc = y_pred[tuple(selection_loc)]
        scale = y_pred[tuple(selection_std)]

        aleatoric_unc = np.mean(np.square(scale), axis=0)
        epistemic_unc = np.square(np.std(loc, axis=0))

        # dist, aleatoric uq, epistemic uq
        return y, aleatoric_unc, epistemic_unc


class _TorchUQEnsembleClassifier(_TorchUQEnsemble):
    """Ensemble with uncertainty quantification for classification based on uniform averaging of the predictions of each members.

    Args:
        model_dir (str): Path to directory containing saved Keras models in .h5 format.
        loss (callable): a callable taking (y_true, y_pred) as input.
        size (int, optional): Number of unique models used in the ensemble. Defaults to 5.
        verbose (bool, optional): Verbose mode. Defaults to True.
        batch_size (int, optional): Batch size used batchify the inference of loaded models. Defaults to 32.
        selection (str, optional): Selection strategy to build the ensemble. Value in ``[["topk", "caruana"]``. Default to ``topk``.
        load_model_func (callable, optional): Function to load checkpointed models. It takes as input the path to the model file and return the loaded model. Defaults to ``None`` for default model loading strategy.
    """

    def __init__(
        self,
        model_dir,
        loss=cce,
        size=5,
        verbose=True,
        batch_size=32,
        selection="topk",
        load_model_func=None,
        evaluator_method="serial",
        evaluator_method_kwargs=None,
    ):
        super().__init__(
            model_dir,
            loss,
            size,
            verbose,
            batch_size,
            selection,
            mode="classification",
            load_model_func=load_model_func,
            evaluator_method=evaluator_method,
            evaluator_method_kwargs=evaluator_method_kwargs,
        )


def apply_metric(metric_name, y_true, y_pred) -> float:
    """Perform the computation of provided metric.

    :meta private:

    Args:
        metric_name (str|callable): If ``str`` then it needs to be a metric available in ``deephyper.nas.metrics``.
        y_true (array): Array of true predictions.
        y_pred (array): Array of predicted predictions

    Returns:
        float: a scalar value of the computed metric.
    """
    metric_func = selectMetric(metric_name)

    if type(y_true) is np.ndarray:
        y_true = torch.from_numpy(y_true).float()
    if type(y_pred) is np.ndarray:
        y_pred = torch.from_numpy(y_pred).float()

    metric = metric_func(y_true, y_pred)
    if metric.size(dim=0) >= 1:
        metric = torch.mean(metric)
    return metric.detach().numpy()


def aggregate_predictions(y_pred, regression=True):
    """Build an ensemble from predictions.

    :meta private:

    Args:
        ensemble_members (np.array): Indexes of selected members in the axis-0 of y_pred.
        y_pred (np.array): Predictions array of shape (n_models, n_samples, n_outputs).
        regression (bool): Boolean (True) if it is a regression (False) if it is a classification.
    Return:
        A TFP Normal Distribution in the case of regression and a np.array with average probabilities
        in the case of classification.
    """
    if regression:
        # assuming first half are means, second half are std
        mid = np.shape(y_pred)[-1] // 2
        selection = [slice(0, s) for s in np.shape(y_pred)]
        selection_loc = selection[:]
        selection_std = selection[:]
        selection_loc[-1] = slice(0, mid)
        selection_std[-1] = slice(mid, np.shape(y_pred)[-1])
        loc = y_pred[tuple(selection_loc)]
        scale = y_pred[tuple(selection_std)]

        mean_loc = np.mean(loc, axis=0)
        sum_loc_scale = np.square(loc) + np.square(scale)
        mean_scale = np.sqrt(np.mean(sum_loc_scale, axis=0) - np.square(mean_loc))

        return torch.distributions.Normal(
            loc=torch.from_numpy(mean_loc), scale=torch.from_numpy(mean_scale)
        )
    else:  # classification
        agg_y_pred = np.mean(y_pred[:, :, :], axis=0)
        return agg_y_pred


def topk(loss_func, y_true, y_pred, k=2, verbose=0):
    """Select the top-k models to be part of the ensemble. A model can appear only once in the ensemble for this strategy.

    :meta private:
    """
    if np.shape(y_true)[-1] * 2 == np.shape(y_pred)[-1]:  # regression
        mid = np.shape(y_true)[-1]
        y_pred = torch.distributions.Normal(
            loc=torch.from_numpy(y_pred[:, :, :mid]).float(),
            scale=torch.from_numpy(y_pred[:, :, mid:]).float(),
        )
    else:
        y_pred = torch.from_numpy(y_pred).float()
    y_true = torch.from_numpy(y_true).float()
    # losses is of shape: (n_models, n_outputs)
    losses = torch.mean(loss_func(y_true, y_pred), axis=1).detach().numpy()
    if verbose:
        print(f"Top-{k} losses: {losses.reshape(-1)[:k]}")
    ensemble_members = np.argsort(losses, axis=0)[:k].reshape(-1).tolist()
    return ensemble_members


def greedy_caruana(loss_func, y_true, y_pred, k=2, verbose=0):
    """Select the top-k models to be part of the ensemble. A model can appear only once in the ensemble for this strategy.

    :meta private:
    """
    # print(f"{y_true=}")
    # print(f"{y_pred=}")
    regression = np.shape(y_true)[-1] * 2 == np.shape(y_pred)[-1]
    n_models = np.shape(y_pred)[0]
    if regression:  # regression
        mid = np.shape(y_true)[-1]
        selection = [slice(0, s) for s in np.shape(y_pred)]
        selection_loc = selection[:]
        selection_std = selection[:]
        selection_loc[-1] = slice(0, mid)
        selection_std[-1] = slice(mid, np.shape(y_pred)[-1])
        y_pred_ = torch.distributions.Normal(
            loc=torch.from_numpy(y_pred[tuple(selection_loc)]),
            scale=torch.from_numpy(y_pred[tuple(selection_std)]),
        )
    else:
        y_pred_ = torch.from_numpy(y_pred).float()
    y_true = torch.from_numpy(y_true).float()

    losses = (
        torch.mean(torch.reshape(loss_func(y_true, y_pred_), [n_models, -1]), axis=1)
        .detach()
        .numpy()
    )
    assert n_models == np.shape(losses)[0]

    i_min = np.nanargmin(losses)
    loss_min = losses[i_min]
    ensemble_members = [i_min]
    if verbose:
        print(f"Loss: {loss_min:.3f} - Ensemble: {ensemble_members}")

    def loss(y_true, y_pred):
        return torch.mean(loss_func(y_true, y_pred)).detach().numpy()

    while len(np.unique(ensemble_members)) < k:
        losses = [
            loss(
                y_true,
                aggregate_predictions(
                    y_pred[ensemble_members + [i]], regression=regression
                ),
            )
            for i in range(n_models)  # iterate over all models
        ]
        i_min_ = np.nanargmin(losses)
        loss_min_ = losses[i_min_]

        if loss_min_ < loss_min:
            if (
                len(np.unique(ensemble_members)) == 1 and ensemble_members[0] == i_min_
            ):  # numerical errors...
                return ensemble_members
            loss_min = loss_min_
            ensemble_members.append(i_min_)
            if verbose:
                print(f"Loss: {loss_min:.3f} - Ensemble: {ensemble_members}")
        else:
            return ensemble_members

    return ensemble_members
