from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from deephyper.ensemble.aggregator._aggregator import Aggregator


class MixedNormalAggregator(Aggregator):
    """Aggregate a collection of predictions, each representing a normal distribution.

    This aggregator combines the mean (`loc`) and standard deviation (`scale`)
    of multiple normal distributions into a single mixture distribution.

    Eventhough the mixture of normal distributions is not a normal distribution, this aggregator
    approximates it as a normal and only returns the mean and standard deviation of the mixture.

    .. list-table::
        :widths: 25 25
        :header-rows: 1

        * - Array (Fixed Set)
          - MaskedArray
        * - ✅
          - ❌

    Args:
        decomposed_scale (bool, optional): If ``True``, the scale of the mixture distribution
            is decomposed into aleatoric and epistemic components. Default is ``False``.
    """

    def __init__(self, decomposed_scale: bool = False):
        self.decomposed_scale = decomposed_scale

    def aggregate(
        self,
        y: List[Dict[str, np.ndarray]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Aggregate the predictions.

        Args:
            y (List[Dict[str, np.ndarray]]): Predictions with keys:
                - ``loc``: Mean of each normal distribution, shape ``(n_predictors, n_samples, ...,
                n_outputs)``.
                - ``scale``: Standard deviation of each normal distribution.

            weights (Optional[List[float]]): Predictor weights. Defaults to uniform weights.

        Returns:
            Dict[str, np.ndarray]: Aggregated predictions with:
                - `loc`: Mean of the mixture distribution.
                - `scale`: Standard deviation (or decomposed components if `decomposed_scale` is
                    `True`).
        """
        if not y:
            raise ValueError("Input list 'y' must not be empty.")

        # Validate all dictionaries in y have the same keys
        keys = y[0].keys()

        if "loc" not in keys:
            raise ValueError("All elements of 'y' must have a 'loc' key.")

        if "scale" not in keys:
            raise ValueError("All elements of 'y' must have a 'scale' key.")

        if any(set(yi.keys()) != set(keys) for yi in y):
            raise ValueError("All elements of 'y' must have the 'loc' and 'scale' keys.")

        if weights is not None and len(weights) != len(y):
            raise ValueError("The length of `weights` must match the number of predictors in `y`.")

        # Stack the loc and scale arrays
        y_dict = defaultdict(list)
        for yi in y:
            for k, v in yi.items():
                y_dict[k].append(v)
        y = {k: v for k, v in y_dict.items()}

        self._np = np
        if all(isinstance(yi, np.ma.MaskedArray) for yi in y["loc"]) and all(
            isinstance(yi, np.ma.MaskedArray) for yi in y["scale"]
        ):
            self._np = np.ma

        y["loc"] = self._np.stack(y["loc"], axis=0)
        y["scale"] = self._np.stack(y["scale"], axis=0)

        loc = y["loc"]
        scale = y["scale"]

        mean_loc = self._np.average(loc, weights=weights, axis=0)
        agg = {"loc": mean_loc}

        if not self.decomposed_scale:
            sum_loc_scale = loc**2 + scale**2
            mean_scale = self._np.sqrt(
                self._np.average(sum_loc_scale, weights=weights, axis=0) - mean_loc**2
            )
            agg["scale"] = mean_scale

        else:
            # Here we assume that the mixture distribution is a normal distribution with a scale
            # that is the sum of the aleatoric and epistemic scales. This is a significant
            # approximation that could be improved by returning the true GMM.
            scale_aleatoric = self._np.sqrt(
                self._np.average(scale**2, weights=weights, axis=0),
            )
            scale_epistemic = self._np.sqrt(self._np.std(loc, axis=0) ** 2)
            agg["scale_aleatoric"] = scale_aleatoric
            agg["scale_epistemic"] = scale_epistemic

        return agg
