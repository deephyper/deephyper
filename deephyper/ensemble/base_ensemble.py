import argparse
import traceback
import os
import json

import tensorflow as tf
import ray



class BaseEnsemble:
    def __init__(
        self,
        model_dir,
        loss,
        size=5,
        verbose=True,
        ray_address="",
        num_cpus=1,
        num_gpus=None,
    ):
        self.model_dir = os.path.abspath(model_dir)
        self.loss = loss
        self.members_files = []
        self.size = size
        # assert self.size >= 2, "an ensemble size must be >= 2"

        self.verbose = verbose
        if self.verbose:
            print(self)

        self.ray_address = ray_address
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus

        if not(ray.is_initialized()):
            ray.init(address=self.ray_address)

    def __repr__(self) -> str:
        out = ""
        out += f"Model Dir: {self.model_dir}\n"
        out += f"Members files: {self.members_files}\n"
        out += f"Ensemble size: {self.size}\n"
        return out

    def list_all_model_files(self):
        return [f for f in os.listdir(self.model_dir) if f[-2:] == "h5"]

    @classmethod
    def get_parser(cls, parser=None) -> argparse.ArgumentParser:
        """Return the fully extended parser.

        Returns:
            ArgumentParser: the fully extended parser.
        """
        base_parser = cls._base_parser(parser)
        parser = cls._extend_parser(base_parser)
        return parser

    @classmethod
    def parse_args(cls, arg_str=None) -> None:
        parser = cls.get_parser()
        if arg_str is not None:
            return parser.parse_args(arg_str)
        else:
            return parser.parse_args()

    @staticmethod
    def _extend_parser(base_parser) -> argparse.ArgumentParser:
        return base_parser

    @staticmethod
    def _base_parser(parser=None) -> argparse.ArgumentParser:
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler="resolve")

        parser.add_argument("--problem", type=str, required=True)
        parser.add_argument("--loss", type=str, default=None)
        parser.add_argument("--model-dir", type=str, required=True)
        parser.add_argument("--size", type=int, default=3)
        parser.add_argument(
            "-v", "--verbose", type=bool, const=True, default=False, nargs="?"
        )
        parser.add_argument("--batch-size", default=1)
        parser.add_argument("--ray-address", default="")
        parser.add_argument("--num-cpus", default=1)
        parser.add_argument("--num-gpus", default=None)

        return parser

    def fit(self, X, y) -> None:
        raise NotImplementedError

    def load_models(self, model_files: list):
        for i, model_file in enumerate(model_files):
            model_path = os.path.join(self.model_dir, model_file)
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                model._name = f"model_{i}"
            except:
                print(f"Could not load model {i}")
                traceback.print_exc()
                model = None
            yield model

    def load_members_files(self, file: str = "members.json") -> None:
        with open(file, "r") as f:
            self.members_files = json.load(f)

    def save_members_files(self, file: str = "members.json") -> None:
        with open(file, "w") as f:
            json.dump(self.members_files, f)

    def load(self, file: str) -> None:
        raise NotImplementedError

    def save(self, file: str) -> None:
        raise NotImplementedError