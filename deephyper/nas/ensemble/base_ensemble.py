import argparse
import os
import json

import tensorflow as tf

from deephyper.search import util as search_util


class BaseEnsemble:
    def __init__(
        self,
        problem,
        model_dir,
        members_files=None,
        size=5,
        history_dir=None,
        output_members_files=None,
        verbose=True,
    ):
        self.problem = search_util.generic_loader(problem, "Problem")
        self.model_dir = os.path.abspath(model_dir)
        self.members_files = self.load_members_files(members_files)
        self.members_models = list(self.load_models(self.members_files))
        self.size = size if len(self.members_models) == 0 else len(self.members_models)
        self.history_dir = history_dir
        self.output_members_files = output_members_files
        self.verbose = verbose
        if self.verbose:
            print(self)

    def __repr__(self) -> str:
        out = ""
        out += f"Model Dir: {self.model_dir}\n"
        out += f"Members files: {self.members_files}\n"
        out += f"Ensemble size: {self.size}\n"
        out += f"{self.problem}"
        return out

    def load_members_files(self, members_files) -> list:
        if members_files is None:
            return []

        if type(members_files) is list:
            return members_files

        if type(members_files) is str:
            if not (os.path.exists(members_files)):
                return []
            else:
                with open(members_files, "r") as f:
                    model_files = json.load(f)
                return model_files

        return []

    def load_models(self, model_files: list):
        for model_file in model_files:
            model_path = os.path.join(self.model_dir, model_file)
            model = tf.keras.models.load_model(model_path, compile=False)
            yield model

    def save_members_files(self):
        if type(self.output_members_files) is str:
            with open(self.output_members_files, "w") as f:
                json.dump(self.members_files, f)

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
        parser.add_argument(
            "--problem",
            type=str,
            help="Module path to the Problem instance you want to use for the search.",
        )
        parser.add_argument("-md", "--model-dir", type=str)
        parser.add_argument("-mf", "--members-files", type=str, default="members.json")
        parser.add_argument(
            "-omf", "--output-members-files", type=str, default="members.json"
        )
        parser.add_argument("-s", "--size", type=int, default=3)
        parser.add_argument("-hd", "--history-dir", type=str, default=None)
        parser.add_argument("-v", "--verbose", type=bool, default=True)
        return parser

    def main(self):
        raise NotImplemented