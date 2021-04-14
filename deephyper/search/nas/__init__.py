from deephyper.core.parser import add_arguments_from_signature
from deephyper.search import Search

__all__ = ["ambs", "random", "regevo", "agebo"]


class NeuralArchitectureSearch(Search):
    def __init__(
        self, problem, run="deephyper.nas.run.alpha.run", evaluator="ray", **kwargs
    ):

        super().__init__(problem, run=run, evaluator=evaluator, **kwargs)
        if self.problem._space["log_dir"] is None:
            self.problem._space["log_dir"] = self.log_dir

    @staticmethod
    def _extend_parser(parser):
        add_arguments_from_signature(parser, NeuralArchitectureSearch)
        return parser
