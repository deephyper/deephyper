from deephyper.core.parser import add_arguments_from_signature
from deephyper.search import Search

class NeuralArchitectureSearch(Search):
    def __init__(self, problem,
        run="deephyper.search.nas.model.run.alpha.run",
        evaluator="ray",
        **kwargs):

        super().__init__(problem, run=run, evaluator=evaluator, **kwargs)


    @staticmethod
    def _extend_parser(parser):
        add_arguments_from_signature(parser, NeuralArchitectureSearch)
        return parser