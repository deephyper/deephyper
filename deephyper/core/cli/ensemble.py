"""
deephyper ensemble uqbagging --model-dir save/model --loss tfp_nll --batch-size 1 --size 5 --verbose --mode regression --selection greedy
"""
import sys


from deephyper.search.util import load_attr_from, generic_loader
from deephyper.nas.losses import selectLoss

ENSEMBLE_CLASSES = {
    "uqbagging": "deephyper.nas.ensemble.uq_bagging_ensemble.UQBaggingEnsemble",
}


def add_subparser(parsers):
    parser_name = "ensemble"

    parser = parsers.add_parser(
        parser_name,
        help="Command line to build ensemble from models saved by neural architecture search.",
    )

    subparsers = parser.add_subparsers()

    for name, module_attr in ENSEMBLE_CLASSES.items():
        ensemble_cls = load_attr_from(module_attr)

        subparser = subparsers.add_parser(name=name, conflict_handler="resolve")
        subparser = ensemble_cls.get_parser(subparser)

        subparser.set_defaults(func=main)


def main(problem, model_dir, size, verbose, loss, **kwargs):

    ensemble_type = sys.argv[2]
    assert ensemble_type in ENSEMBLE_CLASSES

    problem = generic_loader(problem, "Problem")
    if loss is None:
        loss = problem.space["loss"]
    loss = selectLoss(loss)

    ensemble_cls = load_attr_from(ENSEMBLE_CLASSES[ensemble_type])

    ensemble_obj = ensemble_cls(
        model_dir=model_dir, size=size, verbose=verbose, loss=loss, **kwargs
    )

    load_data = problem.space["load_data"]["func"]
    load_data_kwargs = problem.space["load_data"].get("kwargs")
    (x, y), (vx, vy) = load_data() if kwargs is None else load_data(**load_data_kwargs)

    ensemble_obj.fit(vx, vy)

    ensemble_obj.save("members.json")
