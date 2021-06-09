"""
deephyper ensemble uqbagging --model-dir save/model --loss tfp_nll --batch-size 1 --size 5 --verbose --mode regression --selection greedy
"""
import os
import sys

import ray
from deephyper.nas.losses import selectLoss
from deephyper.search.util import generic_loader, load_attr_from

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

    if not (ray.is_initialized()):
        ray.init(address=kwargs["ray_address"])

    ensemble_cls = ray.remote(
        num_cpus=kwargs.get("num_cpus", 1), num_gpus=kwargs.get("num_gpus")
    )(ensemble_cls)

    ensemble_obj = ensemble_cls.remote(
        model_dir=os.path.abspath(model_dir), size=size, verbose=verbose, loss=loss, **kwargs
    )

    load_data = problem.space["load_data"]["func"]
    load_data_kwargs = problem.space["load_data"].get("kwargs")
    (x, y), (vx, vy) = load_data() if kwargs is None else load_data(**load_data_kwargs)

    ray.get(ensemble_obj.fit.remote(vx, vy))

    ray.get(ensemble_obj.save.remote(os.path.abspath("members.json")))

