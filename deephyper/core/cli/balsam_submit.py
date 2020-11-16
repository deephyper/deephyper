import argparse
import os
import sys
from deephyper.search.util import generic_loader, banner
from deephyper.problem.neuralarchitecture import Problem
from deephyper.problem import BaseProblem


def add_subparser(subparsers):
    subparser_name = "balsam-submit"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Create and submit an HPS or NAS job directly via Balsam."
    )

    subparser.add_argument("mode", choices=["nas", "hps"], help="Type of search")
    subparser.add_argument("workflow", help="Unique workflow name")
    subparser.add_argument(
        "-p", "--problem", required=True, help="Problem definition path or object"
    )
    subparser.add_argument(
        "-r", "--run", required=True, help="Run function path or object"
    )

    subparser.add_argument(
        "-t", "--time-minutes", type=int, required=True, help="Job walltime (minutes)"
    )
    subparser.add_argument(
        "-n", "--nodes", type=int, required=True, help="Number of nodes"
    )
    subparser.add_argument("-q", "--queue", required=True, help="Queue to submit into")
    subparser.add_argument("-A", "--project", required=True, help="Project Name")
    subparser.add_argument(
        "-j",
        "--job-mode",
        required=True,
        help="Launcher job mode",
        choices=["mpi", "serial"],
    )
    subparser.add_argument(
        "--nas-nodes",
        default=1,
        type=int,
        help="Number of nodes over which to parallelize NAS",
    )
    subparser.set_defaults(func=function_to_call)


def main(mode, workflow, problem, run, **kwargs):
    """Create & submit the DH search via Balsam"""
    job = pre_submit(problem, run, workflow)
    if os.path.exists(problem):
        problem = os.path.abspath(problem)
    if os.path.exists(run):
        run = os.path.abspath(run)

    if mode == "nas":
        print("Creating NAS(PPO) BalsamJob...", end="", flush=True)
        setup_nas(job, problem, run, kwargs["nas_nodes"])
        print("OK")
    elif mode == "hps":
        print("Creating HPS(AMBS) BalsamJob...", end="", flush=True)
        setup_ambs(job, problem, run)
        print("OK")

    print("Performing job submission...")
    submit_qlaunch(
        kwargs["project"],
        kwargs["queue"],
        kwargs["nodes"],
        kwargs["time_minutes"],
        kwargs["job_mode"],
        workflow,
    )
    banner(f"Success. The search will run at: {job.working_directory}")


def validate(problem, run, workflow):
    """Validate problem, run, and workflow"""
    from balsam.core.models import BalsamJob

    print("Validating Problem...", end="", flush=True)
    prob = generic_loader(problem, "Problem")
    assert isinstance(prob, (Problem, BaseProblem)), f"{prob} is not a Problem instance"
    print("OK", flush=True)

    print("Validating run...", end="", flush=True)
    run = generic_loader(run, "run")
    assert callable(run), f"{run} must be a a callable"
    print("OK", flush=True)

    qs = BalsamJob.objects.filter(workflow=workflow)
    if qs.exists():
        print(f"There are already jobs matching workflow {workflow}")
        print("Please remove these, or use a unique workflow name")
        sys.exit(1)


def bootstrap_apps():
    """Ensure Balsam ApplicationDefinitions are populated"""
    from balsam.core.models import ApplicationDefinition

    apps = {
        "AMBS": f"{sys.executable} -m deephyper.search.hps.ambs",
        "NAS-AMBS": f"{sys.executable} -m deephyper.search.nas.ambs",
        "NAS-RANDOM": f"{sys.executable} -m deephyper.search.nas.random",
        "NAS-REGEVO": f"{sys.executable} -m deephyper.search.nas.regevo",
        "NAS-AGEBO": f"{sys.executable} -m deephyper.search.nas.agebo",
    }

    for name, exe in apps.items():
        app, created = ApplicationDefinition.objects.get_or_create(
            name=name, defaults={"executable": exe}
        )
        if not created:
            app.executable = exe
            app.save()


def pre_submit(problem, run, workflow):
    """Validate command line; prepare apps"""
    from balsam.core.models import BalsamJob

    validate(problem, run, workflow)
    print("Bootstrapping apps...", end="", flush=True)
    bootstrap_apps()
    print("OK")

    job = BalsamJob(name=workflow, workflow=workflow)
    return job


def setup_ambs(job, problem, run):
    job.application = "AMBS"
    job.args = f"--evaluator balsam --problem {problem} --run {run}"
    job.save()
    return job


def setup_nas(job, problem, run, nas_nodes):
    job.application = "PPO"
    job.args = f"--evaluator balsam --problem {problem}"
    job.num_nodes = nas_nodes
    job.save()
    return job


def submit_qlaunch(project, queue, nodes, time_minutes, job_mode, wf_filter):
    """Submit Balsam launcher job to batch scheduler"""
    from balsam.service import service
    from balsam.core import models

    QueuedLaunch = models.QueuedLaunch
    qlaunch = QueuedLaunch(
        project=project,
        queue=queue,
        nodes=nodes,
        wall_minutes=time_minutes,
        job_mode=job_mode,
        wf_filter=wf_filter,
        prescheduled_only=False,
    )
    qlaunch.save()
    service.submit_qlaunch(qlaunch, verbose=True)
