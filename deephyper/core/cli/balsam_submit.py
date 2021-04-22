import os
import sys

try:
    from balsam.core import models
    from balsam.core.models import ApplicationDefinition, BalsamJob
    from balsam.service import service
except:
    pass

from deephyper.core.cli.utils import generate_other_arguments
from deephyper.core.exceptions import DeephyperRuntimeError
from deephyper.problem import BaseProblem, NaProblem
from deephyper.search.util import banner, generic_loader

APPS = {
    "HPS": {
        "AMBS": f"{sys.executable} -m deephyper.search.hps.ambs",
    },
    "NAS": {
        "AMBS": f"{sys.executable} -m deephyper.search.nas.ambs",
        "RANDOM": f"{sys.executable} -m deephyper.search.nas.random",
        "REGEVO": f"{sys.executable} -m deephyper.search.nas.regevo",
        "AGEBO": f"{sys.executable} -m deephyper.search.nas.agebo",
    },
}


def add_subparser(subparsers):
    subparser_name = "balsam-submit"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Create and submit an HPS or NAS job directly via Balsam."
    )

    subparser.add_argument("mode", choices=["nas", "hps"], help="Type of search")
    subparser.add_argument(
        "search",
        choices=["ambs", "regevo", "random", "agebo", "ambsmixed", "regevomixed"],
        help="Search strategy",
    )
    subparser.add_argument("workflow", help="Unique workflow name")
    subparser.add_argument(
        "-p", "--problem", required=True, help="Problem definition path or object"
    )
    subparser.add_argument(
        "-r", "--run", required=False, help="Run function path or object"
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
        "--num-evals-per-node",
        default=1,
        type=int,
        help="Number of evaluations performed on each node. Only valid if evaluator==balsam and balsam job-mode is 'serial'.",
    )
    subparser.set_defaults(func=function_to_call)


def main(mode: str, search: str, workflow: str, problem: str, run: str, **kwargs) -> None:
    """Create & submit the DH search via Balsam"""

    job = pre_submit(mode, search, workflow, problem, run)

    if os.path.exists(problem):  # the problem was given as a PATH
        problem = os.path.abspath(problem)
    if run and os.path.exists(run):  # the run function was given as a PATH
        run = os.path.abspath(run)

    print(
        f"Creating BalsamJob using application {job.application}...", end="", flush=True
    )
    setup_job(job, problem, run, **kwargs)
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


def validate(mode: str, search: str, workflow: str, problem: str, run: str) -> str:
    """Validate problem, run, and workflow"""

    # validate the mode
    if not (mode.upper() in APPS):
        raise DeephyperRuntimeError(f"The mode '{mode}' is not valid!")

    # validate the search
    if not (search.upper()in APPS[mode.upper()]):
        raise DeephyperRuntimeError(f"The search '{search}' is not valid!")
    app = f"{mode.upper()}-{search.upper()}"

    print(f"Validating Problem({problem})...", end="", flush=True)
    prob = generic_loader(problem, "Problem")
    assert isinstance(prob, (NaProblem, BaseProblem)), f"{prob} is not a Problem instance"
    print("OK", flush=True)

    # validate run
    if run: # it is not mandatory to pass a run function for NAS
        print("Validating run...", end="", flush=True)
        run = generic_loader(run, "run")
        assert callable(run), f"{run} must be a a callable"
        print("OK", flush=True)
    else:
        if mode == "hps":
            raise DeephyperRuntimeError(f"No '--run' was passed for the mode 'hps'")

    qs = BalsamJob.objects.filter(workflow=workflow)
    if qs.exists():
        raise DeephyperRuntimeError(
            f"There are already jobs matching workflow {workflow}"
            f"Please remove these, or use a unique workflow name"
        )

    return app


def bootstrap_apps():
    """Ensure Balsam ApplicationDefinitions are populated"""

    for mode, mode_apps in APPS.items():
        for app_name, app_exe in mode_apps.items():
            app, created = ApplicationDefinition.objects.get_or_create(
                name=f"{mode}-{app_name}", defaults={"executable": app_exe}
            )
            if not created:
                app.executable = app_exe
                app.save()


def pre_submit(
        mode: str, search: str, workflow: str, problem: str, run: str
):
    """Validate command line; prepare apps"""

    app = validate(mode, search, workflow, problem, run)

    # creating the APPS in the balsam DB
    print("Bootstrapping apps...", end="", flush=True)
    bootstrap_apps()
    print("OK")

    job = BalsamJob(name=workflow, workflow=workflow, application=app)
    return job


def setup_job(job, problem, run, **kwargs):
    job.args = f"--evaluator balsam --problem {problem}"

    #! it is not required for NAS to pass a run function
    if run:
        job.args += f" --run {run}"

    invalid_keys = ["time_minutes", "nodes", "queue", "project", "job_mode"]
    for k in invalid_keys:
        kwargs.pop(k)
    args = generate_other_arguments(**kwargs)
    if len(args) > 0:
        job.args += f" {args}"

    job.save()
    return job


def submit_qlaunch(project, queue, nodes, time_minutes, job_mode, wf_filter):
    """Submit Balsam launcher job to batch scheduler"""

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
