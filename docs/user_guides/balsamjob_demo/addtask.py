import json
import os
import sys
import shlex

from balsam.core.models import BalsamJob, ApplicationDefinition
from deephyper.evaluator.balsam_utils import balsamjob_spec, JSONEncoder

app_name = "run_poly2"
here = os.path.dirname(os.path.abspath(__file__))
script_path = os.path.join(here, "model_run.py")
app_cmd = f'{sys.executable} {script_path}'

app, created = ApplicationDefinition.objects.get_or_create(
    name = app_name,
    defaults={'executable': app_cmd}
)
if not created:
    app.executable = app_cmd
    app.save()

@balsamjob_spec
def add_task(point):
    job = BalsamJob(
        application = app_name,
        args = shlex.quote(json.dumps(point, cls=JSONEncoder)),
        num_nodes = 1,
        ranks_per_node = 1,
    )
    return job