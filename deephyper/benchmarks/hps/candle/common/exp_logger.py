import json
import os
from datetime import datetime


def start(params, sys_desc):
    """Capture exp start """
    exp_start = datetime.now()
    experiment_id = params['experiment_id']

    search_space = []
    for key, val in params.items():
        search_space.append("{}: {}".format(key, val))

    msg = [{
        'experiment_id': experiment_id,
        'start_time': str(exp_start),
        'system_description': sys_desc,
        'search_space': search_space
    }]
    save("experiment_start.json", msg)

def end(experiment_id):
    """Capture exp end """
    exp_end = datetime.now()
    msg = [{
        'experiment_id': experiment_id,
        'status': {'set': 'Finished'},
        'end_time': {'set': str(exp_end)}
    }]
    save("experiment_end.json", msg)

def save(filename, msg):
    """Save log message"""
    path = os.getenv('TURBINE_OUTPUT')
    with open(path + "/" + filename, "w") as file_json:
        file_json.write(json.dumps(
            msg, indent=4, separators=(',', ': ')))
