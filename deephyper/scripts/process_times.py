from itertools import chain
import sys
import numpy as np
try:
    from matplotlib import pyplot as plt
except:
    plt = None

from balsam.launcher import dag
from balsam.service.models import process_job_times

def analyze(plot=False, csv=False):
    TIME_STEP = 0.05

    state_times = process_job_times(state0='PREPROCESSED')
    next_states = { 
                   'CREATED' : ['RUNNING'],
                   'PREPROCESSED' : ['RUNNING'],
                   'RUNNING': ['RUN_DONE', 'RUN_ERROR', 'FAILED', 'RUN_TIMEOUT', 'USER_KILLED'],
                   'RUN_DONE' : [],
                   'RUN_ERROR': [],
                   'USER_KILLED': [],
                   'RUN_TIMEOUT': [],
                   'FAILED' : [],
                   'JOB_FINISHED': [],
                  }

    order = 'CREATED PREPROCESSED RUNNING RUN_DONE JOB_FINISHED RUN_ERROR USER_KILLED RUN_TIMEOUT FAILED'.split()
    order = {key : i for key,i in zip(order, range(len(order)))}
    states = sorted(state_times.keys(), key=lambda x: order[x])

    for state in states:
        state_times[state] = np.asarray(state_times[state], dtype=np.float64)
        state_times[state] = state_times[state][:, np.newaxis]

    times = np.asarray(list(chain(*state_times.values())))
    time_grid = np.arange(times.min(), times.max(), TIME_STEP)

    state_counts = {}
    for state in states:
        counts = (state_times[state] <= time_grid).sum(axis=0)
        assert counts.shape == time_grid.shape

        for state2 in next_states[state]:
            if state2 in states:
                counts -= (state_times[state2] <= time_grid).sum(axis=0)
                assert counts.shape == time_grid.shape
        state_counts[state] = counts

    # Generate report
    report_states = ['CREATED', 'RUNNING', 'RUN_DONE']
    if 'RUN_ERROR' in states: report_states.append('RUN_ERROR')
    if 'USER_KILLED' in states: report_states.append('USER_KILLED')

    data = np.hstack( (state_counts[s].reshape(len(time_grid),1) for s in report_states ) )
    data = np.hstack((time_grid.reshape(len(time_grid),1), data))
    assert data.shape == (len(time_grid), len(report_states)+1)

    if csv:
        header = ["Elapsed_sec"]
        header.extend(report_states)
        print('#', ' '.join(header))
        for row in data: print(' '.join(map(str, row)))

    if plot:
        if plt is None:
            raise RuntimeError("Can't plot because matplotlib could not be imported!")
        for i, state in enumerate(report_states, 1):
            plt.plot(data[:,0], data[:,i], label=state)
        plt.xlabel('Elapsed seconds')
        plt.ylabel('Job count')
        plt.legend()
        plt.show()

    return state_counts


if __name__ == "__main__":
    csv = False
    plot = False
    if 'csv' in sys.argv: csv = True
    if 'plot' in sys.argv: plot = True
    if not (csv or plot):
        print("Please add either 'csv' or 'plot' (or both) keywords to command line")
    else:
        analyze(plot=plot, csv=csv)
