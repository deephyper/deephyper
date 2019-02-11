import json
import sys
import os
import datetime
from shutil import copyfile

try:
    from balsam.core.models import (BalsamJob, process_job_times, utilization_report)
    BALSAM_EXIST = True
    print('Module: \'balsam\' has been loaded successfully!')
except ModuleNotFoundError as err:
    BALSAM_EXIST = False
    print('Module: \'balsam\' was not found!')


HERE = os.path.dirname(os.path.abspath(__file__))
now = '_'.join(str(datetime.datetime.now(datetime.timezone.utc)).split(":")[0].split(" "))

def get_workload(wf_name):
    qs = BalsamJob.objects.filter(workflow=wf_name)
    time_data = process_job_times(qs)
    times, num_running = utilization_report(time_data)
    times = [str(t) for t in times]
    num_running = [int(n) for n in num_running]
    return times, num_running


def max_list(l):
    rl = [l[0]]
    mx = l[0]
    for i in range(1, len(l)):
        mx = max(mx, l[i])
        rl.append(mx)
    return rl

def parseline_json(line, data):
    line = "".join(line)
    date = line.split('|')[0]
    jsn_str = line.split('>>>')[-1]
    info = json.loads(jsn_str)
    if data.get(info['type']) == None:
        data[info['type']] = list()
    value = info['type']
    info['timestamp'] = date
    info.pop('type')
    data[value].append(info)


def parseline_reward(line, data):
    data['raw_rewards'].append(float(line[-1]))

def parseline_id_worker(line, data):
    i = line.index("'w':") + 1
    id_int = int(line[i][:-1])
    data['id_worker'].append(id_int)
    i = line.index("'rank':") + 1
    id_int = int(line[i][:-1])
    data['rank'].append(id_int)

def parseline_arch_seq(line, data):
    i_sta = line.index("'arch_seq':") + 1
    i_end = i_sta
    while not ']' in line[i_end]:
        i_end += 1
    l = []
    for i in range(i_sta, i_end+1):
        l.append(float(line[i].replace('[', '').replace(',', '').replace(']', '')))
    data['arch_seq'].append(l)

def parsing(f, data):
    line = f.readline()
    while line:
        line = line.split()
        if "y:" in line:
            parseline_reward(line, data)
            parseline_arch_seq(line, data)
            parseline_id_worker(line, data)
        elif ">>>" in line:
            parseline_json(line, data)

        line = f.readline()

def main():

    if len(sys.argv) == 1:
        print(
            f' -- HELP -- \n'
            f'The parsing script takes only 1 argument: the relative path to the log file.\n'
            f'If you want to compute the workload data with \'balsam\' you should specify a path starting at least from the workload parent directory,\n'
            f'eg. \'nas_exp1/nas_exp1_ue28s2k0/deephyper.log\'\n'
            f'where \'nas_exp1\' is the workload.')
        return
    else:
        path = sys.argv[1]
    print(f'Path to deephyper.log file: {path}')

    data = dict()
    if len(path.split('/')) >= 3:
        data['fig'] = path.split('/')[-3] + '_' + now
        workload_in_path = True
    else:
        workload_in_path = False
        data['fig'] = 'data_' + now

    data['raw_rewards'] = list()
    data['max_rewards'] = list()
    data['arch_seq'] = list()
    data['id_worker'] = list()
    data['rank'] = list()

    with open(path, 'r') as flog:
        print('File has been opened')
        parsing(flog, data)
        data['max_rewards'] = max_list(data['raw_rewards'])
    print('File closed')

    if BALSAM_EXIST and workload_in_path:
        try:
            print('Computing workload!')
            times, num_running = get_workload(path.split('/')[-3])
            data['workload'] = dict(times=times, num_running=num_running)
        except:
            print('Failed to compute workload!...')
        else:
            print('Workload has been computed successfuly!')


    with open(HERE+'/json/'+data['fig']+'.json', 'w') as fjson:
        print(f'Create json file: {data["fig"]+".json"}')
        json.dump(data, fjson, indent=2)
    print('Json dumped!')

    print(f'len raw_rewards: {len(data["raw_rewards"])}')
    print(f'len max_rewards: {len(data["max_rewards"])}')
    print(f'len id_worker  : {len(data["id_worker"])}')
    print(f'len arch_seq   : {len(data["arch_seq"])}')

if __name__ == '__main__':
    main()
