from jinja2 import Template, Environment, FileSystemLoader
import argparse
import json
import subprocess
import os
import sys
from socket import gethostname

def time_str(minutes):
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:00"

def default_conf():
    return '''{
    "DEEPHYPER_ENV_NAME":   "dl-hps",
    "DEEPHYPER_TOP":        "/projects/datascience/regele/deephyper",
    "DATABASE_TOP":         "/projects/datascience/regele/database",
    "BALSAM_PATH":          "/projects/datascience/regele/hpc-edge-service/balsam",
    "STAGE_IN_DIR":         "/local/scratch",
    "DISABLE_SUBMIT":       true
}
    ''' # TODO : What are DISABLE_SUBMIT & STAGE_IN_DIR ?

def check_conf(conf, args):
    assert os.path.exists(conf['DEEPHYPER_TOP'])
    assert os.path.exists(conf['DATABASE_TOP'])
    assert os.path.exists(conf['BALSAM_PATH'])
    assert args.time_minutes > 10, 'need more than 10 minutes'
    assert args.nodes == None or args.nodes > 2, 'need more than 2 nodes'

    env_name = conf["DEEPHYPER_ENV_NAME"]
    try:
        subprocess.run(f'source activate {env_name}', check=True, shell=True)
    except subprocess.CalledProcessError:
        raise ValueError(f"Cannot activate {env_name} referenced in runjob.conf")

    hostname = gethostname()
    if 'theta' in hostname: assert args.platform in ['theta', 'theta_postgres'], "please use a theta platform"
    else: assert args.platform == 'cooley', "please use cooley platform"

def get_conf(args):
    here = os.path.dirname(os.path.abspath(__file__))
    conf_fname = 'runjob.conf' if 'cooley' not in args.platform else 'runjob.conf.cooley'
    conf_fpath = os.path.join(here, conf_fname)
    if not os.path.exists(conf_fpath):
        with open(conf_fpath, 'w') as fp:
            fp.write(default_conf())
        print(f"Created {conf_fpath}. Please modify this file appropriately and re-run!")
        sys.exit(0)
    else:
        with open(conf_fpath) as fp: conf = json.load(fp)
        return conf

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('platform', choices=['cooley', 'theta', 'theta_postgres'])
    parser.add_argument('benchmark', choices=['mnistNas', 'cifar10Nas', 'ptbNas'])
    parser.add_argument('run_module_name', choices=['deephyper.run.nas', 'model.ptb_nas'])
    parser.add_argument('-q', required=True, dest='queue')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-n', type=int, dest='nodes', default=None)
    group.add_argument('-w', type=int, dest='num_workers', default=None)

    parser.add_argument('-sync', dest='sync', action='store_true', default=False)
    parser.add_argument('-t', type=int, required=True, dest='time_minutes')
    parser.add_argument('--stage-in-path')
    parser.add_argument('--saved-model-path')
    parser.add_argument('--max-evals', type=int, default=100000000)
    parser.add_argument('--eval-timeout-minutes', type=int, default=0)
    parser.add_argument('--ga-num-gen', type=int, default=100)
    parser.add_argument('--project', default='datascience')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    conf = get_conf(args)
    here = os.path.dirname(os.path.abspath(__file__))

    template_env = Environment(loader=FileSystemLoader(here))
    script_template = template_env.get_template('job_nas.tmpl')
    check_conf(conf, args)

    conf['platform'] = args.platform
    conf['method'] = 'NAS' # args.method
    conf['benchmark'] = args.benchmark
    conf['run_module_name'] = args.run_module_name
    conf['project'] = args.project
    conf['queue'] = args.queue

    if args.nodes is not None:
        conf['nodes'] = int(args.nodes)
        if args.platform == 'cooley':
            conf['num_workers'] = 2*conf['nodes'] - 2
        else:
            conf['num_workers'] = conf['nodes'] - 2

    if args.num_workers is not None:
        conf['num_workers'] = int(args.num_workers)
        if args.platform == 'cooley':
            conf['nodes'] = (conf['num_workers'] + 2) // 2 + (conf['num_workers']%2)
        else:
            conf['nodes'] = conf['num_workers'] + 2

    conf['sync'] = args.sync

    conf['time_minutes'] = args.time_minutes
    conf['time_str'] = time_str(args.time_minutes)
    conf['max_evals'] = args.max_evals
    conf['eval_timeout_minutes'] = args.eval_timeout_minutes

    if args.stage_in_path is not None:
        conf['STAGE_IN_DIR'] = args.stage_in_path
        print("Overriding STAGE_IN_DIR with", args.stage_in_path)
    #if args.method == 'hyperband':
    #    modelpath = os.path.abspath(os.path.expanduser(args.saved_model_path))
    #    conf['saved_model_path'] = modelpath

    jobname = '.'.join(str(conf[key]) for key in 'benchmark nodes'.split())
    if args.platform == 'cooley':
        jobname += '.cooley'
    elif args.platform == 'theta':
        jobname += '.theta'
    jobname += '.sync' if conf['sync'] else '.async'
    jobname += '.'+str(conf['time_minutes'])
    db_path = os.path.join(conf['DATABASE_TOP'], jobname)
    i = 0
    while os.path.exists(db_path):
        db_path = os.path.join(conf['DATABASE_TOP'], jobname) + f".run{i}"
        i += 1
    conf['db_path'] = db_path
    conf['jobname'] = jobname

    run_name = os.path.basename(db_path)
    run_dir = os.path.join(here, 'runs')
    if not os.path.exists(run_dir): os.makedirs(run_dir)

    with open(os.path.join(run_dir, run_name+'.sh'), 'w') as fp:
        fp.write(script_template.render(conf))
        fname = fp.name
        cwd = os.path.dirname(fname)
        subprocess.run(f'chmod +x {fname}', check=True, shell=True)

    print(f"CREATED JOB IN {fname}")
    if not conf['DISABLE_SUBMIT']:
        subprocess.run(f'qsub -O {run_name} --cwd {cwd} {fname}', check=True, shell=True)
        print("qsub done!")
        print("Job database will be at:", db_path)
    else:
        print("Dry run -- change DISABLE_SUBMIT in runjob.conf to enable auto-submission")

if __name__ == "__main__":
    main()
