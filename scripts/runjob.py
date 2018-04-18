from jinja2 import Template, Environment, FileSystemLoader
import argparse
import json
import subprocess
import os
from socket import gethostname

def time_str(minutes):
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:00"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('platform', choices=['cooley', 'theta', 'theta_postgres'])
    parser.add_argument('method', choices=['xgb', 'rf', 'rs', 'ga', 'hyperband'])
    parser.add_argument('benchmark', choices=['b1.addition_rnn',
                                             'b2.babi_memnn',
                                             'b3.babi_rnn',
                                             'candlep1b1.p1b1_baseline_keras2',
                                             'capsule.capsule',
                                             'cifar10cnn.cifar10_cnn',
                                             'gcn.gcn',
                                             'mnistmlp.mnist_mlp',
                                             'dummy1.three_hump_camel',
                                             'dummy2.regression',
                                             'rosen2.rosenbrock2',
                                             'rosen10.rosenbrock10',
                                             'rosen30.rosenbrock30',
                                             ]
                       )
    parser.add_argument('-q', required=True, dest='queue')
    parser.add_argument('-n', type=int, required=True, dest='nodes')
    parser.add_argument('-t', type=int, required=True, dest='time_minutes')
    parser.add_argument('--stage-in-path')
    parser.add_argument('--saved-model-path')
    parser.add_argument('--max-evals', type=int, default=100000000)
    parser.add_argument('--project', default='datascience')
    return parser


def check_conf(conf, args):
    assert os.path.exists(conf['DEEPHYPER_TOP'])
    assert os.path.exists(conf['DATABASE_TOP'])
    assert os.path.exists(conf['BALSAM_PATH'])
    assert args.time_minutes > 10, 'need at least 10 minutes'

    env_name = conf["DEEPHYPER_ENV_NAME"]
    try:
        subprocess.run(f'source activate {env_name}', check=True, shell=True)
    except subprocess.CalledProcessError:
        raise ValueError(f"Cannot activate {env_name} referenced in runjob.conf")
    
    if args.method == 'hyperband':
        assert args.saved_model_path is not None, 'hyperband requires --saved_model_path'
        args.saved_model_path = os.path.abspath(os.path.expanduser(args.saved_model_path))
        assert os.path.exists(args.saved_model_path), f'{args.saved_model_path} not found'

    hostname = gethostname()
    if 'theta' in hostname: assert args.platform in ['theta', 'theta_postgres'], "please use a theta platform"
    else: assert args.platform == 'cooley', "please use cooley platform"

def main():
    parser = get_parser()
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    conf_fname = 'runjob.conf' if 'cooley' not in args.platform else 'runjob.conf.cooley'
    with open(os.path.join(here, conf_fname)) as fp: conf = json.load(fp)
    template_env = Environment(loader=FileSystemLoader(here))#, lstrip_blocks=True, trim_blocks=True)
    script_template = template_env.get_template('job.tmpl')
    #template_env.trim_blocks = True
    #template_env.lstrip_blocks = True
    check_conf(conf, args)

    conf['platform'] = args.platform
    conf['method'] = args.method
    conf['benchmark'] = args.benchmark
    conf['project'] = args.project
    conf['queue'] = args.queue
    conf['nodes'] = args.nodes
    conf['time_minutes'] = args.time_minutes
    conf['time_str'] = time_str(args.time_minutes)
    conf['max_evals'] = args.max_evals
    if args.stage_in_path is not None:
        conf['STAGE_IN_DIR'] = args.stage_in_path
        print("Overriding STAGE_IN_DIR with", args.stage_in_path)
    if args.method == 'hyperband':
        modelpath = os.path.abspath(os.path.expanduser(args.saved_model_path))
        conf['saved_model_path'] = modelpath

    jobname = '.'.join(str(conf[key]) for key in 'benchmark nodes method'.split())
    if args.platform == 'cooley': 
        jobname += '.gpu'
    elif args.platform == 'theta_postgres':
        jobname += '.pg'
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

    print(f"SUBMITTING JOB IN {fname}")
    if not conf['DISABLE_SUBMIT']:
        subprocess.run(f'qsub -O {run_name} --cwd {cwd} {fname}', check=True, shell=True)
        print("qsub done!")

if __name__ == "__main__":
    main()
