import sys
from pprint import pprint
import os
import pickle

here = os.path.dirname(os.path.abspath(__file__))
top = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.append(top)
BNAME = os.path.splitext(os.path.basename(__file__))[0]

from deephyper.benchmarks import util, keras_cmdline

timer = util.Timer()
timer.start('module loading')

import numpy as np
timer.end()

class Model:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def save(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump(self, fp)

def linear(x, a, b):
    return a*x + b

def run(param_dict):
    param_dict = keras_cmdline.fill_missing_defaults(augment_parser, param_dict)
    pprint(param_dict)
    
    timer.start('stage in')
    if param_dict['data_source']:
        data_source = param_dict['data_source']
    else:
        data_source = os.path.dirname(os.path.abspath(__file__))
        data_source = os.path.join(data_source, 'data')

    paths = util.stage_in(['dataset'], source=data_source, dest=param_dict['stage_in_destination'])
    path = paths['dataset']
    
    data = np.loadtxt(path)
    training_x = data[:,0]
    training_y = data[:,1]
    n_pt = len(training_x)
    timer.end()

    timer.start('preprocessing')
    penalty = param_dict['penalty']
    epochs = param_dict['epochs']
    if type(epochs) is not int:
        print("converting epochs to int:", epochs)
        epochs = int(epochs)
    lr = param_dict['lr']
    
    model_path = param_dict['model_path']
    model_mda_path = None
    model = None
    initial_epoch = 0

    if model_path:
        savedModel = util.resume_from_disk(BNAME, param_dict, data_dir=model_path)
        model_mda_path = savedModel.model_mda_path
        model_path = savedModel.model_path
        model = savedModel.model
        initial_epoch = savedModel.initial_epoch

    if model is None:
        a = np.random.uniform(-0.4, 0.4)
        b = np.random.uniform(0, 1)
        print("starting new model", a, b)
    else:
        a, b = model.a, model.b
        print("loaded model from disk:", a, b)
        print("on epoch", initial_epoch)

    timer.end()

    timer.start('model training')
    for i in range(initial_epoch, epochs):
        predict = linear(training_x, a, b)
        error = predict - training_y
        grad_b = error.sum() / n_pt
        grad_a = (error*training_x).sum() / n_pt

        a -= lr * grad_a
        b -= lr * grad_b
    timer.end()

    print(f"training done\na={a}\nb={b}")
    predict = linear(training_x, a, b)
    error = predict - training_y
    mse = 0.5 * (error**2).sum() / n_pt
    mse += penalty
    print("OUTPUT:", mse)

    if model_path:
        timer.start('model save')
        model = Model(a, b)
        model.save(model_path)
        util.save_meta_data(param_dict, model_mda_path)
        timer.end()
        print(f"saved model to {model_path} and MDA to {model_mda_path}")
    return mse

def augment_parser(parser):
    parser.add_argument('--penalty', type=float, default=0.0)
    return parser


if __name__ == "__main__":
    parser = keras_cmdline.create_parser()
    parser = augment_parser(parser)
    cmdline_args = parser.parse_args()
    param_dict = vars(cmdline_args)
    run(param_dict)
