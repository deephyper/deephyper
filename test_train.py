import os
import sys

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
top  =os.path.dirname(HERE) # directory containing deephyper
sys.path.append(top)

from deephyper.model.nas import run as train
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

param_dict = {"num_outputs": 10, "regression": False, "max_layers": 8, "layer_type": "conv2D", "state_space": [["filter_height", [3, 5, 7, 9]], ["filter_width", [3, 5, 7, 9]], ["pool_height", [2, 3, 4]], ["pool_width", [2, 3, 4]], ["stride_height", [1, 2]], ["stride_width", [1, 2]], ["drop_out", []], ["num_filters", [32, 64, 128, 256, 512]], ["skip_conn", []]], "max_episodes": 50, "hyperparameters": {"batch_size": 64, "eval_batch_size": 32, "activation": "relu", "learning_rate": 0.001, "optimizer": "adam", "num_epochs": 100, "loss_metric": "softmax_cross_entropy", "test_metric": "accuracy", "eval_freq": 10}, "load_data_module_name": "deephyper.benchmarks.cifar10Nas.load_data", "global_step": 0, "arch_seq": [[3.0, 9.0, 4.0, 3.0, 1.0, 1.0, 0.4802331030368805, 256.0, 0.0, 3.0, 9.0, 4.0, 3.0, 1.0, 1.0, 0.4804039001464844, 512.0, 0.0, 0.0, 3.0, 9.0, 4.0, 3.0, 1.0, 1.0, 0.4801722764968872, 256.0, 0.0, 0.0, 1.0, 3.0, 9.0, 4.0, 3.0, 1.0, 1.0, 0.4799531102180481, 512.0, 0.0, 0.0, 1.0, 0.0, 3.0, 9.0, 4.0, 3.0, 1.0, 1.0, 0.4802292585372925, 512.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 9.0, 4.0, 3.0, 1.0, 1.0, 0.4801389276981354, 256.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 3.0, 9.0, 4.0, 3.0, 1.0, 1.0, 0.47998419404029846, 512.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 3.0, 9.0, 4.0, 3.0, 1.0, 1.0, 0.48000261187553406, 512.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], "model_path": "", "stage_in_destination": ""}

print('-- -- -- -- > TRAINING START')
train(param_dict)
print('-- -- -- -- > TRAINING END')
