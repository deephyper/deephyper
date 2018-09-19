import os
import sys

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
top  =os.path.dirname(HERE) # directory containing deephyper
sys.path.append(top)

from deephyper.model.nas import run as train
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#param_dict = {"num_outputs": 10, "regression": False, "max_layers": 8, "layer_type": "conv2D", "state_space": [["filter_height", [3, 5, 7, 9]], ["filter_width", [3, 5, 7, 9]], ["pool_height", [2, 3, 4]], ["pool_width", [2, 3, 4]], ["stride_height", [1, 2]], ["stride_width", [1, 2]], ["drop_out", []], ["num_filters", [32, 64, 128, 256, 512]], ["skip_conn", []]], "max_episodes": 50, "hyperparameters": {"batch_size": 64, "eval_batch_size": 32, "activation": "relu", "learning_rate": 0.001, "optimizer": "adam", "num_epochs": 100, "loss_metric": "softmax_cross_entropy", "test_metric": "accuracy", "eval_freq": 10}, "load_data_module_name": "deephyper.benchmarks.cifar10Nas.load_data", "global_step": 0, "arch_seq": [[3.0, 9.0, 4.0, 3.0, 1.0, 1.0, 0.4802331030368805, 256.0, 0.0, 3.0, 9.0, 4.0, 3.0, 1.0, 1.0, 0.4804039001464844, 512.0, 0.0, 0.0, 3.0, 9.0, 4.0, 3.0, 1.0, 1.0, 0.4801722764968872, 256.0, 0.0, 0.0, 1.0, 3.0, 9.0, 4.0, 3.0, 1.0, 1.0, 0.4799531102180481, 512.0, 0.0, 0.0, 1.0, 0.0, 3.0, 9.0, 4.0, 3.0, 1.0, 1.0, 0.4802292585372925, 512.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 9.0, 4.0, 3.0, 1.0, 1.0, 0.4801389276981354, 256.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 3.0, 9.0, 4.0, 3.0, 1.0, 1.0, 0.47998419404029846, 512.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 3.0, 9.0, 4.0, 3.0, 1.0, 1.0, 0.48000261187553406, 512.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], "model_path": "", "stage_in_destination": ""}

param_dict = {"num_outputs": 10, "regression": False, "max_layers": 10, "layer_type": "conv2D", "state_space": [["filter_height", [3, 5]], ["filter_width", [3, 5]], ["pool_height", [1, 2]], ["pool_width", [1, 2]], ["stride_height", [1]], ["stride_width", [1]], ["drop_out", []], ["num_filters", [24, 36, 48, 64]], ["skip_conn", []]], "hyperparameters": {"batch_size": 64, "eval_batch_size": 64, "activation": "relu", "learning_rate": 0.1, "optimizer": "momentum", "num_epochs": 50, "loss_metric": "softmax_cross_entropy", "test_metric": "accuracy"}, "load_data_module_name": "deephyper.benchmarks.cifar10Nas.load_data", "global_step": 0, "num_worker": 0, "step": 0, "arch_seq": [[5.0, 3.0, 1.0, 2.0, 1.0, 1.0, 0.4646647572517395, 36.0, 0.0, 5.0, 3.0, 1.0, 2.0, 1.0, 1.0, 0.46416598558425903, 36.0, 0.0, 0.0, 5.0, 3.0, 1.0, 2.0, 1.0, 1.0, 0.46500393748283386, 36.0, 0.0, 0.0, 1.0, 5.0, 3.0, 1.0, 2.0, 1.0, 1.0, 0.46422216296195984, 36.0, 0.0, 0.0, 1.0, 1.0, 5.0, 3.0, 1.0, 2.0, 1.0, 1.0, 0.4652351140975952, 36.0, 0.0, 0.0, 1.0, 1.0, 0.0, 5.0, 3.0, 1.0, 2.0, 1.0, 1.0, 0.46335896849632263, 36.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 5.0, 3.0, 1.0, 2.0, 1.0, 1.0, 0.4636077582836151, 36.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 5.0, 3.0, 1.0, 2.0, 1.0, 1.0, 0.4634059965610504, 36.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 5.0, 3.0, 1.0, 2.0, 1.0, 1.0, 0.46421971917152405, 36.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 3.0, 1.0, 2.0, 1.0, 1.0, 0.4649583697319031, 36.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]}

print('-- -- -- -- > TRAINING START')
train(param_dict)
print('-- -- -- -- > TRAINING END')
