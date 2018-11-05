''' Import TensorRT Modules '''
import tensorrt as trt
import uff
from tensorrt.parsers import uffparser

config = {
    # Where to save models (Tensorflow + TensorRT)
    "graphdef_file": "/gpfs/jlse-fs0/users/pbalapra/tensorrt/Benchmarks/Pilot1/NT3/nt3.pb",
    "frozen_model_file": "/gpfs/jlse-fs0/users/pbalapra/tensorrt/Benchmarks/Pilot1/NT3/nt3_frozen_model.pb",
    "snapshot_dir": "/gpfs/jlse-fs0/users/pbalapra/tensorrt/Benchmarks/Pilot1/NT3/snapshot",
    "engine_save_dir": "/gpfs/jlse-fs0/users/pbalapra/tensorrt/Benchmarks/Pilot1/NT3",
    # Needed for TensorRT
    "inference_batch_size": 1,  # inference batch size
    "input_layer": "conv1d_1",  # name of the input tensor in the TF computational graph
    "out_layer": "activation_5/Softmax",  # name of the output tensorf in the TF conputational graph
    "output_size" : 2,  # number of classes in output (5)
    "precision": "fp32",  # desired precision (fp32, fp16)
    "test_image_path" : "/home/data/val/roses"
}

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
INPUT_LAYERS = [config['input_layer']]
OUTPUT_LAYERS = [config['out_layer']]
INFERENCE_BATCH_SIZE = config['inference_batch_size']

# Load your newly created Tensorflow frozen model and convert it to UFF
uff_model = uff.from_tensorflow_frozen_model(config['frozen_model_file'], OUTPUT_LAYERS)

# Create a UFF parser to parse the UFF file created from your TF Frozen model
parser = uffparser.create_uff_parser()
parser.register_input(INPUT_LAYERS[0],(1,60464,128),0)
parser.register_output(OUTPUT_LAYERS[0])

# Build your TensorRT inference engine
if(config['precision'] == 'fp32'):
    engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, INFERENCE_BATCH_SIZE, 1<<20, trt.infer.DataType.FLOAT)
elif(config['precision'] == 'fp16'):
    engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, INFERENCE_BATCH_SIZE, 1<<20, trt.infer.DataType.HALF)
    
    # Serialize TensorRT engine to a file for when you are ready to deploy your model.
save_path = str(config['engine_save_dir']) + "keras_vgg19_b" + str(INFERENCE_BATCH_SIZE) + "_"+ str(config['precision']) + ".engine"

trt.utils.write_engine_to_file(save_path, engine.serialize())
    
print("Saved TRT engine to {}".format(save_path))
