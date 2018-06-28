'''
 * @Author: romain.egele, dipendra.jha
 * @Date: 2018-06-20 16:05:18
'''
# definition of a key
layer_type = 'layer_type'

features = 'features'
input_shape = 'input_shape'
num_outputs = 'num_outputs'
max_layers = 'max_layers'
hyperparameters = 'hyperparameters'
summary = 'summary'
logs = 'logs'
data = 'data'

# hyperparameters
batch_size = 'batch_size'
learning_rate = 'learning_rate'
num_epochs = 'num_epochs'
patience = 'patience'
eval_freq = 'eval_freq'
loss_metric = 'loss_metric'
test_metric = 'test_metric'

# data
train_X = 'train_X'
train_Y = 'train_Y'
valid_X = 'valid_X'
valid_Y = 'valid_Y'
train_set = 'train_set'

# conv
num_filters = 'num_filters'

# conv 1D
filter_size = 'filter_size'
stride_size = 'stride_size'
pool_size = 'pool_size'

# conv 2D
filter_height = 'filter_height'
filter_width = 'filter_width'
stride_height = 'stride_height'
stride_width = 'stride_width'
pool_height = 'pool_height'
pool_width = 'pool_width'
padding = 'padding'

# activation function
relu = 'relu'
tanh = 'tanh'
sigmoid = 'sigmoid'

# optimizers
adam = 'adam'
sgd = 'sgd'
rmsprop = 'rmsprop'
adagrad = 'adagrad'

optimizer = 'optimizer'
activation = 'activation'
batch_norm = 'batch_norm'
batch_norm_bef = 'batch_norm_bef'
drop_out = 'drop_out'

conv1D = 'conv1D'
conv2D = 'conv2D'
dense = 'dense'
conv1D_params = [num_filters, filter_size, stride_size, pool_size, drop_out, padding]
conv2D_params = [num_filters, filter_height, filter_width, stride_height, stride_width,
  pool_height, pool_width, padding, activation, batch_norm, batch_norm_bef, drop_out]
dense_params = [num_outputs, drop_out, batch_norm, batch_norm_bef, activation]

# definition of possible values of the key arch_type
layer_type_values = {conv1D: conv1D_params, conv2D:conv2D_params, dense: dense_params}

max_episodes = 'max_episodes'
