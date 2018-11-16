import pprint
from collections import OrderedDict

import numpy as np

# definition of a key
layer_type = 'layer_type'
features = 'features'
input_shape = 'input_shape'
output_shape = 'output_shape'
num_outputs = 'num_outputs'
num_steps = 'num_steps'
max_layers = 'max_layers'
min_layers = 'min_layers'
hyperparameters = 'hyperparameters'
summary = 'summary'
logs = 'logs'
data = 'data'
regression = 'regression'
num_features = 'num_features'
state_space = 'state_space'
model_path = 'model_path'

# hyperparameters
batch_size = 'batch_size'
learning_rate = 'learning_rate'
num_epochs = 'num_epochs'
patience = 'patience'
eval_freq = 'eval_freq'
loss_metric = 'loss_metric'
metrics = 'metrics'
test_metric = 'test_metric'
text_input = 'text_input'


# data
train_X = 'train_X'
train_Y = 'train_Y'
valid_X = 'valid_X'
valid_Y = 'valid_Y'
train_set = 'train_set'
test_X = 'test_X'
test_Y = 'test_Y'
vocabulary = 'vocabulary'

#all
skip_conn = 'skip_conn'

# conv
num_filters = 'num_filters'

# conv 1D
filter_size = 'filter_size'
stride_size = 'stride_size'
pool_size = 'pool_size'

# temp conv
dilation = 'dilation'

# conv 2D
filter_height = 'filter_height'
filter_width = 'filter_width'
stride_height = 'stride_height'
stride_width = 'stride_width'
pool_height = 'pool_height'
pool_width = 'pool_width'
padding = 'padding'

# rnn
num_units = 'num_units'
unit_type = 'unit_type'
drop_out = 'drop_out'
vocab_size = 'vocab_size'
max_grad_norm = 'max_grad_norm'


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
tempconv = 'tempconv'
rnn = 'rnn'
conv1D_params = [num_filters, filter_size, stride_size, pool_size, drop_out, padding]
conv2D_params = [num_filters, filter_height, filter_width, stride_height, stride_width,
  pool_height, pool_width, padding, activation, batch_norm, batch_norm_bef, drop_out]
dense_params = [num_outputs, drop_out, batch_norm, batch_norm_bef, activation]
tempconv_params = [num_filters, filter_size, stride_size, pool_size, drop_out, dilation]
rnn_params = [num_units, drop_out]

# definition of possible values of the key arch_type
layer_type_values = {conv1D: conv1D_params, conv2D:conv2D_params, dense: dense_params, tempconv:tempconv_params, rnn:rnn_params}


max_episodes = 'max_episodes'

class StateSpace:
    '''
    State Space manager

    Provides utilities functions for holding "states" / "actions" that the controller
    must use to train and predict.

    Also provides a more convenient way to define the search space
    '''
    def __init__(self, states=None):
        self.states = OrderedDict()
        self.state_count_ = 0
        if ( states != None ):
            for state in states:
                self.add_state(state[0], state[1])
        self.max_num_class = 0

    def add_state(self, name, values):
        '''
        Adds a "state" to the state manager, along with some metadata for efficient
        packing and unpacking of information required by the RNN Controller.

        Stores metadata such as:
        -   Global ID
        -   Name
        -   Valid Values
        -   Number of valid values possible
        -   Map from value ID to state value
        -   Map from state value to value ID

        Args:
            name: name of the state / action
            values: valid values that this state can take

        Returns:
            Global ID of the state. Can be used to refer to this state later.
        '''
        index_map = {}
        for i, val in enumerate(values):
            index_map[i] = val

        value_map = {}
        for i, val in enumerate(values):
            value_map[val] = i

        metadata = {
            'id': self.state_count_,
            'name': name,
            'values': values,
            'size': len(values),
            'index_map_': index_map,
            'value_map_': value_map,
        }

        self.max_num_class = max(self.max_num_class, metadata['size'])

        self.states[self.state_count_] = metadata
        self.state_count_ += 1

        return self.state_count_ - 1

    def feature_is_defined(self, name):
        for feature_i in range(self.state_count_):
            md = self.states[feature_i]
            if (md['name'] == name):
                return True
        return False

    def get_num_tokens(self, num_layers):
        if self.feature_is_defined('skip_conn'):
            return (self.size-1)*num_layers + (num_layers-1)*num_layers//2
        else:
            return self.size*num_layers

    def embedding_encode(self, id, value):
        '''
        Embedding index encode the specific state value

        Args:
            id: global id of the state
            value: state value

        Returns:
            embedding encoded representation of the state value
        '''
        state = self[id]
        size = state['size']
        value_map = state['value_map_']
        value_idx = value_map[value]

        one_hot = np.zeros((1, size), dtype=np.float32)
        one_hot[np.arange(1), value_idx] = value_idx + 1
        return one_hot

    def get_state_value(self, id, index):
        '''
        Retrieves the state value from the state value ID

        Args:
            id: global id of the state
            index: index of the state value (usually from argmax)

        Returns:
            The actual state value at given value index
        '''
        state = self[id]
        index_map = state['index_map_']

        if (type(index) == list or type(index) == np.ndarray) and len(index) == 1:
            index = index[0]

        value = index_map[index]
        return value

    def get_random_state_space(self, num_layers, num=1): # ok for skip_co
        '''
        Constructs a random initial state space for feeding as an initial value
        to the Controller RNN

        Args:
            num_layers: number of layers to duplicate the search space

        Returns:
            A list of one hot encoded states
        '''
        states = []

        for state_i in range(num):
            current_state = []
            for layer_n in range(num_layers):
                for feature_i in range(self.size):
                    if (self[feature_i]['name'] == 'skip_conn'):
                        for j in range(layer_n+1):
                            current_state.append(float(np.random.randint(0,2)))
                    elif (self[feature_i]['name'] == 'drop_out'):
                        current_state.append(np.random.random())
                    else:
                        feature = self[feature_i]
                        # size = feature['size']
                        # sample = np.random.randint(0, size)
                        sample = np.random.choice(feature['values'])
                        current_state.append(sample)
            states.append(current_state)

        return states

    def extends_num_layer_of_state(self, state, num_layers):
        '''
        Args:
            state: list of tokens
            num_layers: int, number of in the new extended states
        '''
        for layer_n in range(num_layers-1, num_layers):
            for feature_i in range(self.size):
                if (self[feature_i]['name'] == 'skip_conn'):
                    for j in range(layer_n+1):
                        state.append(float(np.random.randint(0,2)))
                elif (self[feature_i]['name'] == 'drop_out'):
                    state.append(np.random.random())
                else:
                    feature = self[feature_i]
                    size = feature['size']
                    sample = np.random.randint(0, size)
                    state.append(feature['values'][sample])

    def parse_state_space_list(self, state_list): # not ok for skip_co
        '''
        Parses a list of one hot encoded states to retrieve a list of state values

        Args:
            state_list: list of one hot encoded states

        Returns:
            list of state values
        '''
        state_values = []
        for id, state_one_hot in enumerate(state_list):
            state_val_idx = np.argmax(state_one_hot, axis=-1)[0]
            value = self.get_state_value(id, state_val_idx)
            state_values.append(value)

        return state_values

    def parse_state(self, state_list, num_layers): # ok for skip_co
        list_values = []
        cursor = 0
        num_classes = self.max_num_class
        for layer_n in range(num_layers):
            for feature_i in range(self.size):
                state = self[feature_i]
                if (self[feature_i]['name'] == 'skip_conn'):
                    for j in range(layer_n):
                        ratio = state_list[cursor]/num_classes
                        token = 0 if (ratio < 0.5) else 1
                        list_values.append(token)
                        cursor += 1
                else:
                    index = int(state_list[cursor]/num_classes * state['size'])
                    list_values.append(self.get_state_value(feature_i,
                                                            index))
                    cursor += 1
        return list_values

    def print_state_space(self):
        ''' Pretty print the state space '''
        print('*' * 40, 'STATE SPACE', '*' * 40)

        pp = pprint.PrettyPrinter(indent=2, width=100)
        for id, state in self.states.items():
            pp.pprint(state)
            print()

    def print_actions(self, actions):
        ''' Print the action space properly '''
        print('Actions :')

        for id, action in enumerate(actions):
            if id % self.size == 0:
                print("*" * 20, "Layer %d" % (((id + 1) // self.size) + 1), "*" * 20)

            state = self[id]
            name = state['name']
            vals = [(n, p) for n, p in zip(state['values'], *action)]
            print("%s : " % name, vals)
        print()

    def __getitem__(self, id):
        return self.states[id % self.size]

    @property
    def size(self):
        return self.state_count_

    def argmax2tokens(self, argmax_list, num_layers, num_classes):
        token_list = []
        cursor = 0
        for layer_i in range(num_layers):
            for feature_i in range(self.size):
                state = self[feature_i]
                if (state['name'] == 'skip_conn'):
                    for j in range(layer_i):
                        ratio = argmax_list[cursor]/num_classes
                        token = 0 if (ratio < 0.5) else 1
                        token_list.append(token)
                else:
                    index = int(argmax_list[cursor]/num_classes * state['size'])
                    token = self.get_state_value(feature_i, index)
                    token_list.append(token)
        return token_list

def test_random_and_extends():
    state_space = StateSpace()
    state_space.add_state('filter_height', ['FH'])
    state_space.add_state('drop_out', [])
    state_space.add_state('skip_conn', [])
    states = state_space.get_random_state_space(1, 1)
    assert len(states[0]) == 3
    print(f'num_layer = 1, state ={states[0]}')
    state_space.extends_num_layer_of_state(states[0], 2)
    assert len(states[0]) == 7
    print(f'num_layer = 2, state ={states[0]}')

def test_argmax2tokens():
    sp = StateSpace()
    sp.add_state('filter_size', [10., 20., 30.])
    sp.add_state('num_filters', [10., 20.])
    sp.add_state('skip_conn', [])
    argmax_list = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    print(f'argmax_list: {argmax_list}')
    num_layers = 4
    num_classes = 3
    token_list = sp.argmax2tokens(argmax_list, num_layers, num_classes)
    print(f'token_list: {token_list}')

if __name__ == '__main__':
    #test_random_and_extends()
    test_argmax2tokens()
