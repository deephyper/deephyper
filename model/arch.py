'''
 * @Author: romain.egele, dipendra.jha
 * @Date: 2018-06-20 16:05:18
'''

import numpy as np
import pprint
from collections import OrderedDict

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

class StateSpace:
    '''
    State Space manager

    Provides utilities functions for holding "states" / "actions" that the controller
    must use to train and predict.

    Also provides a more convenient way to define the search space
    '''
    def __init__(self):
        self.states = OrderedDict()
        self.state_count_ = 0

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
        self.states[self.state_count_] = metadata
        self.state_count_ += 1

        return self.state_count_ - 1

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

    def get_random_state_space(self, num_layers):
        '''
        Constructs a random initial state space for feeding as an initial value
        to the Controller RNN

        Args:
            num_layers: number of layers to duplicate the search space

        Returns:
            A list of one hot encoded states
        '''
        states = []

        for id in range(self.size * num_layers):
            state = self[id]
            size = state['size']

            sample = np.random.choice(size, size=1)
            sample = state['index_map_'][sample[0]]
            state = self.embedding_encode(id, sample)
            states.append(state)
        return states

    def parse_state_space_list(self, state_list):
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

if __name__ == '__main__':
    """ EXAMPLE OUTPUT :
        space size = 1
        space["filter_size"][0] = a
        **************************************** STATE SPACE ****************************************
        { 'id': 0,
        'index_map_': {0: 'a', 1: 'b', 2: 'c'},
        'name': 'filter_size',
        'size': 3,
        'value_map_': {'a': 0, 'b': 1, 'c': 2},
        'values': ['a', 'b', 'c']}

        random state space = [array([[ 0.,  0.,  3.]], dtype=float32), array([[ 1.,  0.,  0.]], dtype=float32)]
        Actions :
        ******************** Layer 2 ********************
        filter_size :  [('a', 0.0), ('b', 0.0), ('c', 3.0)]
        ******************** Layer 3 ********************
        filter_size :  [('a', 1.0), ('b', 0.0), ('c', 0.0)]

        state values = ['c', 'a']
        embedding "a" = [[ 1.  0.  0.]]
    """
    sp = StateSpace()
    sp.add_state('filter_size', ['a', 'b', 'c'])
    print(f'space size = {sp.size}')
    print(f'space["filter_size"][0] = {sp.get_state_value(0, 0)}')
    sp.print_state_space()
    state_space = sp.get_random_state_space(2)
    print(f'random state space = {state_space}')
    sp.print_actions(state_space)
    print()
    print(f'state values = {sp.parse_state_space_list(state_space)}')
    print(f'embedding "a" = {sp.embedding_encode(0, "a")}')
