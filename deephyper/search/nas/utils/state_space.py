import pprint
from collections import OrderedDict
import numpy as np

import deephyper.model.arch as a

class StateSpace:
    '''
    State Space manager

    Provides utilities functions for holding "states" / "actions" that the controller
    must use to train and predict.

    Also provides a more convenient way to define the search space
    '''
    def __init__(self, states=None, num_blocks=1):
        self.states = OrderedDict()
        self.state_count_ = 0
        if ( states != None ):
            for state in states:
                self.add_state(state[0], state[1])
        self.max_num_classes = 0
        self.num_blocks = num_blocks

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

        self.max_num_classes = max(self.max_num_classes, metadata['size'])

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

    def parse_state(self, state_list): # ok for skip_co
        list_values = []
        cursor = 0
        num_classes = self.max_num_classes
        num_layers = self.num_blocks
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

    def action2dict(self, config, action):
        layer_type = config[a.layer_type] # TODO : next should be list of types
        state_space = self
        num_layers = self.num_blocks
        arch = {}

        # must check that state_space features are compatible with layer type
        # must check that length of action list correspond to num_layers and state_space features

        cursor = 0
        #logger.debug(f'conversions: config: {config}')
        #logger.debug(f'conversions: action: {action}')
        #logger.debug(f'conversions: numlayers: {num_layers}')
        max_size = 1
        skip_conn = False
        for layer_n in range(num_layers):
            layer_name = f'layer_{layer_n+1}'
            layer_arch = {}
            layer_arch[a.layer_type] = layer_type
            #logger.debug(action)
            #logger.debug(f'{cursor}, {layer_n}, {layer_name}, {layer_type}, {action[cursor]}')
            for feature_i in range(state_space.size):
                feature = state_space[feature_i]
                if feature['size'] > max_size: max_size = feature['size']
                #logger.debug(f'{cursor}, {layer_n}, {layer_name}, {layer_type}, {action[cursor]}')
                #logger.debug(f'{cursor}, {feature}')
                if (feature['name'] == 'skip_conn'):
                    skip_conn = True
                    continue
                layer_arch[feature['name']] = feature['values'][int(action[cursor])%feature['size']]
                cursor += 1
            if skip_conn:
                layer_arch['skip_conn'] = []
                for j in range(layer_n):
                    #logger.debug(f'skip conn  {cursor}, {action[cursor]}')
                    if (int(action[cursor])%2):
                        layer_arch['skip_conn'].append(j+1)
                        cursor += 1
            arch[layer_name] = layer_arch
        #logger.debug(f'architecture is: {arch}')
        return arch
