'''
 * @Author: romain.egele, dipendra.jha
 * @Date: 2018-06-20 10:23:16
'''
import os
import sys

import tensorflow as tf
import numpy as np
from pprint import pprint

HERE = os.path.dirname(os.path.abspath(__file__)) # policy dir
top  = os.path.dirname(os.path.dirname(HERE)) # search dir
top  = os.path.dirname(os.path.dirname(top)) # dir containing deephyper
sys.path.append(top)

import deephyper.model.arch as a
from deephyper.model.arch import StateSpace
from deephyper.model.utilities.conversions import action2dict_v2


class NASCellPolicy:
    def __init__(self, num_features):
        '''
            * num_feature : number of features you want to search per layer
        '''
        self.num_features = num_features

    def get(self, state, max_layers):
        with tf.name_scope('policy_network'):
            nas_cell = tf.contrib.rnn.NASCell(num_units=self.num_features*max_layers)

            outputs, state = tf.nn.dynamic_rnn(
                cell=nas_cell,
                inputs=tf.expand_dims(state, -1),
                dtype=tf.float32
                )

            bias = tf.Variable([0.05]*self.num_features*max_layers)
            outputs = tf.nn.bias_add(outputs, bias)

            # Returned last output of rnn
            return outputs[:, -1:, :]

class NASCellPolicyV2:
    def __init__(self, state_space):
        '''
            * num_feature : number of features you want to search per layer
        '''
        assert isinstance(state_space, a.StateSpace)
        self.state_space = state_space
        self.num_features = state_space.size

    def get(self, state, max_layers):
        # max_layers = number of layers for now
        with tf.name_scope('policy_network'):
            num_units = 0
            if (self.state_space.feature_is_defined('skip_conn')):
                num_units += (self.num_features-1)*max_layers
                num_units += max_layers*(max_layers+1)/2
            else:
                num_units += self.num_features*max_layers

            nas_cell = tf.contrib.rnn.NASCell(num_units)

            rnn_outputs, state = tf.nn.dynamic_rnn(
                cell=nas_cell,
                inputs=tf.expand_dims(state, -1),
                dtype=tf.float32
                )

            bias = tf.Variable([0.05]*int(num_units))
            rnn_outputs = tf.nn.bias_add(rnn_outputs, bias)

            outputs = None
            cursor = 0
            for layer_n in range(max_layers):
                for feature_i in range(self.num_features):

                    current_feature = self.state_space[feature_i]
                    if ( current_feature['name'] == 'skip_conn'):
                        # by convention-1 means input layer
                        # layer_in \in [|0, max_layers-1 |],
                        # at layer_n we can create layer_n skip_connection
                        for skip_co_i in range(layer_n+1):
                            classifier = tf.layers.dense(
                                inputs=rnn_outputs[0][cursor:cursor+1],
                                units=2,
                                name=f'classifier_skip_co_to_layer_{skip_co_i}',
                                reuse=tf.AUTO_REUSE)
                            prediction = tf.nn.softmax(classifier)
                            arg_max = tf.argmax(prediction, axis=1)
                            arg_max = tf.cast(arg_max, tf.float32)

                            if (outputs == None):
                                outputs = arg_max
                            else:
                                outputs = tf.concat(values=[outputs, arg_max], axis=0)
                            cursor += 1
                    elif ( current_feature['name'] == a.drop_out):
                        classifier = tf.layers.dense(
                            inputs=rnn_outputs[0][cursor:cursor+1],
                            units=1,
                            activation=tf.nn.sigmoid,
                            name=f'classifier_{current_feature["name"]}',
                            reuse=tf.AUTO_REUSE
                        )
                        classifier = tf.reshape(classifier, [1])

                        if (outputs == None):
                            outputs = arg_max
                        else:
                            outputs = tf.concat(values=[outputs, classifier], axis=0)
                        cursor += 1
                    else:
                        classifier = tf.layers.dense(
                            inputs=rnn_outputs[0][cursor:cursor+1],
                            units=current_feature['size'],
                            name=f'classifier_{current_feature["name"]}',
                            reuse=tf.AUTO_REUSE)
                        prediction = tf.nn.softmax(classifier)
                        arg_max = tf.argmax(prediction, axis=1)
                        arg_max = tf.cast(arg_max, tf.float32)

                        if (outputs == None):
                            outputs = arg_max
                        else:
                            outputs = tf.concat(values=[outputs, arg_max], axis=0)
                        cursor += 1

        outputs = tf.py_func(self.parse_state_tf, [outputs, max_layers], tf.float32)
        return rnn_outputs[:, -1:, :], outputs

    def parse_state_tf(self, x, num_layers):
        np_array = np.array([self.state_space.parse_state(x, num_layers)], dtype=np.float32)
        return np_array

class NASCellPolicyV3:
    def __init__(self, state_space):
        '''
            * num_feature : number of features you want to search per layer
        '''
        assert isinstance(state_space, a.StateSpace)
        self.state_space = state_space
        self.num_features = state_space.size
        self.rnn_outputs = None

    def get(self, state, max_layers):
        '''
        Args:
            state: a tensor
            max_layers: maximum number of layers
        '''
        # max_layers = number of layers for now
        with tf.name_scope('policy_network'):
            num_units = 0
            if (self.state_space.feature_is_defined('skip_conn')):
                num_units += (self.num_features-1)*max_layers
                num_units += max_layers*(max_layers+1)/2
            else:
                num_units += self.num_features*max_layers

            with tf.name_scope('RNNCell'):
                nas_cell = tf.contrib.rnn.NASCell(num_units, reuse=tf.AUTO_REUSE)

            with tf.name_scope('dynamic_rnn'):
                rnn_outputs, state = tf.nn.dynamic_rnn(
                    cell=nas_cell,
                    inputs=tf.expand_dims(state, -1),
                    dtype=tf.float32
                    )

            bias = tf.Variable([0.05]*int(num_units))
            rnn_outputs = tf.nn.bias_add(rnn_outputs, bias)
            self.rnn_outputs = rnn_outputs

            outputs = []
            cursor = 0
            with tf.name_scope('outputs'):
                for layer_n in range(max_layers):
                    for feature_i in range(self.num_features):

                        current_feature = self.state_space[feature_i]
                        if ( current_feature['name'] == 'skip_conn'):
                            # by convention-1 means input layer
                            # layer_in \in [|0, max_layers-1 |],
                            # at layer_n we can create layer_n skip_connection
                            for skip_co_i in range(layer_n+1):
                                classifier = tf.layers.dense(
                                    inputs=rnn_outputs[0][cursor:cursor+1],
                                    units=2,
                                    name=f'classifier_skip_co_to_layer_{skip_co_i}',
                                    reuse=tf.AUTO_REUSE)
                                prediction = tf.nn.softmax(classifier)
                                arg_max = tf.argmax(prediction, axis=1)
                                arg_max = tf.cast(arg_max, tf.float32)

                                outputs.append(arg_max)
                                cursor += 1
                        elif ( current_feature['name'] == a.drop_out):
                            classifier = tf.layers.dense(
                                inputs=rnn_outputs[0][cursor:cursor+1],
                                units=1,
                                activation=tf.nn.sigmoid,
                                name=f'classifier_{current_feature["name"]}',
                                reuse=tf.AUTO_REUSE
                            )
                            classifier = tf.reshape(classifier, [1])

                            outputs.append(classifier)
                            cursor += 1
                        else:
                            classifier = tf.layers.dense(
                                inputs=rnn_outputs[0][cursor:cursor+1],
                                units=current_feature['size'],
                                name=f'classifier_{current_feature["name"]}',
                                reuse=tf.AUTO_REUSE)
                            prediction = tf.nn.softmax(classifier)
                            arg_max = tf.argmax(prediction, axis=1)
                            arg_max = tf.cast(arg_max, tf.float32)

                            outputs.append(arg_max)
                            cursor += 1

        self.outputs_before_py_func = outputs
        return rnn_outputs[:, -1:, :], outputs

class NASCellPolicyV4:
    def __init__(self, state_space):
        '''
        Args:
            state_space: an Object representing the space of the tokens we want to generate
        '''
        assert isinstance(state_space, a.StateSpace)
        self.state_space = state_space
        self.num_tokens_per_layer = state_space.size

    @property
    def max_num_classes(self):
        max_num = 0
        for feature_i in range(self.num_tokens_per_layer):
            current_feature = self.state_space[feature_i]
            max_num = max(max_num, current_feature['size'])
        return max_num

    def get(self, input_rnn, max_layers, num_units=20):
        '''
        This function build the maximal graph of the controller network. If we want to compute a number of layers < max_layers we will ask the computation on a sub graph.
        Args:
            input_rnn: a tensor (placeholder), input of the first time step of our policy
                network, shape = [batch_size]
            max_layers: int, maximum number of layers
            num_units: int, state_size of all LSTMCells
        '''

        with tf.name_scope('policy_network'):
            RNNCell = lambda x: tf.contrib.rnn.LSTMCell(num_units=x, state_is_tuple=True)
            StackedCell = tf.contrib.rnn.MultiRNNCell([RNNCell(num_units),
                                                       RNNCell(num_units)],
                                                       state_is_tuple=True)
            batch_size    = tf.shape(input_rnn)[0]
            initial_state = StackedCell.zero_state(batch_size, tf.float32)

            x_t = input_rnn # input at time t
            s_t = initial_state
            output_tensors = []
            output_dense_layers = []
            for token_i in range(max_layers*self.num_tokens_per_layer):
                # o_t : final output at time t
                # h_t : hidden state at time t
                # s_t : state at time t
                h_t, s_t = StackedCell(x_t, s_t)
                with tf.name_scope(f'token_{token_i}'):
                    o_t, o_t_before_softmax = self.softmax(h_t)
                    output_dense_layers.append(o_t_before_softmax)
                    output_tensors.append(o_t)
                    x_t = tf.expand_dims(o_t, 1)

        logits = tf.concat(output_dense_layers, axis=0)
        return logits, output_tensors

    def softmax(self, h_t):
        classifier = tf.layers.dense(
            inputs=h_t,
            units=self.max_num_classes,
            name=f'softmax',
            reuse=tf.AUTO_REUSE)
        prediction = tf.nn.softmax(classifier)
        arg_max = tf.cast(tf.argmax(prediction, axis=1), tf.float32)
        return arg_max, classifier


class NASCellPolicyV5:
    def __init__(self, state_space, save_path):
        '''
        Args:
            state_space: an Object representing the space of the tokens we want to generate
        '''
        assert isinstance(state_space, a.StateSpace)
        self.state_space = state_space
        self.num_tokens_per_layer = state_space.size
        self.save_path = save_path

    @property
    def max_num_classes(self):
        max_num = 0
        for feature_i in range(self.num_tokens_per_layer):
            current_feature = self.state_space[feature_i]
            max_num = max(max_num, current_feature['size'])
        return max_num

    def save_model(self, sess):
        print('saving model to '+self.save_path)
        self.saver.save(sess, self.save_path)

    def restore_model(self, sess):
        if os.path.exists(self.save_path+'.meta'):
            print('restoring the model from '+self.save_path)
            self.saver.restore(sess, save_path=self.save_path)

    def get(self, input, max_layers, num_units=512):
        '''
        This function build the maximal graph of the controller network. If we want to compute a number of layers < max_layers we will ask the computation on a sub graph.
        Args:
            input_rnn: a tensor (placeholder), input of the first time step of our policy
                network, shape = [batch_size]
            max_layers: int, maximum number of layers
            num_units: int, state_size of all LSTMCells
        '''

        with tf.name_scope('policy_network'):
            RNNCell = lambda x: tf.contrib.rnn.LSTMCell(num_units=num_units, state_is_tuple=True)
            stacked_lstm = tf.contrib.rnn.MultiRNNCell([RNNCell(num_units),
                                                       RNNCell(num_units)],
                                                       state_is_tuple=True)
            batch_size    = input.get_shape()[0]
            initial_state = stacked_lstm.zero_state(batch_size, tf.float32)
            print('max num classes: ', self.max_num_classes)
            #x_t = input # input at time t
            state = initial_state
            token_inds = []
            input = tf.expand_dims(input,1)
            state_skip_conns = []
            states = []
            softmax_out_prob = []
            softmax_outputs = []
            with tf.variable_scope('skip_weights', reuse=tf.AUTO_REUSE):
                skip_W_prev = tf.get_variable('skip_W_prev',[num_units, num_units])
                skip_W_curr = tf.get_variable('skip_W_curr', [num_units, num_units])
                skip_v = tf.get_variable('skip_v', [num_units, self.max_num_classes])

            token_i = 0
            self.num_tokens_exc_skip = self.state_space.size * max_layers
            print(self.state_space.states)
            if self.state_space.feature_is_defined('skip_conn'):
                self.num_tokens_exc_skip = (self.state_space.size) * max_layers
            print('num_tokens exc skip: ', self.num_tokens_exc_skip, self.state_space.size, max_layers)
            for token_i in range(self.num_tokens_exc_skip):
                # o_t : final output at time t
                # h_t : hidden state at time t
                # s_t : state at time t
                #print('\n\n',token_i, ': input shape: ', input.get_shape())
                input = tf.expand_dims(input,1)
                #print(token_i,': input shape after exp: ',input.get_shape())
                outputs, state = tf.nn.dynamic_rnn(stacked_lstm, input, initial_state=state, dtype=tf.float32, time_major=False)
                #print('output shape: ', outputs.get_shape())
                #print('state shape: ', state[-1][-1].get_shape())
                outputs = tf.squeeze(outputs)
                if batch_size==1:
                    outputs = tf.reshape(outputs, shape=(1,-1))
                #print('output shape: ', outputs.get_shape(), batch_size)
                states.append(state)
                with tf.name_scope(f'token_{token_i}'):
                    softmax_output = tf.layers.dense(inputs=outputs, units=self.max_num_classes, activation=None ,reuse=tf.AUTO_REUSE, name = 'softmax')
                    softmax_outputs.append(softmax_output)
                    softmax_output = tf.nn.softmax(softmax_output)
                    softmax_out_prob.append(tf.reduce_max(softmax_output, axis=1))
                    #print('softmax output: ', softmax_output.get_shape())
                    token_ind = tf.cast(tf.argmax(softmax_output, axis=1), tf.float32)
                    #print('token ind: ', token_ind.get_shape())
                    token_inds.append(token_ind)
                    input = token_ind
                    input = tf.expand_dims(input, 1)
                    if not (token_i+1)%(self.num_tokens_per_layer-1):
                        if not self.state_space.feature_is_defined('skip_conn'): continue
                        input = tf.expand_dims(input, 1)
                        # print(token_i,': input shape after exp: ',input.get_shape())
                        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, input, initial_state=state, dtype=tf.float32,
                                                           time_major=False)
                        # print('output shape: ', outputs.get_shape())
                        # print('state shape: ', state[-1][-1].get_shape())
                        outputs = tf.squeeze(outputs)
                        if batch_size == 1:
                            outputs = tf.reshape(outputs, shape=(1, -1))
                        # print('output shape: ', outputs.get_shape(), batch_size)
                        states.append(state)

                        #skip connections
                        state_res = tf.reshape(state[-1][-1], [batch_size, num_units])
                        #print('skip conn for token: ', token_i)
                        for state_skip in state_skip_conns:
                            #print('state skip: ', state_skip.get_shape())
                            #print('state curr: ', state_res.get_shape())
                            #print('skip W prev: ', skip_W_prev.get_shape())
                            #print('skip W curr: ', skip_W_curr.get_shape())
                            sum_prod_W = tf.matmul(state_res, skip_W_curr)+tf.matmul(state_skip, skip_W_prev)
                            #print('sum_prod_W: ', sum_prod_W.get_shape())
                            tan_out = tf.nn.tanh(sum_prod_W)
                            v_tan = tf.matmul(tan_out, skip_v)
                            #print('v tan: ', v_tan.get_shape())
                            skip_conn = tf.nn.softmax(v_tan)
                            #print('skip conn adding to softmax outputs shape: ', skip_conn.get_shape())
                            softmax_outputs.append(v_tan)
                            softmax_out_prob.append(tf.reduce_max(skip_conn, axis=1))
                            #skip_conn = tf.squeeze(skip_conn)
                            #print('skipp conn: ', skip_conn.get_shape())

                            skip_conn = tf.cast(tf.argmax(skip_conn,axis=1), tf.float32)
                            #print('skipp conn: ', skip_conn.get_shape())

                            #skip_conn = tf.round(skip_conn)
                            #print('skipp conn: ', skip_conn.get_shape())
                            #skip_conn = tf.divide(skip_conn, self.max_num_classes)

                            #print('skip conn adding to token inds: ', skip_conn.get_shape())

                            #print('skipp conn: ', skip_conn.get_shape())
                            token_inds.append(skip_conn)
                        state_skip_conns.append(state_res)
                        input = token_inds[-1]
                        input = tf.expand_dims(input, 1)

        logits = tf.concat(token_inds, axis=0)
        #states = tf.concat(states, axis=0)
        softmax_out_prob = tf.convert_to_tensor(softmax_out_prob)
        softmax_outputs = tf.convert_to_tensor(softmax_outputs)
        #logits = token_inds
        print('logits shape: ', logits.get_shape(), logits)
        self.saver = tf.train.Saver()
        if not os.path.exists(self.save_path):
            print('created save path: ' + self.save_path)
            os.system('mkdir -p ' + self.save_path)
        return logits, softmax_outputs



def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print([str(i.name) for i in not_initialized_vars]) # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def test_NASCellPolicyV2():
    max_layers = 3
    sp = a.StateSpace()
    sp.add_state('filter_size', [10., 20., 30.])
    sp.add_state('drop_out', [])
    sp.add_state('num_filters', [32., 64.])
    sp.add_state('skip_conn', [])
    policy = NASCellPolicyV2(sp)
    state = [[10., 0.5, 32., 1.,
              10., 0.5, 32., 1., 1.,
              10., 0.5, 32., 1., 1., 1.]]
    print(f'hand = {state[0]}, length = {len(state[0])}')
    state = sp.get_random_state_space(max_layers)
    print(f'auto = {state[0]}, length = {len(state[0])}')
    cfg = {}
    cfg[a.layer_type] = a.conv1D
    cfg['state_space'] = sp
    pprint(action2dict_v2(cfg, state[0], max_layers))
    #state = [[10., 10.]]
    if (sp.feature_is_defined('skip_conn')):
        seq_length = (sp.size-1)*max_layers
        seq_length += max_layers*(max_layers+1)/2
    else:
        seq_length = sp.size*max_layers
    with tf.name_scope("model_inputs"):
        ph_states = tf.placeholder(
            tf.float32, [None, seq_length], name="states")
    _, net_outputs = policy.get(ph_states, max_layers)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        res1 = sess.run([net_outputs], {ph_states: state})
        print('Prediction : ')
        pprint(res1[0][0])
        pprint(action2dict_v2(cfg, res1[0][0], max_layers))

def test_NASCellPolicyV3():
    max_layers = 3

    sp = a.StateSpace()
    sp.add_state('filter_size', [10., 20., 30.])
    sp.add_state('drop_out', [])
    sp.add_state('num_filters', [32., 64.])
    sp.add_state('skip_conn', [])
    policy = NASCellPolicyV3(sp)

    # STATE with 2 layers
    state_l2 = [[10., 0.5, 32., 1.,
                 10., 0.5, 32., 1., 1.]]
    num_layers = 2
    print(f'Input = {state_l2[0]}, length = {len(state_l2[0])}')
    with tf.name_scope("model_inputs"):
        ph_states = tf.placeholder(
            tf.float32, [None, None], name="states")
    o1, net_outputs = policy.get(ph_states, max_layers)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        tf.summary.FileWriter('graph', graph=tf.get_default_graph())
        sess.run(init)
        res1 = sp.parse_state(np.array(sess.run(net_outputs[:len(state_l2[0])], {ph_states: state_l2})).flatten().tolist(), num_layers)
        print(f'Predict = {res1}, length = {len(res1)} ')

        # STATE with 3 layers
        state_l3 = [[10., 0.5, 32., 1.,
                    10., 0.5, 32., 1., 1.,
                    10., 0.5, 32., 1., 1., 1.]]
        num_layers = 3
        print(f'Input = {state_l3[0]}, length = {len(state_l3[0])}')
        res1 = sp.parse_state(np.array(sess.run(net_outputs[:len(state_l3[0])], {ph_states: state_l3})).flatten().tolist(), num_layers)
        print(f'Predict = {res1}, length = {len(res1)} ')

def test_NASCellPolicyV4():
    max_layers = 3
    sp = a.StateSpace()
    sp.add_state('filter_size', [10., 20., 30.])
    sp.add_state('num_filters', [10., 20.])
    policy = NASCellPolicyV4(sp)
    input_rnn = tf.placeholder(tf.float32, [None, 1], name="input_rnn")
    _, output_tensors = policy.get(input_rnn, max_layers)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        tf.summary.FileWriter('graph', graph=tf.get_default_graph())
        sess.run(init)
        num_layers = 1
        num_tokens = sp.size * num_layers
        res = sess.run(output_tensors[:num_tokens], {input_rnn: [[0.1]]})
        print(f'num_tokens: {num_tokens}, encoded_tokens: {np.array(res).tolist()}')
        num_layers = 2
        num_tokens = sp.size * num_layers
        res = sess.run(output_tensors[:num_tokens], {input_rnn: [[0.1]]})
        print(f'num_tokens: {num_tokens}, encoded_tokens: {np.array(res).tolist()}')
        num_layers = 3
        num_tokens = sp.size * num_layers
        res = sess.run(output_tensors[:num_tokens], {input_rnn: [[0.0], [0.1]]})
        print(f'num_tokens: {num_tokens}, encoded_tokens: {np.array(res).tolist()}')

def test_NASCellPolicyV5():
    start_num_layers = 1
    max_layers = 3
    sp = a.StateSpace()
    sp.add_state('filter_size', [10., 20., 30.])
    sp.add_state('num_filters', [10., 20.])
    sp.add_state('skip_conn',[])
    policy = NASCellPolicyV5(sp, save_path='savepoint/model')
    batch_size = 2
    init_seeds = [1.*i/batch_size for i in range(batch_size)]
    rnn_input = tf.placeholder(shape=(batch_size), dtype= tf.float32, name="input")
    logits, softmax_outputs = policy.get(rnn_input, max_layers)
    with tf.Session() as sess:
        policy.restore_model(sess=sess)
        init = tf.global_variables_initializer()
        sess.run(init)
        tf.summary.FileWriter('graph', graph=tf.get_default_graph())
        for num_layers in range(start_num_layers, max_layers+1):
            num_tokens = ((sp.size-1) * num_layers + num_layers * (num_layers-1)//2) * batch_size
            logits_, softmax_outputs_ = sess.run([logits[:num_tokens], softmax_outputs[:num_tokens//batch_size]], {rnn_input: init_seeds})
            #print(logits_, softmax_outputs_)
            print(f'num layers: {num_layers} num_tokens: {num_tokens}, encoded_tokens: {np.array(logits_).tolist()}')
            policy.save_model(sess)

if __name__ == '__main__':
    test_NASCellPolicyV5()
