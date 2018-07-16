'''
 * @Author: romain.egele, dipendhra.jha
 * @Date: 2018-06-20 15:44:33
'''
import deephyper.model.arch as a

def action2dict(config, action):
    '''
      * config : a dict
      * action : a list of float
    {'num_filters': 32, 'filter_height': 3, 'filter_width': 3,
                              'stride_height': 1, 'stride_width': 1, 'pool_height': 2,
                              'pool_width': 2, 'padding': 'SAME', 'activation': 'relu',
                              'batch_norm': False, 'batch_norm_bef': True,
                              'drop_out': 1}
    '''
    layer_type = config[a.layer_type]
    features = config[a.features]
    arch = {}
    if not(layer_type in list(a.layer_type_values.keys())):
        raise RuntimeError("Can't complete conversion because type : '{0}' doesn't\
            exist. Please chose one of these types : {1}".format(type,
            list(a.layer_type_values.keys())))
    for f in features:
        if not(f in a.layer_type_values[layer_type]):
            raise RuntimeError("feature : '{0}' is not known for this layer_type :'{1}'.".format(f, layer_type))
    num_features = len(features)
    num_action = len(action)
    if ( num_action % num_features != 0):
        raise RuntimeError("'action' doesn't correspond to the good features : \n {0} .".format(action))
    num_layers = num_action // num_features
    for l_i in range(num_layers):
        layer_name = 'layer_{0}'.format(l_i)
        layer_actions = action[l_i * num_features:(l_i + 1) * num_features]
        # int(v) is here because json.dumps(...) raise TypeError: Object of type 'int32' is not JSON serializable
        layer_arch = { k:int(v) for k, v in zip(features, layer_actions) }
        # layer_type is not a feature yet
        layer_arch[a.layer_type] = layer_type
        arch[layer_name] = layer_arch
    return arch

def action2dict_v2(config, action, num_layers):
    layer_type = config[a.layer_type]
    state_space = config['state_space']
    arch = {}
    assert isinstance(state_space, a.StateSpace)

    # must check that state_space features are compatible with layer type
    # must check that length of action list correspond to num_layers and state_space features

    cursor = 0
    for layer_n in range(num_layers):
        layer_name = f'layer_{layer_n+1}'
        layer_arch = {}
        layer_arch[a.layer_type] = layer_type
        for feature_i in range(state_space.size):
            feature = state_space[feature_i]
            if (feature['name'] == 'skip_conn'):
                layer_arch[feature['name']] = []
                for j in range(layer_n+1):
                    if (action[cursor] == 1. ):
                        layer_arch[feature['name']].append(j)
                    cursor += 1
            else:
                layer_arch[feature['name']] = action[cursor]
                cursor += 1
        arch[layer_name] = layer_arch
    return arch

def test_action2dict_v1():
    cfg = {}
    cfg[a.layer_type] = 'conv1D'
    cfg[a.features] = ['num_filters']
    action = [32, 10, 5]
    print(action2dict(cfg, action))
