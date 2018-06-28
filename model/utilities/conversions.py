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
        layer_arch = { k:v for k, v in zip(features, layer_actions) }
        # layer_type is not a feature yet
        layer_arch[a.layer_type] = layer_type
        arch[layer_name] = layer_arch
    return arch

def test():
    cfg = {}
    cfg[a.layer_type] = 'conv1D'
    cfg[a.features] = ['num_filters']
    action = [32, 10, 5]
    print(action2dict(cfg, action))
