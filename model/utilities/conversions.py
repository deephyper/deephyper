'''
 * @Author: romain.egele, dipendhra.jha
 * @Date: 2018-06-20 15:44:33
'''
import deephyper.model.arch as a
from deephyper.search import util


logger = util.conf_logger('deephyper.model.util.conversions')

def action2dict_v2(config, action, num_layers):
    layer_type = config[a.layer_type]
    state_space = config['state_space']
    arch = {}
    assert isinstance(state_space, a.StateSpace)

    # must check that state_space features are compatible with layer type
    # must check that length of action list correspond to num_layers and state_space features

    cursor = 0
    #logger.debug(f'conversions: config: {config}')
    logger.debug(f'conversions: action: {action}')
    logger.debug(f'conversions: numlayers: {num_layers}')
    max_size = 1
    skip_conn = False
    for layer_n in range(num_layers):
        layer_name = f'layer_{layer_n+1}'
        layer_arch = {}
        layer_arch[a.layer_type] = layer_type
        #logger.debug(action)
        logger.debug(f'{cursor}, {layer_n}, {layer_name}, {layer_type}, {action[cursor]}')
        for feature_i in range(state_space.size):
            feature = state_space[feature_i]
            if feature['size'] > max_size: max_size = feature['size']
            logger.debug(f'{cursor}, {layer_n}, {layer_name}, {layer_type}, {action[cursor]}')
            logger.debug(f'{cursor}, {feature}')
            if (feature['name'] == 'skip_conn'):
                skip_conn = True
                continue
            layer_arch[feature['name']] = feature['values'][int(action[cursor])%feature['size']]
            cursor += 1
        if skip_conn:
            layer_arch['skip_conn'] = []
            for j in range(layer_n):
                logger.debug(f'skip conn  {cursor}, {action[cursor]}')
                if (int(action[cursor])%2):
                    layer_arch['skip_conn'].append(j+1)
                    cursor += 1
        arch[layer_name] = layer_arch
    logger.debug(f'architecture is: {arch}')
    return arch

def test_action2dict_v2():
    cfg = {}
    state_space = a.StateSpace()
    state_space.add_state('filter_height', [3, 5])
    state_space.add_state('filter_width', [3, 5])
    state_space.add_state('pool_height', [1, 2])
    state_space.add_state('pool_width', [1, 2])
    state_space.add_state('stride_height', [1])
    state_space.add_state('stride_width', [1])
    state_space.add_state('drop_out', [])
    state_space.add_state('num_filters', [2**i for i in range(5, 8)])
    state_space.add_state('skip_conn', [])
    action = [3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 0.5101234316825867, 64.0, 0.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 0.5100855231285095, 64.0, 0.0, 0.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 0.5108792185783386, 64.0, 0.0, 0.0, 0.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 0.5111612677574158, 64.0, 0.0, 0.0, 0.0, 0.0]
    cfg['state_space'] = state_space
    cfg['max_layers'] = 4
    cfg['layer_type'] = 'conv2D'
    print(action2dict_v2(cfg, action, cfg['max_layers']))

if __name__ == '__main__':
    test_action2dict_v2()
