import hashlib
import pickle
from collections import NamedTuple

def extension_from_parameters(param_dict):
    extension = ''
    for key in sorted(param_dict):
        if key != 'epochs':
            print ('%s: %s' % (key, param_dict[key]))
            extension += '.{}={}'.format(key,param_dict[key])
    print(extension)
    return extension

def save_meta_data(data, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_meta_data(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data
    
def resume_from_disk(benchmark_name, param_dict, data_dir=''):
    SavedModel = NamedTuple('SavedModel', ['model', 'model_path',
                                       'initial_epoch', 'model_mda_path']
                            )
    extension = extension_from_parameters(param_dict)
    hex_name = hashlib.sha224(extension.encode('utf-8')).hexdigest()
    model_name = '{}-{}.h5'.format(benchmark_name, hex_name)
    model_mda_name = '{}-{}.pkl'.format(benchmark_name, hex_name)

    model_path = os.path.join(data_dir, model_name)
    model_mda_path = os.path.join(data_dir, model_mda_name)

    initial_epoch = 0
    model = None

    if os.path.exists(model_path) and os.path.exists(model_mda_path):
        print('model and meta data exists; loading model from h5 file')
        model = load_model(model_path)
        saved_param_dict = load_meta_data(model_mda_path)
        initial_epoch = saved_param_dict['epochs']
        if initial_epoch < param_dict['epochs']:
            print(f"loading from epoch {initial_epoch}")
            print(f"running to epoch {param_dict['epochs']}")
        else:
            raise RuntimeError("Specified Epochs is less than the initial epoch; will not run")

    return SavedModel(model=model, model_path=model_path,
              model_mda_path=model_mda_path,
              inital_epoch=initial_epoch)

def fill_missing_defaults(augment_parser_fxn, param_dict):
    '''Build an augmented parser; return param_dict filled in
    with missing values that were not supplied directly'''
    def_parser = keras_cmdline.create_parser()
    def_parser = augment_parser_fxn(def_parser)
    default_params = vars(def_parser.parse_args(''))

    missing = (k for k in default_params if k not in param_dict)
    for k in missing:
        param_dict[k] = default_params[k]
    return param_dict
