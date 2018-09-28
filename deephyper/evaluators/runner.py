import importlib
import sys
import json

def load_module(name, path):
    try:
        module = importlib.import_module(name)
    except ImportError:
        sys.path.append(path)
        module = importlib.import_module(name)
    return module

if __name__ == "__main__":
    path = sys.argv[1]
    moduleName = sys.argv[2]
    funcName = sys.argv[3]
    args = sys.argv[4]
    d = json.loads(args)
    module = load_module(moduleName, path)
    func = getattr(module, funcName)
    retval = func(d)
    print("OUTPUT:", retval)
