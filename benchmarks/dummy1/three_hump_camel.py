import argparse

def run(param_dict):
    x = param_dict['x']
    y = param_dict['y']
    penalty = param_dict['penalty']

    result = 2*x**2 - 1.05*x**4 + (1./6.)*x**6 + x*y + y**2
    if penalty == 'yes': 
        result += 2.0

    print("OUTPUT: ", result)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, default=2.5)
    parser.add_argument('--y', type=float, default=-3.1)
    parser.add_argument('--penalty', type=str, default='no')
    param_dict = vars(parser.parse_args())
    run(param_dict)
