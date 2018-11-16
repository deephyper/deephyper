from deephyper.benchmark.benchmark_functions import *

def griewank_():
    max_griewank = lambda v: -griewank(v)
    a = - 50
    b = 50
    minimas = lambda d : [ 0 for i in range(d)]
    return max_griewank, (a, b), minimas

def ackley_():
    '''
    Many local minimas
    global minimum = [0, 0, 0...0]
    '''
    max_ackley = lambda v: -ackley(v)
    a = -32.768
    b = 32.768
    minimas = lambda d: [0 for i in range(d)]
    return max_ackley, (a, b), minimas

def dixonprice_():
    '''
    Boal function
    global minimum = math.inf
    '''
    max_dixonprice = lambda v : -dixonprice(v)
    a = -50
    b = 50
    min_i = lambda i: 2**(-(2**i - 2)/(2**i))
    minimas = lambda d: [min_i(i) for i in range(d)]
    return max_dixonprice, (a, b), minimas

def polynome_2():
    p = lambda x: -sum([x_i**2 for x_i in x])
    a = -50
    b = 50
    minimas = lambda d: [0 for i in range(d)]
    return p, (a, b), minimas

def levy_():
    p = lambda x: -levy(x)
    a = -10
    b = 10
    minimas = lambda d: [1 for i in range(d)]
    return p, (a, b), minimas

def linear_():
    p = lambda x: sum(x)
    a = 100
    b = 1
    minimas = lambda d: [-1 for i in range(d)]
    return p, (a, b), minimas

def saddle_():
    p = lambda x: saddle(x)
    a = 50
    b = -50
    minimas = lambda d: None
    return p, (a, b), minimas
