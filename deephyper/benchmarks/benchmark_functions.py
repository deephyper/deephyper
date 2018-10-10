'''
Created by Dipendra Jha (dipendra@u.northwestern.edu) on 8/9/18
'''
#!/usr/bin/env python
from __future__ import division

""" some n-dimensional test functions for optimization in Python.
    Example:

    import numpy as np
    from ... import ndtestfuncs  # this file, ndtestfuncs.py

    funcnames = "ackley ... zakharov"
    dim = 8
    x0 = e.g. np.zeros(dim)
    for func in ndtestfuncs.getfuncs( funcnames ):
        fmin, xmin = myoptimizer( func, x0 ... )
        # calls func( x ) with x a 1d numpy array or array-like of any size >= 2

These are the n-dim Matlab functions by A. Hedar (2005), translated to Python-numpy.
http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO.htm
    ackley.m dp.m griew.m levy.m mich.m perm.m powell.m power.m
    rast.m rosen.m schw.m sphere.m sum2.m trid.m zakh.m
    + ellipse nesterov powellsincos randomquad logsumexp

--------------------------------------------------------------------------------
    All functions appearing in this work are fictitious;
    any resemblance to real-world functions, living or dead, is purely coincidental.
--------------------------------------------------------------------------------

Notes
-----

Each `func( x )` works for `x` of any size >= 2.
Each starts off
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf

The values of most functions increase as O(dim) or O(dim^2),
so the convergence curves for dim 5 10 20 ... are not comparable,
and `ftol_abs` depends on func() scaling.
Better would be to scale function values to min 1, max 100 in all dimensions.
Similarly, `xtol_abs` depends on `x` scaling;
`x` should be scaled to -1 .. 1 in all dimensions.
(`ftol_rel` and `xtol_rel` can misbehave near 0; see `isclose`, abs or rel.)

Results from any optimizer depend of course on `ftol_abs xtol_abs maxeval ...`
plus hidden or derived parameters, e.g. BOBYQA rho.
Methods like Nelder-Mead that track sets of points, starting with `x0 + initstep I`,
are sensitive to `initstep`. And what are `ftol` and `xtol` for sets ?

Some functions have many local minima or saddle points (more in higher dimensions ?),
making the final fmin very sensitive to the starting x0.
Also, some have a local minimum at [0 0 ...] so starting there is boring.
*Always* look at a few points near a purported xmin.

Fun constrained problems: min f(x) over the surface of f's bounding box.


See also
--------

http://en.wikipedia.org/wiki/Test_functions_for_optimization  -- 2d plots
http://www.scipy.org/NumPy_for_Matlab_Users

nlopt/test/... runs and plots BOBYQA PRAXIS SBPLX ... on these ndtestfuncs
    from several random startpoints, in several dimensions
Nd-testfuncs-python.md
F = Funcmon(func): wrap func() to monitor and plot F.fmem F.xmem F.cost
"""
    # zillions of papers and methods for derivative-free optimization alone


#...............................................................................
import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum

try:
    from opt.testfuncs.powellsincos import Powellsincos
except ImportError:
    Powellsincos = None
try:
    from opt.testfuncs.randomquad import randomquad
    from opt.testfuncs.logsumexp import logsumexp
except ImportError:
    randomquad = logsumexp = None

__version__ = "2015-03-06 mar  denis-bz-py t-online de"  # + randomquad logsumexp


#.......................
#has only one global minima


#...............................................................................
def ackley( x, a=20, b=0.2, c=2*pi ):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = sum( x**2 )
    s2 = sum( cos( c * x ))
    return -a*exp( -b*sqrt( s1 / n )) - exp( s2 / n ) + a + exp(1)

#...............................................................................
def dixonprice( x ):  # dp.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 2, n+1 )
    x2 = 2 * x**2
    return sum( j * (x2[1:] - x[:-1]) **2 ) + (x[0] - 1) **2

#...............................................................................
def griewank( x, fr=4000 ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    s = sum( x**2 )
    p = prod( cos( x / sqrt(j) ))
    return s/fr - p + 1

#...............................................................................
def levy( x ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    z = 1 + (x - 1) / 4
    return (sin( pi * z[0] )**2
        + sum( (z[:-1] - 1)**2 * (1 + 10 * sin( pi * z[:-1] + 1 )**2 ))
        +       (z[-1] - 1)**2 * (1 + sin( 2 * pi * z[-1] )**2 ))

#...............................................................................
michalewicz_m = .5  # orig 10: ^20 => underflow

def michalewicz( x ):  # mich.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    return - sum( sin(x) * sin( j * x**2 / pi ) ** (2 * michalewicz_m) )

#...............................................................................
def perm( x, b=.5 ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    xbyj = np.fabs(x) / j
    return mean([ mean( (j**k + b) * (xbyj ** k - 1) ) **2
            for k in j/n ])
    # original overflows at n=100 --
    # return sum([ sum( (j**k + b) * ((x / j) ** k - 1) ) **2
    #       for k in j ])

#...............................................................................
def powell( x ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    n4 = ((n + 3) // 4) * 4
    if n < n4:
        x = np.append( x, np.zeros( n4 - n ))
    x = x.reshape(( 4, -1 ))  # 4 rows: x[4i-3] [4i-2] [4i-1] [4i]
    f = np.empty_like( x )
    f[0] = x[0] + 10 * x[1]
    f[1] = sqrt(5) * (x[2] - x[3])
    f[2] = (x[1] - 2 * x[2]) **2
    f[3] = sqrt(10) * (x[0] - x[3]) **2
    return sum( f**2 )

#...............................................................................
def powersum( x, b=[8,18,44,114] ):  # power.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    s = 0
    for k in range( 1, n+1 ):
        bk = b[ min( k - 1, len(b) - 1 )]  # ?
        s += (sum( x**k ) - bk) **2  # dim 10 huge, 100 overflows
    return s

#...............................................................................
def rastrigin( x ):  # rast.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 10*n + sum( x**2 - 10 * cos( 2 * pi * x ))

#...............................................................................
def roscenbrock( x ):  # rosen.m
    """ http://en.wikipedia.org/wiki/Rosenbrock_function """
        # a sum of squares, so LevMar (scipy.optimize.leastsq) is pretty good
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    return (sum( (1 - x0) **2 )
        + 100 * sum( (x1 - x0**2) **2 ))

#...............................................................................
def schwefel( x ):  # schw.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 418.9829*n - sum( x * sin( sqrt( abs( x ))))

#...............................................................................
def sphere( x ):
    x = np.asarray_chkfinite(x)
    return sum( x**2 )

#...............................................................................
def sum2( x ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    return sum( j * x**2 )

#...............................................................................
def trid( x ):
    x = np.asarray_chkfinite(x)
    return sum( (x - 1) **2 ) - sum( x[:-1] * x[1:] )

#...............................................................................
def zakharov( x ):  # zakh.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    s2 = sum( j * x ) / 2
    return sum( x**2 ) + s2**2 + s2**4

#...............................................................................
    # not in Hedar --

def ellipse( x ):
    x = np.asarray_chkfinite(x)
    return mean( (1 - x) **2 )  + 100 * mean( np.diff(x) **2 )

#...............................................................................
def nesterov( x ):
    """ Nesterov's nonsmooth Chebyshev-Rosenbrock function, Overton 2011 variant 2 """
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    return abs( 1 - x[0] ) / 4 \
        + sum( abs( x1 - 2*abs(x0) + 1 ))

#...............................................................................
def saddle( x ):
    x = np.asarray_chkfinite(x) - 1
    return np.mean( np.diff( x **2 )) \
        + .5 * np.mean( x **4 )


#-------------------------------------------------------------------------------
allfuncs = [
    ackley,
    dixonprice,
    ellipse,
    griewank,
    levy,
    michalewicz,  # min < 0
    nesterov,
    perm,
    powell,
    # powellsincos,  # many local mins
    powersum,
    rastrigin,
    roscenbrock,
    schwefel,  # many local mins
    sphere,
    saddle,
    sum2,
    trid,  # min < 0
    zakharov,
    ]

if Powellsincos is not None:  # try import
    _powellsincos = {}  # dim -> func
    def powellsincos( x ):
        x = np.asarray_chkfinite(x)
        n = len(x)
        if n not in _powellsincos:
            _powellsincos[n] = Powellsincos( dim=n )
        return _powellsincos[n]( x )

    allfuncs.append( powellsincos )

if randomquad is not None:  # try import
    allfuncs.append( randomquad )
    allfuncs.append( logsumexp )

allfuncs.sort( key = lambda f: f.__name__ )


# #...............................................................................
# allfuncnames = " ".join([ f.__name__ for f in allfuncs ])
# name_to_func = { f.__name__ : f  for f in allfuncs }
#
#     # bounds from Hedar, used for starting random_in_box too --
#     # getbounds evals ["-dim", "dim"]
# ackley._bounds       = [-15, 30]
# dixonprice._bounds   = [-10, 10]
# griewank._bounds     = [-600, 600]
# levy._bounds         = [-10, 10]
# michalewicz._bounds  = [0, pi]
# perm._bounds         = ["-dim", "dim"]  # min at [1 2 .. n]
# powell._bounds       = [-4, 5]  # min at tile [3 -1 0 1]
# powersum._bounds     = [0, "dim"]  # 4d min at [1 2 3 4]
# rastrigin._bounds    = [-5.12, 5.12]
# rosenbrock._bounds   = [-2.4, 2.4]  # wikipedia
# schwefel._bounds     = [-500, 500]
# sphere._bounds       = [-5.12, 5.12]
# sum2._bounds         = [-10, 10]
# trid._bounds         = ["-dim**2", "dim**2"]  # fmin -50 6d, -200 10d
# zakharov._bounds     = [-5, 10]
#
# ellipse._bounds      =  [-2, 2]
# logsumexp._bounds    = [-20, 20]  # ?
# nesterov._bounds     = [-2, 2]
# powellsincos._bounds = [ "-20*pi*dim", "20*pi*dim"]
# randomquad._bounds   = [-10000, 10000]
# saddle._bounds       = [-3, 3]
#
#
# #...............................................................................
# def getfuncs( names, dim=0 ):
#     """ for f in getfuncs( "a b ..." ):
#             y = f( x )
#     """
#     if names == "*":
#         return allfuncs
#     funclist = []
#     for nm in names.split():
#         if nm not in name_to_func:
#             raise ValueError( "getfuncs( \"%s\" ) not found" % names )
#         funclist.append( name_to_func[nm] )
#     return funclist
#
# def getbounds( funcname, dim ):
#     """ "ackley" or ackley -> [-15, 30] """
#     funcname = getattr( funcname, "__name__", funcname )
#     func = getfuncs( funcname )[0]
#     b = func._bounds[:]
#     if isinstance( b[0], basestring ):  b[0] = eval( b[0] )
#     if isinstance( b[1], basestring ):  b[1] = eval( b[1] )
#     return b
#
# #...............................................................................
# _minus = "dixonprice perm powersum schwefel sphere sum2 trid zakharov "  # nlopt ~ same ?
#
# def allfuncs_minus( minus=_minus ):
#     return [f for f in allfuncs
#             if f.__name__ not in minus.split() ]
#
# def funcnames_minus( minus=_minus ):
#     return " ".join([ f.__name__
#             for f in allfuncs_minus( minus=minus )])
#
#
# #-------------------------------------------------------------------------------
# if __name__ == "__main__":  # standalone test --
#     import sys
#
#     dims = [2, 4, 10]  # , 100]
#     nstep = 11  # 11: 0 .1 .2 .. 1
#     seed = 0
#
#         # to change these params in sh or ipython, run this.py  a=1  b=None  c=[3] ...
#     for arg in sys.argv[1:]:
#         exec( arg )
#
#     np.set_printoptions( threshold=20, edgeitems=5, linewidth=120, suppress=True,
#         formatter = dict( float = lambda x: "%.2g" % x ))  # float arrays %.2g
#     np.random.seed(seed)
#
#     #...........................................................................
#     for dim in dims:
#         print ("\n# ndtestfuncs dim %d  along the diagonal low .. high corner --" % dim)
#         # cmp matlab, anyone ?
#
#         for func in allfuncs:
#             lo, hi = getbounds( func, dim )
#             steps = np.linspace( lo, hi, nstep )
#
#             Y = np.array([ func( t * np.ones(dim) ) for t in steps ])
#             jmin = Y.argmin()
#             Ymin = Y[jmin]
#             print ("%-12s %dd  min %-8.3g  at %.3g \t lo hi %4.3g %4.3g \t Y-min %s" % (
#                     func.__name__, dim, Ymin, steps[jmin], lo, hi, Y - Ymin ))
#
#     # plot-ndtestfuncs.py
