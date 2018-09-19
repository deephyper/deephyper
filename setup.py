from distutils.core import setup

extras_require = {
    'tf': ['tensorflow>=1.10.0'],
    'tf_gpu': ['tensorflow-gpu>=1.10.0'],
}

setup(
    name='Deephyper',
    version='0.1dev',
    packages=['deephyper',],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    extras_require=extras_require,
)
