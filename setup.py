from distutils.core import setup


# pip3 install -U Sphinx
# pip3 install sphinx_rtd_theme
install_requires = [
    'scikit-optimize',
    'scikit-learn',
    'tqdm',
    'tensorflow',
    'keras',
    'deap', # GA search
    'numpy'
]

extras_require = {
    'tf': ['tensorflow>=1.11.0'],
}

setup(
    name='deephyper',
    version='0.0.3',
    packages=['deephyper',],
    license=open('LICENSE.md').read(),
    long_description=open('README.md').read(),
    install_requires=install_requires,
    extras_require=extras_require,
)
