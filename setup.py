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
    # nas
    'gym',
    'networkx'
]

extras_require = {
    'tf': ['tensorflow>=1.11.0'],
}

setup(
    name='Deephyper',
    version='0.0.4',
    packages=['deephyper',],
    license='',
    long_description=open('README.md').read(),
    install_requires=install_requires,
    extras_require=extras_require,
)
