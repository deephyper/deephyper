"""
INSTALL REQUIRED:
    pip install jupyter_contrib_nbextensions
"""

import os
import nbformat as nbf

nb = nbf.v4.new_notebook()


def gmd(c): return nbf.v4.new_markdown_cell(c)


def gcode(c): return nbf.v4.new_code_cell(c)


cells = list()

# Generating notebook content
title = """\
# Deephyper Analytics
"""
cells.append(gmd(title))

c1 = """\
%pylab inline
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [1, 2, 3])
plt.show()
"""
cells.append(gcode(c1))

nb['cells'] = cells

# Writing file on disk
nbf.write(nb, 'test.ipynb')

# Executing notebook
os.popen("jupyter nbconvert --execute --inplace test.ipynb")
#os.popen("jupyter nbconvert --to notebook --execute test.ipynb")
