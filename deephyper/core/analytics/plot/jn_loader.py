"""
INSTALL REQUIRED:
    pip install jupyter_contrib_nbextensions
"""

import os
import nbformat as nbf


class NbEdit:
    def __init__(self, path_to_jnb, path_to_save="dh-analytics.ipynb"):
        with open(path_to_jnb) as fp:
            self.nb = nbf.read(fp, 4)
        self.path_to_save = path_to_save

    def setkernel(self, name):
        self.nb['metadata']['kernelspec'] = {
                    "name": name,
                    "display_name": f"Python {name}",
                    "language": "python"
                  }

    def edit(self, n_cell, old, new):
        self.nb['cells'][n_cell]['source'] = self.nb['cells'][n_cell]['source'].replace(
            old, new)

    def write(self):
        """Write jupyter notebook (.ipynb) file on disk.
        """
        nbf.write(self.nb, self.path_to_save)

    def execute(self):
        """Execute jupyter notebook to generate plots for instance.
        """
        #os.popen(f"jupyter nbconvert --execute --inplace {self.path_to_save}")
