# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys

import git
import sphinx_book_theme
import sphinx_gallery.gen_rst

from textwrap import indent


# -- Project information -----------------------------------------------------

project = "DeepHyper"
copyright = "2018-2024, DeepHyper Team"
author = "DeepHyper Team"

# The short X.Y version
about = {}
with open("../src/deephyper/__version__.py") as f:
    exec(f.read(), about)

version = about["__version__"]

# The full version, including alpha/beta/rc tags
if about["__version__"] == "":
    release = f'v{about["__version__"]}'
else:
    release = f'v{about["__version__"]}-{about["__version_suffix__"]}'

# PULL Tutorials
branch_name_map = {"master": "main", "stable": "main", "latest": "develop", "develop": "develop"}
if os.environ.get("READTHEDOCS"):
    doc_version = os.environ["READTHEDOCS_VERSION"]
else:
    github_repo = git.Repo(search_parent_directories=True)
    doc_version = github_repo.active_branch.name

tutorial_branch = branch_name_map.get(doc_version, "develop")
tutorials_github_link = "https://github.com/deephyper/tutorials.git"
tutorials_dest_dir = "tutorials"


def pull_tutorials(github_link, dest_dir, tutorial_branch):
    os.system(f"rm -rf {dest_dir}/")
    os.system(
        f"git clone --depth=1 --branch={tutorial_branch} {github_link} {dest_dir}"
    )
    os.system(f"rm -rf {dest_dir}/.git")


pull_tutorials(tutorials_github_link, tutorials_dest_dir, tutorial_branch)

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx_book_theme",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinx_lfs_content",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autosummary_imported_members = True

# Add any paths that contain templates here, relative to this directory.
templates_path = [
    "_templates",
    os.path.join(sphinx_book_theme.get_html_theme_path(), "components"),
]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {".rst": "restructuredtext"}

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = [
    "_build",
    "_templates",
    "Thumbs.db",
    ".DS_Store",
    "examples/**.ipynb",
    "examples/**.py",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_theme_path = [sphinx_book_theme.get_html_theme_path()]


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_logo = "_static/logo/medium.png"

html_theme_options = {
    # header settings
    "repository_url": "https://github.com/deephyper/deephyper",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "develop",
    "path_to_docs": "docs",
    "use_download_button": True,
    # sidebar settings
    "show_navbar_depth": 1,
    # "logo_only": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "deephyperdoc"

# CopyButton Settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "deephyper.tex", "deephyper Documentation", "ArgonneMCS", "manual")
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "deephyper", "deephyper Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "deephyper",
        "deephyper Documentation",
        author,
        "Automated Machine Learning Software for HPC",
        "Miscellaneous",
    )
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

branch_name_map = {"master": "stable", "stable": "stable", "latest": "latest", "develop": "latest"}
if os.environ.get("READTHEDOCS"):
    doc_version = os.environ["READTHEDOCS_VERSION"]
else:
    github_repo = git.Repo(search_parent_directories=True)
    doc_version = branch_name_map.get(github_repo.active_branch.name, "latest")

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "torch": ("https://pytorch.org/docs/main/", None),
    "ConfigSpace": ("https://automl.github.io/ConfigSpace/latest/", None),
    "deephyper": (f"https://deephyper.readthedocs.io/en/{doc_version}/", None),
}


# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# makes sphinx do a mock import of mpi4py so it’s not broken when you try to do auto-docs and import mpi4py
autodoc_mock_imports = [
    "deepxde",
    "jax",
    # "joblib",
    "matplotlib",
    "mpi4py",
    # "nbformat",
    "networkx",
    "numpyro",
    "ray",
    "redis",
    "sklearn",
    "tensorflow_probability",
    "tensorflow",
    "torch",
    "tf_keras",
    # "tqdm",
    "xgboost",
]
autosummary_mock_imports = autodoc_mock_imports + [
    "ConfigSpace",
    "deephyper.skopt.learning.gaussian_process",
    "deephyper.skopt.learning.tests",
    "deephyper.skopt.plots",
    "deephyper.test",
    "joblib",
    "scipy.optimize",
    "tqdm",
]

# Remove <BLANKLINE>
trim_doctest_flags = True

# Add custom JS/CSS
def setup(app):
    app.add_css_file("custom.css")
    app.add_js_file("custom.js")

# Sphinx Gallery
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "examples",  # path to where to save gallery generated output
    "filename_pattern": r"/plot_",
    # "filename_pattern": r"/plot_experimental_design\.py",
    "ignore_pattern": r"_util\.py",
    "remove_config_comments": True,
    "plot_gallery": True,
}

# Patch Sphinx Gallery
def parse_dropdown(rst_text):
    pattern = re.compile(
        r"# \.\. dropdown::(.*?)\n"  # Match the title
        r"(?:    :(\w+):(?: (.*?))?\n)*",  # Match optional keys and values
        re.DOTALL
    )

    match = pattern.search(rst_text)
    if match:

        title = match.group(1).strip()

        # Extract options as key-value pairs
        options = dict(re.findall(r"    :(\w+):(?: (.*))?", rst_text))

        # Remove matched lines from rst_text
        cleaned_rst_text = pattern.sub("", rst_text)
        for _ in options:
            i = cleaned_rst_text.index("\n")
            cleaned_rst_text = cleaned_rst_text[i+1:]

        return {"title": title, "options": options}, cleaned_rst_text
    
    return None, rst_text


def codestr2rst(codestr, lang="python", lineno=None):
    """Return reStructuredText code block from code string."""

    # Start by checking if there is a dropdown directive
    dropdown_config, codestr = parse_dropdown(codestr)

    if lineno is not None:
        # Sphinx only starts numbering from the first non-empty line.
        blank_lines = codestr.count("\n", 0, -len(codestr.lstrip()))
        lineno = f"   :lineno-start: {lineno + blank_lines}\n"
    else:
        lineno = ""
    # If the whole block is indented, prevent Sphinx from removing too much whitespace
    dedent = "   :dedent: 1\n"
    for line in codestr.splitlines():
        if line and not line.startswith((" ", "\t")):
            dedent = ""
            break
    code_directive = f".. code-block:: {lang}\n{dedent}{lineno}\n"
    indented_block = indent(codestr, " " * 4)
    block = code_directive + indented_block

    # Process the dropdown configuration
    if dropdown_config is not None:
        dropdown_directive = f".. dropdown:: Code"
        if len(dropdown_config["title"]) > 0:
            dropdown_directive += f" ({dropdown_config['title']})"
        dropdown_directive += "\n"

        for key, value in dropdown_config["options"].items():
            dropdown_directive += " " * 4
            if len(value) > 0:
                dropdown_directive += f":{key}: {value}"
            else:
                dropdown_directive += f":{key}:"
            dropdown_directive += "\n"

        dropdown_directive += "\n"
        code_block = indent(block, " " * 4)
        block = dropdown_directive + code_block

    return block

# Apply the patch
sphinx_gallery.gen_rst.codestr2rst = codestr2rst