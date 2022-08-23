# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'Bayes-DIP'
copyright = '2022, Riccardo Barbano, Johannes Leuschner, Javier Antorán'
author = 'Riccardo Barbano, Johannes Leuschner, Javier Antorán'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'numpydoc',
    'sphinx_rtd_theme',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# One could use ``autoclass_content = 'both'`` to merge class and __init__
# docstrings and show them directly after the class name and signature, but with
# the numpydoc extension and ``numpydoc_show_class_members = True`` the summary
# tables are duplicated then. Since we want the summary tables, we resort to
# include __init__ like a regular method and document the parameters there,
# while only including the class docstring at the beginning.

# In any case the class docstring should contain the class-related documentation
# and the __init__ docstring should just document the initialization, i.e., it
# may immediately start with the `Parameters` section.


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)

# autoclass_content = 'both'  # either


# numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_members = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
