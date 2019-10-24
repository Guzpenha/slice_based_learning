# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import datetime
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "Snorkel"
copyright = f"{datetime.datetime.now().year}, Snorkel Team"
author = "Snorkel Team"
master_doc = "index"
html_logo = "_static/octopus.png"

VERSION = {}
with open("../snorkel/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

# The full version, including alpha/beta/rc tags
release = VERSION["VERSION"]


# -- General configuration ---------------------------------------------------

# Mock imports for troublesome modules (i.e. any that use C code)
autosummary_mock_imports = ["dask", "pyspark", "spacy"]

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.linkcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": -1, "titles_only": True}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Options for napoleon extension -------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for autodoc extension -------------------------------------------

# This value selects what content will be inserted into the main body of an autoclass
# directive
#
# http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#directive-autoclass
autoclass_content = "class"


# Default options to an ..autoXXX directive.
autodoc_default_options = {
    "members": None,
    "inherited-members": None,
    "show-inheritance": None,
    "special-members": "__call__",
}

# Subclasses should show parent classes docstrings if they don't override them.
autodoc_inherit_docstrings = True

# -- Options for linkcode extension ------------------------------------------


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    module_path = info["module"].replace(".", "/")
    # If only one `.`, assume it's a package
    if info["module"].count(".") == 1:
        return f"https://github.com/snorkel-team/snorkel/tree/master/{module_path}"
    # Otherwise, it's a module
    else:
        return f"https://github.com/snorkel-team/snorkel/blob/master/{module_path}.py"


# -- Exclude PyTorch methods -------------------------------------------------
def skip_torch_module_member(app, what, name, obj, skip, options):
    skip_torch = "Module." in str(obj) and name in dir(torch.nn.Module)
    if name == "dump_patches":  # Special handling for documented attrib
        skip_torch = True
    return skip or skip_torch


# -- Run setup ---------------------------------------------------------------
def setup(app):
    app.connect("autodoc-skip-member", skip_torch_module_member)
