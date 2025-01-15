# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# ruff: noqa: F401, E402
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from pastas_plugins import __version__, responses

# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pastas_plugins"
copyright = "2024, D.A. Brakenhoff, M.A. Vonk & M. Bakker"
author = "D.A. Brakenhoff, M.A. Vonk & M. Bakker"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",  # lowercase didn't work
    "numpydoc",
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.autosectionlabel",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

add_function_parentheses = False
add_module_names = False
show_authors = False  # section and module author directives will not be shown
todo_include_todos = False  # Do not show TODOs in docs


# -- Options for HTML output ----------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": False,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_context = {
    "github_user": "pastas",
    "github_repo": "pastas",
    "github_version": "master",
    "doc_path": "doc",
}

html_sidebars = {
    "map": [],  # Test what page looks like with no sidebar items
}

# -- Napoleon settings ----------------------------------------------------------------
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_type_aliases = {
    "ps": "pastas",
    "ml": "pastas.model.Model",
}

# -- Autodoc, autosummary, and autosectionlabel settings ------------------------------

autosummary_generate = True
autosectionlabel_prefix_document = True
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autoclass_content = "class"

# -- Numpydoc settings ----------------------------------------------------------------

numpydoc_class_members_toctree = True
numpydoc_show_class_members = False
