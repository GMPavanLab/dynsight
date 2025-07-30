# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sphinx.application import Sphinx

project = "dynsight"
project_copyright = "2023, Andrew Tarzia"
author = "Andrew Tarzia"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
]

# Check if SOAPify is available
SOAPIFY_AVAILABLE = importlib.util.find_spec("SOAPify") is not None

# Manually control autodoc
if not SOAPIFY_AVAILABLE:
    autodoc_mock_imports = ["SOAPify"]

autosummary_imported_members = True

autodoc_typehints = "description"
autodoc_member_order = "groupwise"
autoclass_content = "both"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}


templates_path = ["_templates"]
exclude_patterns: list[str] = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]


def setup(app: Sphinx) -> None:
    """Configure the Sphinx app by adding a custom CSS file."""
    app.add_css_file("style.css")
