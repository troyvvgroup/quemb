# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path

# Get the directory where conf.py is located
CONF_DIR = Path(__file__).parent


project = "QuEmb"
copyright = "2024, Van Voorhis Group"
author = "Van Voorhis Group"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

rst_prolog = """
.. role:: python(code)
    :language: python
    :class: highlight

.. role:: bash(code)
   :language: bash
   :class: highlight
"""

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    # https://github.com/tox-dev/sphinx-autodoc-typehints
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx_multiversion",
    "sphinxcontrib.bibtex",
]

autodoc_typehints_format = "short"
autodoc_default_flags = [
    "members",
    "special-members",
    "private-members",
    "undoc-members",
]
always_use_bars_union = True
python_use_unqualified_type_names = True

napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_numpy_docstring = True
napoleon_use_param = True

templates_path = ["_templates"]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "pyscf": ("https://pyscf.org/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "chemcoord": ("https://chemcoord.readthedocs.io/en/latest/", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "furo"

nitpicky = True
nitpick_ignore = []

# taken from https://stackoverflow.com/questions/11417221/sphinx-autodoc-gives-warning-pyclass-reference-target-not-found-type-warning
for line in open("nitpick-exceptions"):
    if not line.strip() or line.startswith("#"):
        continue
    dtype, target = line.split(None, 1)
    target = target.strip()
    nitpick_ignore.append((dtype, target))

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/versions.html",
        "sidebar/scroll-end.html",
    ]
}

# -- Sphinx Multiversion --------------------------------------------------
# https://holzhaus.github.io/sphinx-multiversion/master/configuration.html#
smv_tag_whitelist = r"^v\d+\.\d+\.\d+(-[a-zA-Z0-9\.]+)?$"
smv_branch_whitelist = r"^main$"
smv_remote_whitelist = r"^.*$"


bibtex_bibfiles = [str(CONF_DIR / "literature.bib")]
