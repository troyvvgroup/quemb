# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "QuEmb"
copyright = "2024, Oinam Romesh Meitei"
author = "Oinam Romesh Meitei"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    # https://github.com/tox-dev/sphinx-autodoc-typehints
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
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
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "pyscf": ("https://pyscf.org/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

nitpicky = True
nitpick_ignore = []

# taken from https://stackoverflow.com/questions/11417221/sphinx-autodoc-gives-warning-pyclass-reference-target-not-found-type-warning
for line in open("nitpick-exceptions"):
    if not line.strip() or line.startswith("#"):
        continue
    dtype, target = line.split(None, 1)
    target = target.strip()
    nitpick_ignore.append((dtype, target))
