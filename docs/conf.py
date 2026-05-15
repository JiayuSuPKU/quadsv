# Configuration file for the Sphinx documentation builder.
from datetime import datetime
from importlib.metadata import PackageNotFoundError, metadata
from pathlib import Path

try:
    # Python 3.11+
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

# -- Project information -----------------------------------------------------


def _load_project_info() -> tuple[str, str, str]:
    """Load (name, author, version) from installed metadata or pyproject.toml."""
    try:
        info = metadata("quadsv")
        return info["Name"], info["Author"], info["Version"]
    except PackageNotFoundError:
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            pyproject = tomllib.load(f)
        project_cfg = pyproject.get("project", {})
        name = project_cfg.get("name", "quadsv")
        version = project_cfg.get("version", "0.0.0")
        authors = project_cfg.get("authors", [])
        author = authors[0].get("name", "") if authors and isinstance(authors[0], dict) else ""
        return name, author, version


project_name, author, version = _load_project_info()
project = project_name
copyright = f"{datetime.now():%Y}, {author}"
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_design",  # provides ``.. dropdown::`` directives
]

# MyST configuration
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]

# AutoAPI configuration
autoapi_dirs = ["../src/quadsv"]
autoapi_add_toctree_entry = False
autoapi_python_class_content = "class"
autoapi_ignore = ["**/.ipynb_checkpoints/*", "**/*-checkpoint.py"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_member_order = "groupwise"

# Napoleon configuration
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
# Render "Attributes" sections as :ivar: fields rather than separate
# .. attribute:: directives. This avoids duplicate object description warnings
# from autoapi which already emits its own .. py:attribute:: entries.
napoleon_use_ivar = True

# Autodoc options
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "show-inheritance": True,
}
# Treat single-backtick interpreted text as literal to avoid accidental
# ambiguous cross-references from docstring tokens like `n` or `n_factors`.
default_role = "literal"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

# Autosectionlabel configuration
autosectionlabel_prefix_document = True

# Source file patterns
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "spatialdata": ("https://spatialdata.scverse.org/en/latest/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}
intersphinx_disabled_domains = ["std"]

# LaTeX options for math rendering (MathJax 4)
mathjax4_config = {
    "tex": {
        "inlinemath": [["$", "$"], ["\\(", "\\)"]],
        "displaymath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

# -- Options for HTML output --------------------------------------------------

html_theme = "sphinx_book_theme"
html_theme_options = {
    "logo": {
        "text": "quadsv",
    },
    "search_bar_text": "Search...",
    "show_toc_level": 4,
    "navigation_depth": 4,
    "repository_url": "https://github.com/JiayuSuPKU/EquivSVT",
    "use_repository_button": True,
}

# Custom static assets: only register the directory if it actually exists,
# so a fresh CI checkout (where ``docs/_static/`` is gitignored) doesn't
# emit a ``html_static_path entry '_static' does not exist`` warning that
# the ``-W`` flag turns into a fatal error.
import os as _os

if _os.path.isdir(_os.path.join(_os.path.dirname(__file__), "_static")):
    html_static_path = ["_static"]
else:
    html_static_path = []

# -- Suppress certain warnings ------------------------------------------------

suppress_warnings = ["ref.citation", "autosectionlabel.*", "duplicate_object"]
