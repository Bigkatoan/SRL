import os
import sys

import sphinx_book_theme

sys.path.insert(0, os.path.abspath(".."))


project = "SRL"
copyright = "2026, SRL Contributors"
author = "SRL Contributors"

extensions = [
  "sphinx.ext.autodoc",
  "sphinx.ext.autosummary",
  "myst_parser",
  "sphinx.ext.napoleon",
  "sphinx.ext.intersphinx",
  "sphinx.ext.todo",
  "sphinx.ext.viewcode",
  "sphinx_copybutton",
  "sphinx_design",
  "sphinx_tabs.tabs",
]

source_suffix = {
  ".rst": "restructuredtext",
  ".md": "markdown",
}

myst_enable_extensions = [
  "colon_fence",
  "deflist",
]
myst_heading_anchors = 3

autodoc_typehints = "signature"
autoclass_content = "class"
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = True
autosummary_generate = True

autodoc_mock_imports = [
  "isaaclab",
  "isaaclab_tasks",
  "isaacsim",
  "carb",
  "omni",
  "mjlab",
  "mujoco",
  "mujoco_warp",
  "viser",
  "warp",
  "racecar_gym",
  "gymnasium_robotics",
  "pygame",
  "box2d",
  "rclpy",
]

intersphinx_mapping = {
  "python": ("https://docs.python.org/3", None),
}

exclude_patterns = [
  "_build",
  "Thumbs.db",
  ".DS_Store",
]

language = "en"

html_title = "SRL Documentation"
html_theme_path = [sphinx_book_theme.get_html_theme_path()]
html_theme = "sphinx_book_theme"
html_show_copyright = True
html_show_sphinx = False
html_last_updated_fmt = ""

html_static_path = ["source/_static"]
html_css_files = ["css/custom.css"]

html_theme_options = {
  "path_to_docs": "docs/",
  "repository_url": "https://github.com/Bigkatoan/SRL",
  "use_repository_button": True,
  "use_issues_button": True,
  "use_edit_page_button": True,
  "show_toc_level": 2,
  "use_sidenotes": True,
  "collapse_navigation": True,
  "logo": {
    "text": "SRL Documentation",
  },
}

templates_path = []
