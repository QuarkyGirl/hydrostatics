project = 'hydrostatics'
copyright = '2026, Eleanor Hall'
author = 'Eleanor Hall'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_typehints = "none"

autosummary_generate = True

numpydoc_show_class_members = False
numpydoc_class_members_toctree = True
numpydoc_show_inherited_class_members = True
numpydoc_attributes_as_param_list = True

autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'undoc-members': False,
}

def skip_func(app, what, name, obj, skip, options):
    if name == "__init__":
        return True

    if what in ("attribute", "property"):
        return True
        
    if what in ("function", "method"):
        if not hasattr(obj, "__doc__") or not obj.__doc__ or not obj.__doc__.strip():
            return True

    return None

def setup(app):
    app.connect("autodoc-skip-member", skip_func)

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    'remove_init.css',
]

