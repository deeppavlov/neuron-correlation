import io
import pathlib
import sys
import types

from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell


def find_notebook(fullname, paths=None):
    """find a notebook, given its fully qualified name and an optional list of paths

    This turns "foo.bar" into "foo/bar.ipynb"
    """
    if paths is None:
        paths = [pathlib.Path('.')]
    else:
        paths = [pathlib.Path(p) for p in paths]
    fullname = fullname.split('.')
    fullname[-1] += '.ipynb'
    name = pathlib.Path(*fullname)
    for d in paths:
        nb_path = d / name
        if nb_path.is_file():
            return nb_path


class NotebookLoader(object):
    """Module Loader for Jupyter Notebooks"""
    def __init__(self, paths=None):
        self.shell = InteractiveShell.instance()
        self.paths = paths

    def load_module(self, fullname):
        """import a notebook as a module"""
        path = find_notebook(fullname, self.paths)

        print("importing Jupyter notebook from %s" % path)

        # load the notebook object
        with io.open(path, 'r', encoding='utf-8') as f:
            nb = read(f, 4)

        # create the module and add it to sys.modules
        # if name in sys.modules:
        #    return sys.modules[name]
        mod = types.ModuleType(fullname)
        mod.__file__ = path
        mod.__loader__ = self
        mod.__dict__['get_ipython'] = get_ipython
        sys.modules[fullname] = mod

        # extra work to ensure that magics that would affect the user_ns
        # actually affect the notebook module's ns
        save_user_ns = self.shell.user_ns
        self.shell.user_ns = mod.__dict__

        try:
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    # transform the input to executable Python
                    code = self.shell.input_transformer_manager.transform_cell(cell.source)
                    # run the code in the module
                    exec(code, mod.__dict__)
        finally:
            self.shell.user_ns = save_user_ns
        return mod
