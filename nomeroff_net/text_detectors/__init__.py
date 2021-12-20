import importlib
import pkgutil

path = pkgutil.extend_path(__path__, __name__)
for importer, modname, ispkg in pkgutil.iter_modules(path=__path__, prefix=__name__+'.'):
    importlib.import_module(modname)
