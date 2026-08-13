# Backends like extra.cuda are copied in during Triton setup.py and then
# discovered with pkgutil. That is invisible to Pylance, so import them
# statically when the packages exist on disk.
from . import libdevice as libdevice

try:
    from . import cuda as cuda
except ImportError:
    pass

try:
    from . import hip as hip
except ImportError:
    pass

import pkgutil
from importlib.util import module_from_spec
from sys import modules

_backends = ["libdevice"]
if "cuda" in globals():
    _backends.append("cuda")
if "hip" in globals():
    _backends.append("hip")

for module_finder, module_name, is_pkg in pkgutil.iter_modules(
        __path__,
        prefix=__name__ + ".",
):
    if not is_pkg:
        continue
    short = module_name.rsplit(".", 1)[-1]
    if short in ("cuda", "hip"):
        continue
    spec = module_finder.find_spec(module_name)
    if spec is None or spec.loader is None:
        continue
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    _backends.append(module_name)
    modules[module_name] = module

__all__ = _backends

del _backends
