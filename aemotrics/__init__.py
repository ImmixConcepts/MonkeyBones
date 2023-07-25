try:
    from aemotrics._version import version as __version__
except ImportError:
    __version__ = "unknown"
from logging import info
import sys

try:
    import qtpy
except (ModuleNotFoundError, ImportError) as e:
    info("aemotrics gui not loaded")
if "qtpy" in sys.modules and ("PySide2" in sys.modules or "PyQt5" in sys.modules):
    from aemotrics import viewer
    from aemotrics.aemotrics import main
from aemotrics import analysis, cropper
