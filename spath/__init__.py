__all__ = ["ScanpathPredictor", "load_checkpoint"]


def __getattr__(name):
    if name in __all__:
        from spath.model import ScanpathPredictor, load_checkpoint

        return {"ScanpathPredictor": ScanpathPredictor, "load_checkpoint": load_checkpoint}[name]
    raise AttributeError(name)
