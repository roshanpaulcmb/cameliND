'''cameliND -- camelid nanobody design: MD simulation + binding analysis.

The two classes are imported lazily, so neither heavy dependency is pulled in until
it is actually used:

    import cameliND as cam
    sim = cam.simulate("run/dir")                # imports OpenMM here
    a   = cam.analyze("model.pdb", "traj.dcd")   # imports MDAnalysis here

    # also works:
    from cameliND import simulate, analyze

This keeps `import cameliND` cheap and lets a simulation-only process avoid loading
MDAnalysis (and vice versa). The classes live in the `_simulate` / `_analyze`
submodules (so the lowercase class names don't collide with a same-named module);
run the pipeline from the command line with `python -m cameliND <verb>` (see
`__main__.py`: `simulate` / `restart` / `extend`).
'''
import importlib

# public class name -> submodule it lives in (imported on first access)
_LAZY = {"simulate": "_simulate", "analyze": "_analyze"}

__all__ = ["simulate", "analyze"]


def __getattr__(name):
    '''PEP 562 lazy loading: accessing a class pulls in only its submodule.'''
    if name in _LAZY:
        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
