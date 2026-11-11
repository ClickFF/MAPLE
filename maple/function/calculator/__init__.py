"""Lazy public imports also allow independent JAX runtime loading."""

__all__ = ['CalcABC', 'register_calculator', 'get_registered_calculator',
           'import_calculator_plugin', 'load_calculator_plugins_from_env',
           'SetCalculator', 'SetClaculator']


def __getattr__(name):
    if name in {'SetCalculator', 'SetClaculator'}:
        from . import set_calculator
        return getattr(set_calculator, name)
    if name in __all__:
        from . import calculator_base
        return getattr(calculator_base, name)
    raise AttributeError(name)
