__all__ = ["FeNNolCalculator"]


def __getattr__(name):
    if name == 'FeNNolCalculator':
        from ._fennol_calculator import FeNNolCalculator
        return FeNNolCalculator
    raise AttributeError(name)
