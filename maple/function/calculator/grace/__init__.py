__all__ = ['GRACECalculator']


def __getattr__(name):
    if name == 'GRACECalculator':
        from ._grace_calculator import GRACECalculator
        return GRACECalculator
    raise AttributeError(name)
