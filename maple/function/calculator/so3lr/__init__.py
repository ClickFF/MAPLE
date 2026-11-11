__all__ = ['SO3LRCalculator']


def __getattr__(name):
    if name == 'SO3LRCalculator':
        from ._so3lr_calculator import SO3LRCalculator
        return SO3LRCalculator
    raise AttributeError(name)
