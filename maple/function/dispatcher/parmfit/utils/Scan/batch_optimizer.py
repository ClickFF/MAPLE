"""Drive ParmFit optimizers with joint evaluations of their pending geometries."""

from __future__ import annotations

import numpy as np

from maple.function.calculator._batch import energy_forces_images

from .optimizer import project_forces


def _initialize(batch, optimizers, batch_size):
    batch.optimizers = list(optimizers)
    batch.batch_size = batch_size
    batch.active = []
    count = len(batch.optimizers)
    batch.energies = np.zeros(count)
    batch.forces = [None] * count
    batch.iterations = np.zeros(count, dtype=int)
    batch.converged = np.zeros(count, dtype=bool)
    batch._steps = [None] * count
    batch._requests = [None] * count
    for index, optimizer in enumerate(batch.optimizers):
        _replace(batch, index, optimizer)


def _replace(batch, index, optimizer):
    batch.optimizers[index] = optimizer
    batch.iterations[index] = 0
    batch.converged[index] = False
    batch._steps[index] = optimizer._iterate(batch=True)
    batch._requests[index] = next(batch._steps[index])
    batch.active.append(index)
    batch.active.sort()


def _step(batch):
    indices = batch.active.copy()
    if not indices:
        return []
    requests = [batch._requests[index] for index in indices]
    energies, forces = energy_forces_images(
        [atoms for atoms, _, _ in requests], batch_size=batch.batch_size,
    )
    finished = []
    for index, request, energy, force in zip(indices, requests, energies, forces):
        atoms, constraints, use_projection = request
        if use_projection and constraints.has_any:
            force = project_forces(atoms.get_positions(), force, constraints)
        batch.energies[index] = energy
        batch.forces[index] = force
        try:
            batch._requests[index] = batch._steps[index].send((energy, force))
        except StopIteration as done:
            _, energy, force, iteration, converged = done.value
            batch.energies[index] = energy
            batch.forces[index] = force
            batch.iterations[index] = iteration
            batch.converged[index] = converged
            batch.active.remove(index)
            finished.append(index)
    return finished


def _close(batch):
    for steps in batch._steps:
        if steps is not None:
            steps.close()


def _run(batch):
    try:
        while batch.active:
            batch.step()
        return [optimizer.atoms for optimizer in batch.optimizers]
    finally:
        batch.close()


class BatchLBFGS:
    def __init__(self, optimizers, batch_size=None):
        _initialize(self, optimizers, batch_size)

    def step(self):
        return _step(self)

    def replace(self, index, optimizer):
        _replace(self, index, optimizer)

    def run(self):
        return _run(self)

    def close(self):
        _close(self)


class BatchCGBS:
    def __init__(self, optimizers, batch_size=None):
        _initialize(self, optimizers, batch_size)

    def step(self):
        return _step(self)

    def replace(self, index, optimizer):
        _replace(self, index, optimizer)

    def run(self):
        return _run(self)

    def close(self):
        _close(self)


class BatchCGWS:
    def __init__(self, optimizers, batch_size=None):
        _initialize(self, optimizers, batch_size)

    def step(self):
        return _step(self)

    def replace(self, index, optimizer):
        _replace(self, index, optimizer)

    def run(self):
        return _run(self)

    def close(self):
        _close(self)
