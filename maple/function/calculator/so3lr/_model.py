"""SO3LR complete energy and pure JAX Coulomb reciprocal sums.

Reciprocal functions are adapted from fbruenig/jax-pme, commit
de9a3a9dfeb1f5b635a54a8f277c1ec3ecd2bf6d: kspace.py, mesh.py,
potentials.py and solvers.py. The four-node Lagrange assignment, FFT
normalization, Gaussian convention, self and neutralizing-background terms
are retained. Reciprocal sums use inverse-length units; forward supplies
SO3LR's electrostatic conversion to eV.
Only full three-dimensional periodicity is supported here. Upstream package
metadata declares MIT; author Marcel Langer <dev@marcel.science>. The pinned
repository contains no separate license text.
"""
import jax
import jax.numpy as jnp
import numpy as np


def forward(model, data, *, method, shape, cutoff):
    # The saved graph chooses the matching real-space electrostatic expression.
    data = {key: value for key, value in data.items() if key != 'pbc'}
    data.update(use_kspace=jnp.asarray(method != 'cutoff'), lr_cutoff=jnp.asarray(cutoff, jnp.float32))
    with jax.default_matmul_precision('highest'):
        output = model.call(data)
        energy = output['real_space_energy']
        if method != 'cutoff':
            # A single member may be unpadded or followed by its dummy graph.
            if data['cell'].shape[0] in (1, 2):
                cell = data['cell'][jnp.argmax(data['graph_mask'])]
                correction = reciprocal_energy(
                    output['partial_charges'], data['positions'], cell, shape, method=method,
                )
            else:
                charges = jnp.where(data['node_mask'], output['partial_charges'], 0.)
                correction = reciprocal_energy(
                    charges, data['positions'], data['cell'], shape, method=method,
                    batch=data['batch_segments'],
                )
            energy = energy + 14.399645351950548 * correction * data['graph_mask']
    return {**output, 'energy': energy}


def grid_shape(cell, method, spacing=2.):
    """Choose a concrete native grid before JIT/differentiation."""
    counts = np.ceil(np.linalg.norm(np.asarray(cell), axis=-1) / spacing).astype(int)
    if method == 'pme':
        counts = (2**np.ceil(np.log2(2*counts + 1))).astype(int)
    elif method != 'ewald':
        raise ValueError('reciprocal method must be ewald or pme')
    return tuple(int(x) for x in counts)


def generate_kvectors(cell, shape, *, for_ewald=True):
    reciprocal = jnp.linalg.inv(cell).T * (2*jnp.pi)
    kx = (reciprocal[0]*shape[0]) * jnp.fft.fftfreq(shape[0], dtype=cell.dtype)[:, None]
    ky = (reciprocal[1]*shape[1]) * jnp.fft.fftfreq(shape[1], dtype=cell.dtype)[:, None]
    frequency = jnp.fft.fftfreq if for_ewald else jnp.fft.rfftfreq
    kz = (reciprocal[2]*shape[2]) * frequency(shape[2], dtype=cell.dtype)[:, None]
    vectors = kx[:, None, None] + ky[None, :, None] + kz[None, None, :]
    return vectors.reshape(-1, 3) if for_ewald else vectors


def _kernel(vectors, smearing):
    k2 = (vectors*vectors).sum(axis=-1)
    safe = jnp.where(k2 == 0., 1.e-6, k2)
    value = 4*jnp.pi*jnp.exp(-.5*smearing**2*safe)/safe
    return jnp.where(k2 == 0., 0., value)


def _lagrange(positions, cell, shape, batch=None):
    counts = jnp.array(shape)
    inverse = jnp.linalg.inv(cell)
    relative = counts * (positions @ inverse if batch is None else
                         jnp.einsum('ni,nij->nj', positions, inverse[batch]))
    indices = jnp.floor(relative)
    x = relative - (indices + .5)
    x2, x3 = x*x, x*x*x
    weights = jnp.stack([
        (-3 + 2*x + 12*x2 - 8*x3)/48,
        (27 - 54*x - 12*x2 + 24*x3)/48,
        (27 + 54*x - 12*x2 - 24*x3)/48,
        (-3 - 2*x + 12*x2 + 8*x3)/48,
    ])
    neighbors = jnp.stack([(indices + i) % counts for i in (-1, 0, 1, 2)]).astype(jnp.int32)
    offsets = jnp.meshgrid(jnp.arange(4), jnp.arange(4), jnp.arange(4), indexing='ij')
    a, b, c = [value.reshape(-1) for value in offsets]
    mesh_indices = (neighbors[a, :, 0], neighbors[b, :, 1], neighbors[c, :, 2])
    mesh_weights = weights[a, :, 0]*weights[b, :, 1]*weights[c, :, 2]
    return mesh_indices, mesh_weights


def reciprocal_energy(charges, positions, cell, shape, *, method='pme', smearing=4., batch=None):
    """Complete reciprocal energy, including self/background corrections.

    ``shape`` and ``method`` are static JIT arguments. Cell remains live for
    coordinate/cell derivatives; choosing another grid is a host operation.
    With ``batch``, flat atoms use their member's cell and independent mesh,
    structure factors, self/background corrections and energy reduction.
    """
    volume = jnp.abs(jnp.linalg.det(cell))
    generate = lambda cell: generate_kvectors(cell, shape, for_ewald=method == 'ewald')
    vectors = generate(cell) if batch is None else jax.vmap(generate)(cell)
    kernel = _kernel(vectors, smearing)
    if method == 'ewald':
        arguments = vectors @ positions.T if batch is None else jnp.einsum('nki,ni->kn', vectors[batch], positions)
        cosine, sine = jnp.cos(arguments), jnp.sin(arguments)
        if batch is None:
            c = (cosine*charges).sum(axis=-1)
            s = (sine*charges).sum(axis=-1)
            potential = (kernel[:, None]*(cosine*c[:, None] + sine*s[:, None])).sum(axis=0)/volume
        else:
            c = jax.ops.segment_sum((cosine*charges).T, batch, cell.shape[0]).T
            s = jax.ops.segment_sum((sine*charges).T, batch, cell.shape[0]).T
            potential = (kernel[batch].T*(cosine*c[:, batch] + sine*s[:, batch])).sum(axis=0)/volume[batch]
    elif method == 'pme':
        indices, weights = _lagrange(positions, cell, shape, batch)
        mesh_shape = shape if batch is None else (cell.shape[0], *shape)
        if batch is not None:
            indices = (batch, *indices)
        density = jnp.zeros(mesh_shape, dtype=charges.dtype).at[indices].add(charges*weights)
        density_hat = jnp.fft.rfftn(density, norm='backward', s=shape, axes=(-3, -2, -1))
        potential_mesh = jnp.fft.irfftn(density_hat*kernel, norm='forward', s=shape, axes=(-3, -2, -1))
        potential = (potential_mesh[indices]*weights).sum(axis=0)/(volume if batch is None else volume[batch])
    else:
        raise ValueError('reciprocal method must be ewald or pme')
    potential -= charges * jnp.sqrt(2/jnp.pi)/smearing
    if batch is None:
        potential -= 2*jnp.pi*smearing**2*charges.sum()/volume
        return .5*(charges*potential).sum()
    total_charge = jax.ops.segment_sum(charges, batch, cell.shape[0])
    potential -= 2*jnp.pi*smearing**2*total_charge[batch]/volume[batch]
    return .5*jax.ops.segment_sum(charges*potential, batch, cell.shape[0])
