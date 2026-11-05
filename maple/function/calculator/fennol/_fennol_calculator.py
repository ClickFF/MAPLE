from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import _inputs
from ..calculator_base import CalcABC, _convert_properties, register_calculator


_FENNOL_MODEL_FILES = {
    'fennix-bio-1s': 'fennix-bio-1s.jax',
    'fennix-bio-1m': 'fennix-bio-1m.jax',
}


@register_calculator
class FeNNolCalculator(CalcABC):
    implemented_properties = ["energy", "forces", "free_energy", "hessian", "stress", "virial"]

    MODEL_NAMES = (
        'fennix-bio-1s', 'fennix-bio-1m',
    )
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ("analytic", "numerical")
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = True
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    LOCAL_MODEL_FILENAMES = _FENNOL_MODEL_FILES
    OPTION_KEYS = ("dtype",)

    @staticmethod
    def model_filename(model: str) -> str:
        key = str(model).strip().lower()
        try:
            return _FENNOL_MODEL_FILES[key]
        except KeyError as exc:
            raise ValueError(f"Unknown built-in FeNNol model name: {model!r}") from exc

    @classmethod
    def build_kwargs_from_options(cls, model, model_options, *, resolved_model_path=None):
        return {"dtype": model_options.get("dtype")}

    def __init__(
        self,
        device=None,
        model: str = "fennix-bio-1s",
        implicit: str = "none",
        solvent: str = "none",
        dtype=None,
    ):
        super().__init__()
        self.device = device
        self.model_name = model
        if model not in self.MODEL_NAMES:
            raise ValueError(f"Unknown FENNIX model: {model!r}")
        self.model_path = str(Path(__file__).resolve().parents[1] / 'model' / self.model_filename(model))
        if Path(self.model_path).suffix != '.jax':
            raise ValueError('FENNIX builtins require a standard .jax export; .fnx checkpoints are build inputs')
        self._load_export(dtype=dtype)
        # Qualified by the saved S/M exports' native-reference tests.
        self.hessian = "analytic"
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _load_export(self, dtype=None):
        from .. import _jax
        from ._jax_inputs import GraphGenerator, GraphFilter, BlockIndexer
        import jax

        self.model, self.jax_device = _jax.load_model(self.model_path, self.device, require_vjp=True)
        template = self.model.in_tree.unflatten(self.model.in_avals)[0][0]
        self.dtype = template['coordinates'].dtype
        if dtype is not None and str(dtype) != str(self.dtype):
            raise ValueError(f'Export supports dtype={self.dtype}, requested {dtype}')
        self.block_names = tuple(template['block_index'])

        self.metadata = _jax.read_metadata(self.model, family='fennix', model_id=self.model_name)
        if self.metadata['dtype'] != str(self.dtype):
            raise ValueError('FENNIX metadata dtype conflicts with its input signature')

        config = self.metadata['preprocessing']
        if not isinstance(config, dict) or set(config) != {'graph', 'graph_embed', 'block_indexer'}:
            raise ValueError('FENNIX metadata requires its native preprocessing configuration')
        if not all(isinstance(config[key], dict) for key in config):
            raise ValueError('Invalid FENNIX preprocessing settings')
        if (set(config['graph']) != {'cutoff', 'graph_key'}
                or set(config['graph_embed']) != {'cutoff', 'parent_graph', 'graph_key'}
                or set(config['block_indexer']) != {'split_CNOPSSe'}):
            raise ValueError('Unsupported FENNIX preprocessing fields')
        if (config['graph']['graph_key'] != 'graph'
                or config['graph_embed']['parent_graph'] != 'graph'
                or config['graph_embed']['graph_key'] != 'graph_embed'
                or type(config['block_indexer']['split_CNOPSSe']) is not bool):
            raise ValueError('Unsupported FENNIX preprocessing layout')
        for graph in ('graph', 'graph_embed'):
            cutoff = config[graph]['cutoff']
            if (isinstance(cutoff, bool) or not isinstance(cutoff, (float, int))
                    or not np.isfinite(cutoff) or cutoff <= 0):
                raise ValueError(f'FENNIX {graph} cutoff must be finite and positive')

        self.atomic_numbers = np.asarray(self.metadata['atomic_numbers'])
        self.generator = GraphGenerator(**config['graph'])
        self.filter = GraphFilter(**config['graph_embed'])
        self.indexer = BlockIndexer(**config['block_indexer'])
        self.states = (self.generator.init(), self.filter.init(), self.indexer.init())

        self._call = jax.jit(self.model.call)
        self._energy_call = jax.jit(lambda data: self.model.call(data)['total_energy'])

        self._hvp = _jax.compile_hvp(self.model.call, position_key='coordinates', force_key='forces')
        self._strain_result = _jax.compile_strain(
            self.model.call, position_key='coordinates', cell_key='cells',
            energy_key='total_energy', batch_key='batch_index',
        )

    @staticmethod
    def _validate_multiplicity(atoms) -> None:
        info = getattr(atoms, 'info', {})
        mult = info.get('mult', info.get('multiplicity'))
        if mult is None:
            return
        if not np.isfinite(float(mult)) or float(mult) != 1.:
            raise ValueError(
                "FENNIX BIO1 accepts total charge but has no spin/multiplicity "
                "input; use mult=1."
            )

    def _build_inputs(self, atoms_list, total_charge=None):
        import jax

        members = [atoms_list] if hasattr(atoms_list, 'positions') else list(atoms_list)
        _inputs._validate_geometry(members)
        periodic = [bool(atoms.pbc.any()) for atoms in members]
        if any(periodic) != all(periodic):
            raise ValueError('FENNIX batches must group periodic and gas structures separately')
        for atoms in members:
            self._validate_multiplicity(atoms)
            if atoms.pbc.any() and not atoms.pbc.all():
                raise ValueError('FENNIX exports require full three-dimensional periodicity')
            _inputs._elements(atoms, self.atomic_numbers)
        counts = [len(atoms) for atoms in members]
        nat = sum(counts)
        padding = max(1, len(members) + 2 - nat)
        charges = ([float(total_charge)] if total_charge is not None else [
            _inputs._charge(atoms)
            for atoms in members])
        if len(charges) != len(members) or not np.isfinite(charges).all():
            raise ValueError('One finite total charge per member is required')
        data = {
            'species': np.concatenate([*(atoms.numbers.astype(np.int32) for atoms in members),
                                       np.full(padding, -1, np.int32)]),
            'coordinates': np.concatenate([*(atoms.positions.astype(self.dtype) for atoms in members),
                                           np.zeros((padding, 3), self.dtype)]),
            'natoms': np.array(counts + [0], np.int32),
            'batch_index': np.concatenate([np.repeat(np.arange(len(members), dtype=np.int32), counts),
                                           np.full(padding, len(members), np.int32)]),
            'true_atoms': np.arange(nat + padding) < nat,
            'true_sys': np.arange(len(members) + 1) < len(members),
            'total_charge': np.array(charges + [0.], self.dtype),
        }
        if all(periodic):
            cells = np.stack([*(atoms.cell.array for atoms in members), members[0].cell.array]).astype(self.dtype)
            data.update(cells=cells, reciprocal_cells=np.linalg.inv(cells))
        states = []
        for processor, state in zip((self.generator, self.filter, self.indexer), self.states):
            state, data = processor(state, data)
            states.append(state)
        self.states = tuple(states)
        data.pop('reciprocal_cells', None)
        data.pop('block_index_overflow', None)
        data.setdefault('cells', np.tile(np.eye(3, dtype=self.dtype)[None], (len(members) + 1, 1, 1)))
        data['block_index'] = {name: (data['block_index'][name] if data['block_index'].get(name) is not None
                                     else np.array([nat + padding], np.int32)) for name in self.block_names}
        for name in ('graph', 'graph_embed'):
            graph = data[name]
            graph.pop('d12', None)
            graph.pop('overflow', None)
            if name == 'graph_embed':
                graph.pop('pbc_shifts', None)
            else:
                if graph.get('pbc_shifts') is None:
                    graph['pbc_shifts'] = np.zeros((len(graph['edge_src']), 3), np.int32)
        return jax.device_put(data, self.jax_device)

    def calculate(self, atoms=None, properties=None, system_changes=None):
        from .. import _jax
        return _jax.calculate_single(self, atoms, properties, system_changes)

    def _calculate(self, atoms_list, properties):
        import jax.numpy as jnp
        from .. import _jax

        members = list(atoms_list)
        needs_hessian = 'hessian' in properties
        properties = [name for name in properties if name != 'hessian']
        if needs_hessian and not properties:
            return {'hessian': [self.get_hessian(atoms) for atoms in members]}
        data = self._build_inputs(members)
        strain = bool(set(properties) & {'stress', 'virial'})
        derivative, volume = None, None
        if strain:
            if not all(atoms.pbc.all() for atoms in members):
                raise ValueError('Stress/virial requires periodic members')
            (_, output), derivative = self._strain_result(data, jnp.zeros_like(data['cells']))
            if 'stress' in properties:
                volume = np.abs(np.linalg.det(np.array([a.cell.array for a in members])))
        elif 'forces' in properties:
            output = self._call(data)
        else:
            output = {'total_energy': self._energy_call(data)}
        result = _jax.array_properties(
            output['total_energy'], members, properties, forces=output.get('forces'),
            strain_gradient=derivative, volume=volume, source_unit=self.MODEL_ENERGY_UNIT,
        )
        if needs_hessian:
            # get_hessian already returns Hartree/Å².
            result['hessian'] = [self.get_hessian(atoms) for atoms in members]
        return result

    def _batch_key(self, atoms):
        return (bool(atoms.pbc.any()),)

    def _calculate_batch(self, atoms_list, properties):
        return self._calculate(atoms_list, properties)

    def _analytic_hessian(self, atoms) -> np.ndarray:
        from .. import _jax
        data = self._build_inputs([atoms])
        hessian = _jax.hessian(self._hvp, data, data['coordinates'], len(atoms), self.jax_device)
        result = _convert_properties({'hessian': np.asarray(hessian, dtype=float)},
                                     source_unit=self.MODEL_ENERGY_UNIT)
        return result['hessian']

    def get_hvp(self, atoms, n: np.ndarray):
        if getattr(self, "solvent_correction", None) is not None:
            return super().get_hvp(atoms, n)
        self._reject_unsupported_pbc(atoms)

        if hasattr(n, "detach"):
            vector = n.detach().cpu().numpy()
        else:
            vector = np.asarray(n, dtype=float)
        import jax
        data = self._build_inputs([atoms])
        packed = np.zeros(data['coordinates'].shape, self.dtype)
        vector = np.asarray(vector, dtype=self.dtype)
        if vector.shape not in ((len(atoms) * 3,), (len(atoms), 3)) or not np.isfinite(vector).all():
            raise ValueError('HVP vector must have shape [N,3] or [3N] and finite values')
        packed[:len(atoms)] = vector.reshape(len(atoms), 3)
        product, output = self._hvp(data, jax.device_put(packed, self.jax_device))
        hvp_ev = np.asarray(product)[:len(atoms)].flatten()
        forces_ev = np.asarray(output['forces'])[:len(atoms)].flatten()
        energy_ev = float(np.asarray(output['total_energy'])[0])
        result = _convert_properties(
            {'hvp': (np.asarray(hvp_ev, dtype=float), np.asarray(forces_ev, dtype=float), energy_ev)},
            source_unit=self.MODEL_ENERGY_UNIT,
        )

        import torch

        return tuple(torch.as_tensor(value, dtype=torch.float64, device=self.device)
                     for value in result['hvp'])