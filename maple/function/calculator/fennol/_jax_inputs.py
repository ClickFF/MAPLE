"""FeNNol BIO1 NumPy preprocessing, extracted from native commit 05f362e731223bf3.

Only graph generation/filtering and chemical indexing are retained. The exported
JAX graph owns all continuous geometry processors and neural-network operations.
Source: FeNNol-tools/FeNNol, LGPL-3.0; native host mathematics is preserved.
"""
import dataclasses
from typing import ClassVar
import numpy as np

CHEMICAL_BLOCKS_NAMES = ['H', 'CNOPSSe', 'halogens', 'alkali', 'alkaline', 'transition_metals', 'post_metals', 'metalloids', 'lanthanides', 'actinides', 'noble gases']
CHEMICAL_BLOCKS_NAMES = [b.strip().upper() for b in CHEMICAL_BLOCKS_NAMES]
CHEMICAL_BLOCKS = -np.ones(120, dtype=int)
CHEMICAL_BLOCKS[1] = 0
CHEMICAL_BLOCKS[2] = 10
CHEMICAL_BLOCKS[3] = 3
CHEMICAL_BLOCKS[4] = 4
CHEMICAL_BLOCKS[5] = 7
CHEMICAL_BLOCKS[6] = 1
CHEMICAL_BLOCKS[7] = 1
CHEMICAL_BLOCKS[8] = 1
CHEMICAL_BLOCKS[9] = 2
CHEMICAL_BLOCKS[10] = 10
CHEMICAL_BLOCKS[11] = 3
CHEMICAL_BLOCKS[12] = 4
CHEMICAL_BLOCKS[13] = 6
CHEMICAL_BLOCKS[14] = 7
CHEMICAL_BLOCKS[15] = 1
CHEMICAL_BLOCKS[16] = 1
CHEMICAL_BLOCKS[17] = 2
CHEMICAL_BLOCKS[18] = 10
CHEMICAL_BLOCKS[19] = 3
CHEMICAL_BLOCKS[20] = 4
CHEMICAL_BLOCKS[21] = 5
CHEMICAL_BLOCKS[22] = 5
CHEMICAL_BLOCKS[23] = 5
CHEMICAL_BLOCKS[24] = 5
CHEMICAL_BLOCKS[25] = 5
CHEMICAL_BLOCKS[26] = 5
CHEMICAL_BLOCKS[27] = 5
CHEMICAL_BLOCKS[28] = 5
CHEMICAL_BLOCKS[29] = 5
CHEMICAL_BLOCKS[30] = 5
CHEMICAL_BLOCKS[31] = 6
CHEMICAL_BLOCKS[32] = 7
CHEMICAL_BLOCKS[33] = 7
CHEMICAL_BLOCKS[34] = 1
CHEMICAL_BLOCKS[35] = 2
CHEMICAL_BLOCKS[36] = 10
CHEMICAL_BLOCKS[37] = 3
CHEMICAL_BLOCKS[38] = 4
CHEMICAL_BLOCKS[39] = 5
CHEMICAL_BLOCKS[40] = 5
CHEMICAL_BLOCKS[41] = 5
CHEMICAL_BLOCKS[42] = 5
CHEMICAL_BLOCKS[43] = 5
CHEMICAL_BLOCKS[44] = 5
CHEMICAL_BLOCKS[45] = 5
CHEMICAL_BLOCKS[46] = 5
CHEMICAL_BLOCKS[47] = 5
CHEMICAL_BLOCKS[48] = 5
CHEMICAL_BLOCKS[49] = 6
CHEMICAL_BLOCKS[50] = 6
CHEMICAL_BLOCKS[51] = 7
CHEMICAL_BLOCKS[52] = 7
CHEMICAL_BLOCKS[53] = 2
CHEMICAL_BLOCKS[54] = 10
CHEMICAL_BLOCKS[55] = 3
CHEMICAL_BLOCKS[56] = 4
CHEMICAL_BLOCKS[57] = 8
CHEMICAL_BLOCKS[58] = 8
CHEMICAL_BLOCKS[59] = 8
CHEMICAL_BLOCKS[60] = 8
CHEMICAL_BLOCKS[61] = 8
CHEMICAL_BLOCKS[62] = 8
CHEMICAL_BLOCKS[63] = 8
CHEMICAL_BLOCKS[64] = 8
CHEMICAL_BLOCKS[65] = 8
CHEMICAL_BLOCKS[66] = 8
CHEMICAL_BLOCKS[67] = 8
CHEMICAL_BLOCKS[68] = 8
CHEMICAL_BLOCKS[69] = 8
CHEMICAL_BLOCKS[70] = 8
CHEMICAL_BLOCKS[71] = 8
CHEMICAL_BLOCKS[72] = 5
CHEMICAL_BLOCKS[73] = 5
CHEMICAL_BLOCKS[74] = 5
CHEMICAL_BLOCKS[75] = 5
CHEMICAL_BLOCKS[76] = 5
CHEMICAL_BLOCKS[77] = 5
CHEMICAL_BLOCKS[78] = 5
CHEMICAL_BLOCKS[79] = 5
CHEMICAL_BLOCKS[80] = 5
CHEMICAL_BLOCKS[81] = 6
CHEMICAL_BLOCKS[82] = 6
CHEMICAL_BLOCKS[83] = 6
CHEMICAL_BLOCKS[84] = 6
CHEMICAL_BLOCKS[85] = 2
CHEMICAL_BLOCKS[86] = 10
CHEMICAL_BLOCKS[87] = 3
CHEMICAL_BLOCKS[88] = 4
CHEMICAL_BLOCKS[89] = 9
CHEMICAL_BLOCKS[90] = 9
CHEMICAL_BLOCKS[91] = 9
CHEMICAL_BLOCKS[92] = 9
CHEMICAL_BLOCKS[93] = 9
CHEMICAL_BLOCKS[94] = 9
CHEMICAL_BLOCKS[95] = 9
CHEMICAL_BLOCKS[96] = 9
CHEMICAL_BLOCKS[97] = 9
CHEMICAL_BLOCKS[98] = 9
CHEMICAL_BLOCKS[99] = 9
CHEMICAL_BLOCKS[100] = 9
CHEMICAL_BLOCKS[101] = 9
CHEMICAL_BLOCKS[102] = 9
CHEMICAL_BLOCKS[103] = 9
CHEMICAL_BLOCKS[104] = 5
CHEMICAL_BLOCKS[105] = 5
CHEMICAL_BLOCKS[106] = 5
CHEMICAL_BLOCKS[107] = 5
CHEMICAL_BLOCKS[108] = 5
CHEMICAL_BLOCKS[109] = 5
CHEMICAL_BLOCKS[110] = 5
CHEMICAL_BLOCKS[111] = 5
CHEMICAL_BLOCKS[112] = 5
CHEMICAL_BLOCKS[113] = 6
CHEMICAL_BLOCKS[114] = 6
CHEMICAL_BLOCKS[115] = 6
CHEMICAL_BLOCKS[116] = 6
CHEMICAL_BLOCKS[117] = 2
CHEMICAL_BLOCKS[118] = 10

@dataclasses.dataclass(frozen=True)
class GraphGenerator:
    """Generate a graph from a set of coordinates

    FPID: GRAPH

    For now, we generate all pairs of atoms and filter based on cutoff.
    If a `nblist_skin` is present in the state, we generate a second graph with a larger cutoff that includes all pairs within the cutoff+skin. This graph is then reused by the `update_skin` method to update the original graph without recomputing the full nblist.
    """
    cutoff: float
    'Cutoff distance for the graph.'
    graph_key: str = 'graph'
    'Key of the graph in the outputs.'
    switch_params: dict = dataclasses.field(default_factory=dict, hash=False)
    'Parameters for the switching function. See `fennol.models.misc.misc.SwitchFunction`.'
    kmax: int = 30
    'Maximum number of k-points to consider.'
    kthr: float = 1e-06
    'Threshold for k-point filtering.'
    k_space: bool = False
    'Whether to generate k-space information for the graph.'
    mult_size: float = 1.05
    'Multiplicative factor for resizing the nblist.'
    FPID: ClassVar[str] = 'GRAPH'

    def init(self):
        return dict({'max_nat': 1, 'npairs': 1, 'nblist_mult_size': self.mult_size})

    def __call__(self, state, inputs, return_state_update=False, add_margin=False):
        """build a nblist on cpu with numpy and dynamic shapes + store max shapes"""
        if self.graph_key in inputs:
            graph = inputs[self.graph_key]
            if 'keep_graph' in graph:
                return (state, inputs)
        coords = np.array(inputs['coordinates'], dtype=np.float32)
        natoms = np.array(inputs['natoms'], dtype=np.int32)
        batch_index = np.array(inputs['batch_index'], dtype=np.int32)
        new_state = {**state}
        state_up = {}
        mult_size = state.get('nblist_mult_size', self.mult_size)
        assert mult_size >= 1.0, 'mult_size should be larger or equal than 1.0'
        if natoms.shape[0] == 1:
            max_nat = coords.shape[0]
            true_max_nat = max_nat
        else:
            max_nat = state.get('max_nat', round(coords.shape[0] / natoms.shape[0]))
            true_max_nat = int(np.max(natoms))
            if true_max_nat > max_nat:
                add_atoms = state.get('add_atoms', 0)
                new_maxnat = true_max_nat + add_atoms
                state_up['max_nat'] = (new_maxnat, max_nat)
                new_state['max_nat'] = new_maxnat
        cutoff_skin = self.cutoff + state.get('nblist_skin', 0.0)
        apply_pbc = 'cells' in inputs
        minimage = 'minimum_image' in inputs.get('flags', {})
        include_self_image = apply_pbc and (not minimage)
        shift = 0 if include_self_image else 1
        p1, p2 = np.triu_indices(true_max_nat, shift)
        p1, p2 = (p1.astype(np.int32), p2.astype(np.int32))
        pbc_shifts = None
        if natoms.shape[0] > 1:
            mask_p12 = ((p1[None, :] < natoms[:, None]) * (p2[None, :] < natoms[:, None])).flatten()
            shift = np.concatenate((np.array([0], dtype=np.int32), np.cumsum(natoms[:-1], dtype=np.int32)))
            p1 = np.where(mask_p12, (p1[None, :] + shift[:, None]).flatten(), -1)
            p2 = np.where(mask_p12, (p2[None, :] + shift[:, None]).flatten(), -1)
        if not apply_pbc:
            vec = coords[p2] - coords[p1]
        else:
            cells = np.array(inputs['cells'], dtype=np.float32)
            reciprocal_cells = np.array(inputs['reciprocal_cells'], dtype=np.float32)
            if minimage:
                vec = coords[p2] - coords[p1]
                if cells.shape[0] == 1:
                    vecpbc = np.dot(vec, reciprocal_cells[0])
                    pbc_shifts = -np.round(vecpbc).astype(np.int32)
                    vec = vec + np.dot(pbc_shifts, cells[0])
                else:
                    batch_index_vec = batch_index[p1]
                    vecpbc = np.einsum('aj,aji->ai', vec, reciprocal_cells[batch_index_vec])
                    pbc_shifts = -np.round(vecpbc).astype(np.int32)
                    vec = vec + np.einsum('aj,aji->ai', pbc_shifts, cells[batch_index_vec])
            else:
                if cells.shape[0] == 1:
                    coords_pbc = np.dot(coords, reciprocal_cells[0])
                    at_shifts = -np.floor(coords_pbc).astype(np.int32)
                    coords_pbc = coords + np.dot(at_shifts, cells[0])
                else:
                    coords_pbc = np.einsum('aj,aji->ai', coords, reciprocal_cells[batch_index])
                    at_shifts = -np.floor(coords_pbc).astype(np.int32)
                    coords_pbc = coords + np.einsum('aj,aji->ai', at_shifts, cells[batch_index])
                vec = coords_pbc[p2] - coords_pbc[p1]
                inv_distances = np.sum(reciprocal_cells ** 2, axis=1) ** 0.5
                cdinv = cutoff_skin * inv_distances
                num_repeats_all = np.ceil(cdinv).astype(np.int32)
                if 'true_sys' in inputs:
                    num_repeats_all = np.where(np.array(inputs['true_sys'], dtype=bool)[:, None], num_repeats_all, 0)
                num_repeats = np.max(num_repeats_all, axis=0)
                num_repeats_prev = np.array(state.get('num_repeats_pbc', (0, 0, 0)))
                if np.any(num_repeats > num_repeats_prev):
                    num_repeats_new = np.maximum(num_repeats, num_repeats_prev)
                    state_up['num_repeats_pbc'] = (tuple(num_repeats_new), tuple(num_repeats_prev))
                    new_state['num_repeats_pbc'] = tuple(num_repeats_new)
                cell_shift_pbc = np.array(np.meshgrid(*[np.arange(-n, n + 1) for n in num_repeats]), dtype=np.int32).T.reshape(-1, 3)
                if cells.shape[0] == 1:
                    dvec = np.dot(cell_shift_pbc, cells[0])[None, :, :]
                    vec = (vec[:, None, :] + dvec).reshape(-1, 3)
                    pbc_shifts = np.broadcast_to(cell_shift_pbc[None, :, :], (p1.shape[0], cell_shift_pbc.shape[0], 3)).reshape(-1, 3)
                    p1 = np.broadcast_to(p1[:, None], (p1.shape[0], cell_shift_pbc.shape[0])).flatten()
                    p2 = np.broadcast_to(p2[:, None], (p2.shape[0], cell_shift_pbc.shape[0])).flatten()
                    if natoms.shape[0] > 1:
                        mask_p12 = np.broadcast_to(mask_p12[:, None], (mask_p12.shape[0], cell_shift_pbc.shape[0])).flatten()
                else:
                    dvec = np.einsum('bj,sji->sbi', cell_shift_pbc, cells)
                    cell_shift_pbc = np.broadcast_to(cell_shift_pbc[None, :, :], (num_repeats_all.shape[0], cell_shift_pbc.shape[0], 3))
                    mask = np.all(np.abs(cell_shift_pbc) <= num_repeats_all[:, None, :], axis=-1).flatten()
                    idx = np.nonzero(mask)[0]
                    nshifts = idx.shape[0]
                    nshifts_prev = state.get('nshifts_pbc', 0)
                    if nshifts > nshifts_prev or add_margin:
                        nshifts_new = int(mult_size * max(nshifts, nshifts_prev)) + 1
                        state_up['nshifts_pbc'] = (nshifts_new, nshifts_prev)
                        new_state['nshifts_pbc'] = nshifts_new
                    dvec_filter = dvec.reshape(-1, 3)[idx, :]
                    cell_shift_pbc_filter = cell_shift_pbc.reshape(-1, 3)[idx, :]
                    nrep = np.prod(2 * num_repeats_all + 1, axis=1)
                    bshift = np.concatenate((np.array([0]), np.cumsum(nrep)[:-1]))
                    batch_index_vec = batch_index[p1]
                    nrep_vec = np.where(mask_p12, nrep[batch_index_vec], 0)
                    vec = vec.repeat(nrep_vec, axis=0)
                    nvec_pbc = nrep_vec.sum()
                    nvec_pbc_prev = state.get('nvec_pbc', 0)
                    if nvec_pbc > nvec_pbc_prev or add_margin:
                        nvec_pbc_new = int(mult_size * max(nvec_pbc, nvec_pbc_prev)) + 1
                        state_up['nvec_pbc'] = (nvec_pbc_new, nvec_pbc_prev)
                        new_state['nvec_pbc'] = nvec_pbc_new
                    dshift = np.concatenate((np.array([0]), np.cumsum(nrep_vec)[:-1])).repeat(nrep_vec)
                    icellshift = np.arange(dshift.shape[0]) - dshift + bshift[batch_index_vec].repeat(nrep_vec)
                    vec = vec + dvec_filter[icellshift]
                    pbc_shifts = cell_shift_pbc_filter[icellshift]
                    p1 = np.repeat(p1, nrep_vec)
                    p2 = np.repeat(p2, nrep_vec)
                    if natoms.shape[0] > 1:
                        mask_p12 = np.repeat(mask_p12, nrep_vec)
        d12 = (vec ** 2).sum(axis=-1)
        if natoms.shape[0] > 1:
            d12 = np.where(mask_p12, d12, cutoff_skin ** 2)
        max_pairs = state.get('npairs', 1)
        mask = d12 < cutoff_skin ** 2
        if include_self_image:
            mask_self = np.logical_or(p1 != p2, d12 > 0.001)
            mask = np.logical_and(mask, mask_self)
        idx = np.nonzero(mask)[0]
        npairs = idx.shape[0]
        if npairs > max_pairs or add_margin:
            prev_max_pairs = max_pairs
            max_pairs = int(mult_size * max(npairs, max_pairs)) + 1
            state_up['npairs'] = (max_pairs, prev_max_pairs)
            new_state['npairs'] = max_pairs
        nat = coords.shape[0]
        edge_src = np.full(max_pairs, nat, dtype=np.int32)
        edge_dst = np.full(max_pairs, nat, dtype=np.int32)
        d12_ = np.full(max_pairs, cutoff_skin ** 2)
        edge_src[:npairs] = p1[idx]
        edge_dst[:npairs] = p2[idx]
        d12_[:npairs] = d12[idx]
        d12 = d12_
        if apply_pbc:
            pbc_shifts_ = np.zeros((max_pairs, 3), dtype=np.int32)
            pbc_shifts_[:npairs] = pbc_shifts[idx]
            pbc_shifts = pbc_shifts_
            if not minimage:
                pbc_shifts[:npairs] = pbc_shifts[:npairs] + at_shifts[edge_dst[:npairs]] - at_shifts[edge_src[:npairs]]
        if include_self_image:
            mask_noself = edge_src != edge_dst
            idx_noself = np.nonzero(mask_noself)[0]
            npairs_noself = idx_noself.shape[0]
            max_noself = state.get('npairs_noself', 1)
            if npairs_noself > max_noself or add_margin:
                prev_max_noself = max_noself
                max_noself = int(mult_size * max(npairs_noself, max_noself)) + 1
                state_up['npairs_noself'] = (max_noself, prev_max_noself)
                new_state['npairs_noself'] = max_noself
            edge_src_noself = np.full(max_noself, nat, dtype=np.int32)
            edge_dst_noself = np.full(max_noself, nat, dtype=np.int32)
            d12_noself = np.full(max_noself, cutoff_skin ** 2)
            pbc_shifts_noself = np.zeros((max_noself, 3), dtype=np.int32)
            edge_dst_noself[:npairs_noself] = edge_dst[idx_noself]
            edge_src_noself[:npairs_noself] = edge_src[idx_noself]
            d12_noself[:npairs_noself] = d12[idx_noself]
            pbc_shifts_noself[:npairs_noself] = pbc_shifts[idx_noself]
            edge_src = np.concatenate((edge_src, edge_dst_noself))
            edge_dst = np.concatenate((edge_dst, edge_src_noself))
            d12 = np.concatenate((d12, d12_noself))
            pbc_shifts = np.concatenate((pbc_shifts, -pbc_shifts_noself))
        else:
            edge_src, edge_dst = (np.concatenate((edge_src, edge_dst)), np.concatenate((edge_dst, edge_src)))
            d12 = np.concatenate((d12, d12))
            if apply_pbc:
                pbc_shifts = np.concatenate((pbc_shifts, -pbc_shifts))
        if 'nblist_skin' in state:
            edge_src_skin = edge_src
            edge_dst_skin = edge_dst
            if apply_pbc:
                pbc_shifts_skin = pbc_shifts
            max_pairs_skin = state.get('npairs_skin', 1)
            mask = d12 < self.cutoff ** 2
            idx = np.nonzero(mask)[0]
            npairs_skin = idx.shape[0]
            if npairs_skin > max_pairs_skin or add_margin:
                prev_max_pairs_skin = max_pairs_skin
                max_pairs_skin = int(mult_size * max(npairs_skin, max_pairs_skin)) + 1
                state_up['npairs_skin'] = (max_pairs_skin, prev_max_pairs_skin)
                new_state['npairs_skin'] = max_pairs_skin
            edge_src = np.full(max_pairs_skin, nat, dtype=np.int32)
            edge_dst = np.full(max_pairs_skin, nat, dtype=np.int32)
            d12_ = np.full(max_pairs_skin, self.cutoff ** 2)
            edge_src[:npairs_skin] = edge_src_skin[idx]
            edge_dst[:npairs_skin] = edge_dst_skin[idx]
            d12_[:npairs_skin] = d12[idx]
            d12 = d12_
            if apply_pbc:
                pbc_shifts = np.zeros((max_pairs_skin, 3), dtype=np.int32)
                pbc_shifts[:npairs_skin] = pbc_shifts_skin[idx]
        graph = inputs.get(self.graph_key, {})
        graph_out = {**graph, 'edge_src': edge_src, 'edge_dst': edge_dst, 'd12': d12, 'overflow': False, 'pbc_shifts': pbc_shifts}
        if 'nblist_skin' in state:
            graph_out['edge_src_skin'] = edge_src_skin
            graph_out['edge_dst_skin'] = edge_dst_skin
            if apply_pbc:
                graph_out['pbc_shifts_skin'] = pbc_shifts_skin
        if self.k_space and apply_pbc:
            if 'k_points' not in graph:
                ks, _, _, bewald = get_reciprocal_space_parameters(reciprocal_cells, self.cutoff, self.kmax, self.kthr)
            graph_out['k_points'] = ks
            graph_out['b_ewald'] = bewald
        output = {**inputs, self.graph_key: graph_out}
        if return_state_update:
            return (dict(new_state), output, state_up)
        return (dict(new_state), output)

@dataclasses.dataclass(frozen=True)
class GraphFilter:
    """Filter a graph based on a cutoff distance

    FPID: GRAPH_FILTER
    """
    cutoff: float
    'Cutoff distance for the filtering.'
    parent_graph: str
    'Key of the parent graph in the inputs.'
    graph_key: str
    'Key of the filtered graph in the outputs.'
    remove_hydrogens: int = False
    'Remove edges where the source is a hydrogen atom.'
    switch_params: dict = dataclasses.field(default_factory=dict)
    'Parameters for the switching function. See `fennol.models.misc.misc.SwitchFunction`.'
    k_space: bool = False
    'Generate k-space information for the graph.'
    kmax: int = 30
    'Maximum number of k-points to consider.'
    kthr: float = 1e-06
    'Threshold for k-point filtering.'
    mult_size: float = 1.05
    'Multiplicative factor for resizing the nblist.'
    FPID: ClassVar[str] = 'GRAPH_FILTER'

    def init(self):
        return dict({'npairs': 1, 'nblist_mult_size': self.mult_size})

    def __call__(self, state, inputs, return_state_update=False, add_margin=False):
        """filter a nblist on cpu with numpy and dynamic shapes + store max shapes"""
        graph_in = inputs[self.parent_graph]
        nat = inputs['species'].shape[0]
        new_state = {**state}
        state_up = {}
        mult_size = state.get('nblist_mult_size', self.mult_size)
        assert mult_size >= 1.0, 'nblist_mult_size should be >= 1.'
        edge_src = np.array(graph_in['edge_src'], dtype=np.int32)
        d12 = np.array(graph_in['d12'], dtype=np.float32)
        if self.remove_hydrogens:
            species = inputs['species']
            src_idx = (edge_src < nat).nonzero()[0]
            mask = np.zeros(edge_src.shape[0], dtype=bool)
            mask[src_idx] = (species > 1)[edge_src[src_idx]]
            d12 = np.where(mask, d12, self.cutoff ** 2)
        mask = d12 < self.cutoff ** 2
        max_pairs = state.get('npairs', 1)
        idx = np.nonzero(mask)[0]
        npairs = idx.shape[0]
        if npairs > max_pairs or add_margin:
            prev_max_pairs = max_pairs
            max_pairs = int(mult_size * max(npairs, max_pairs)) + 1
            state_up['npairs'] = (max_pairs, prev_max_pairs)
            new_state['npairs'] = max_pairs
        filter_indices = np.full(max_pairs, edge_src.shape[0], dtype=np.int32)
        edge_src = np.full(max_pairs, nat, dtype=np.int32)
        edge_dst = np.full(max_pairs, nat, dtype=np.int32)
        d12_ = np.full(max_pairs, self.cutoff ** 2)
        filter_indices[:npairs] = idx
        edge_src[:npairs] = graph_in['edge_src'][idx]
        edge_dst[:npairs] = graph_in['edge_dst'][idx]
        d12_[:npairs] = d12[idx]
        d12 = d12_
        graph = inputs[self.graph_key] if self.graph_key in inputs else {}
        graph_out = {**graph, 'edge_src': edge_src, 'edge_dst': edge_dst, 'filter_indices': filter_indices, 'd12': d12, 'overflow': False}
        if 'cells' in inputs:
            pbc_shifts = np.zeros((max_pairs, 3), dtype=np.int32)
            pbc_shifts[:npairs] = graph_in['pbc_shifts'][idx]
            graph_out['pbc_shifts'] = pbc_shifts
        output = {**inputs, self.graph_key: graph_out}
        if return_state_update:
            return (dict(new_state), output, state_up)
        return (dict(new_state), output)

@dataclasses.dataclass(frozen=True)
class BlockIndexer:
    """Build an index that splits atomic arrays by chemical blocks.

    FPID: BLOCK_INDEXER

    If `species_order` is specified, the output will be a dense array with size (len(species_order), max_size) that can directly index atomic arrays.
    If `species_order` is None, the output will be a dictionary with species as keys and an index to filter atomic arrays for that species as values.

    """
    output_key: str = 'block_index'
    'Key for the output dictionary.'
    add_atoms: int = 0
    'Additional atoms to add to the sizes.'
    add_atoms_margin: int = 10
    'Additional atoms to add to the sizes when adding margin.'
    split_CNOPSSe: bool = False
    FPID: ClassVar[str] = 'BLOCK_INDEXER'

    def init(self):
        return dict({'sizes': {}})

    def build_chemical_blocks(self):
        _CHEMICAL_BLOCKS_NAMES = CHEMICAL_BLOCKS_NAMES.copy()
        if self.split_CNOPSSe:
            _CHEMICAL_BLOCKS_NAMES[1] = 'C'
            _CHEMICAL_BLOCKS_NAMES.extend(['N', 'O', 'P', 'S', 'Se'])
        _CHEMICAL_BLOCKS = CHEMICAL_BLOCKS.copy()
        if self.split_CNOPSSe:
            _CHEMICAL_BLOCKS[6] = 1
            _CHEMICAL_BLOCKS[7] = len(CHEMICAL_BLOCKS_NAMES)
            _CHEMICAL_BLOCKS[8] = len(CHEMICAL_BLOCKS_NAMES) + 1
            _CHEMICAL_BLOCKS[15] = len(CHEMICAL_BLOCKS_NAMES) + 2
            _CHEMICAL_BLOCKS[16] = len(CHEMICAL_BLOCKS_NAMES) + 3
            _CHEMICAL_BLOCKS[34] = len(CHEMICAL_BLOCKS_NAMES) + 4
        return (_CHEMICAL_BLOCKS_NAMES, _CHEMICAL_BLOCKS)

    def __call__(self, state, inputs, return_state_update=False, add_margin=False):
        _CHEMICAL_BLOCKS_NAMES, _CHEMICAL_BLOCKS = self.build_chemical_blocks()
        species = np.array(inputs['species'], dtype=np.int32)
        blocks = _CHEMICAL_BLOCKS[species]
        nat = species.shape[0]
        set_blocks, counts = np.unique(blocks, return_counts=True)
        new_state = {**state}
        state_up = {}
        sizes = state.get('sizes', dict({}))
        new_sizes = {**sizes}
        up_sizes = False
        for s, c in zip(set_blocks, counts):
            if s < 0:
                continue
            key = (s, _CHEMICAL_BLOCKS_NAMES[s])
            if c > sizes.get(key, 0):
                up_sizes = True
                add_atoms = state.get('add_atoms', self.add_atoms)
                if add_margin:
                    add_atoms += state.get('add_atoms_margin', self.add_atoms_margin)
                new_sizes[key] = c + add_atoms
        new_sizes = dict(new_sizes)
        if up_sizes:
            state_up['sizes'] = (new_sizes, sizes)
            new_state['sizes'] = new_sizes
        block_index = {n: None for n in _CHEMICAL_BLOCKS_NAMES}
        for (_, n), c in new_sizes.items():
            block_index[n] = np.full(c, nat, dtype=np.int32)
        for s, c in zip(set_blocks, counts):
            if s < 0:
                continue
            block_index[_CHEMICAL_BLOCKS_NAMES[s]][:c] = np.nonzero(blocks == s)[0]
        output = {**inputs, self.output_key: block_index, self.output_key + '_overflow': False}
        if return_state_update:
            return (dict(new_state), output, state_up)
        return (dict(new_state), output)
