import numpy as np
from .torch_tools import to_numpy
from typing import Dict, Optional
import ase

def batch_to_atoms(batched_data: Dict,
                   pred_data: Optional[Dict] = None,
                   output_file: str = None,
                   energy_key: str = 'energy',
                   forces_key: str = 'forces',
                   cace_energy_key: str = 'energy',
                   cace_forces_key: str = 'forces'):
    """
    Create ASE Atoms objects from batched graph data and write to an XYZ file.

    Parameters:
    - batched_data (Dict): Batched data containing graph information.
    - pred_data (Dict): Predicted data. If not given, the pred_data name is assumed to also be the batched_data.
    - energy_key (str): Key for accessing energy information in batched_data.
    - forces_key (str): Key for accessing force information in batched_data.
    - cace_energy_key (str): Key for accessing CACE energy information.
    - cace_forces_key (str): Key for accessing CACE force information.
    - output_file (str): Name of the output file to write the Atoms objects.

    Original partial PBC is not stored in graphs; reconstruction retains the
    legacy all-or-none PBC inference from the cell.
    """

    if pred_data is None:
        pred_data = batched_data
    atoms_list = []
    batch = batched_data.batch
    num_graphs = batch.max().item() + 1

    for i in range(num_graphs):
        # Mask to extract nodes for each graph
        mask = batch == i

        # Extract node features, edge indices, etc., for each graph
        positions = to_numpy(batched_data['positions'][mask])
        atomic_numbers = to_numpy(batched_data['atomic_numbers'][mask])
        cell = to_numpy(batched_data['cell'][3*i:3*i+3])

        # Set periodic boundary conditions if the cell is defined
        pbc = np.all(np.mean(cell, axis=0) > 0)

        # Create the Atoms object
        atoms = ase.Atoms(numbers=atomic_numbers, positions=positions, cell=cell, pbc=pbc)
        for source, ekey, fkey in ((batched_data, energy_key, forces_key),
                                   (pred_data, cace_energy_key, cace_forces_key)):
            source = source if isinstance(source, dict) else source.to_dict()
            if source.get(ekey) is not None:
                energy = to_numpy(source[ekey][i])
                atoms.info[ekey] = energy.item() if np.ndim(energy) == 0 else energy
            if source.get(fkey) is not None:
                atoms.arrays[fkey] = to_numpy(source[fkey][mask]).copy()
        atoms_list.append(atoms)

    # Write all atoms to the output file
    if output_file:
        ase.io.write(output_file, atoms_list, append=True)
    return atoms_list
