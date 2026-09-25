from typing import Union
import numpy as np 
import torch
from torch import nn

from ase import Atoms
from ase.io import write
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator
from ase.stress import full_3x3_to_voigt_6_stress
from ..tools.output import batch_to_atoms

from ..tools import torch_geometric, torch_tools, to_numpy
from ..data import AtomicData

__all__ = ["EvaluateTask"]

class EvaluateTask(nn.Module):
    """CACE Evaluator 
    args:
        model_path: str, path to a trusted pickled model
        device: str, device to run on (cuda or cpu)
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
        energy_key: str, name of energy key in model output
        forces_key: str, name of forces key in model output
        stress_key: str, name of stress key in model output
        atomic_energies: dict, dictionary of atomic energies to add to model output
    """

    def __init__(
        self,
        model_path: Union[str, nn.Module],
        device: str = "cpu",
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        energy_key: str = 'energy',
        forces_key: str = 'forces',
        stress_key: str = 'stress',
        other_keys: list = None,
        atomic_energies: dict = None,
        data_key: dict = None,
        ):

        super().__init__()

        if isinstance(model_path, str):
            self.model = torch.load(f=model_path, map_location=device, weights_only=False)
        elif isinstance(model_path, nn.Module):
            self.model = model_path
        else:
            raise ValueError("model_path must be a string or nn.Module")

        self.model.to(device)

        self.device = torch_tools.init_device(device)
        try:
            self.cutoff = self.model.representation.cutoff
        except AttributeError:
            self.cutoff = self.model.models[0].representation.cutoff
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.stress_key = stress_key
        self.other_keys = [] if other_keys is None else list(other_keys)
        self.data_key = data_key

        self.atomic_energies = atomic_energies
        
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, data=None, batch_size=1, compute_stress=False, xyz_output=None):
        """
        Calculate properties.
        args:
             data: torch_geometric.data.Data, torch_geometric.data.Batch, list of ASE Atoms objects, or torch_geometric.data.DataLoader
             batch_size: int, batch size
             compute_stress: bool, whether to compute stress
        """
        if xyz_output is not None and batch_size != 1:
            raise ValueError("Batch size must be 1 to write xyz files")
        originals = None
        if isinstance(data, Atoms):
            originals = [data]
        elif isinstance(data, list):
            if not data or not all(isinstance(a, Atoms) for a in data):
                raise ValueError("Input data must be a nonempty list of ASE Atoms objects")
            originals = data

        if originals is not None:
            batches = torch_geometric.dataloader.DataLoader(
                [AtomicData.from_atoms(a, cutoff=self.cutoff, data_key=self.data_key)
                 for a in originals], batch_size=batch_size, shuffle=False)
        elif isinstance(data, torch_geometric.batch.Batch):
            batches = [data]
        elif isinstance(data, torch_geometric.dataloader.DataLoader):
            batches = data
        else:
            raise ValueError("Input data type not recognized")

        collected = {key: [] for key in ('energy', 'forces', 'stress', *self.other_keys)}
        atoms_list = []
        for source_batch in batches:
            batch = source_batch.clone().to(self.device)
            if xyz_output is not None and len(batch.ptr) != 2:
                raise ValueError("Batch size must be 1 to write xyz files")
            output = self.model(batch.to_dict(), training=True, compute_stress=compute_stress)
            converted = {}
            for name, key, factor in (
                ('energy', self.energy_key, self.energy_units_to_eV),
                ('forces', self.forces_key, self.energy_units_to_eV / self.length_units_to_A),
                ('stress', self.stress_key, self.energy_units_to_eV / self.length_units_to_A**3),
            ):
                value = output.get(key)
                if value is None or (name == 'stress' and not compute_stress):
                    continue
                value = np.array(to_numpy(value), copy=True)
                if name == 'energy':
                    value = np.atleast_1d(value)
                    if self.atomic_energies is not None:
                        offset = self._add_atomic_energies(batch)
                        value += offset.reshape((-1,) + (1,) * (value.ndim - 1))
                converted[name] = value * factor
                collected[name].append(converted[name])
            for key in self.other_keys:
                if output.get(key) is not None:
                    converted[key] = np.atleast_1d(to_numpy(output[key]))
                    collected[key].append(converted[key])

            if xyz_output is not None:
                if originals is not None:
                    original = originals[len(atoms_list)]
                else:
                    # Graphs lack original partial-PBC metadata (see batch_to_atoms).
                    original = batch_to_atoms(source_batch)[0]
                    if source_batch['stress'] is not None:
                        original.info['stress'] = to_numpy(source_batch.stress)[0]
                info, arrays = {}, {}
                if 'energy' in converted:
                    info[self.energy_key] = converted['energy'][0]
                if 'forces' in converted:
                    arrays[self.forces_key] = converted['forces']
                if 'stress' in converted:
                    info[self.stress_key] = converted['stress'][0]
                for key in self.other_keys:
                    if key not in converted:
                        continue
                    value = converted[key]
                    if value.ndim > 2 and value.shape[0] == 1:
                        value = value[0]
                    if value.ndim > 2:
                        value = value.reshape(value.shape[0], -1)
                    target = arrays if value.shape[0] == len(original) else info
                    if np.iscomplexobj(value):
                        target[key + '_real'], target[key + '_imag'] = value.real, value.imag
                    else:
                        target[key] = value
                atoms_list.append(_prediction_atoms(original, info, arrays))

        if xyz_output is not None:
            write(xyz_output, atoms_list, format='extxyz')
        return {key: np.concatenate(values, axis=0) if values else None
                for key, values in collected.items()}

    def _add_atomic_energies(self, batch: torch_geometric.batch.Batch):
        e0_list = []
        atomic_numbers_list = np.split(to_numpy(batch['atomic_numbers']),
             indices_or_sections=batch.ptr[1:],
             axis=0,
             )[:-1]
        for atomic_numbers in atomic_numbers_list:
            e0_list.append(sum(self.atomic_energies.get(Z, 0) for Z in atomic_numbers))
        return np.array(e0_list)


def _prediction_atoms(original, info, arrays):
    """Copy valid references, then replace only explicitly named predictions."""
    atoms = original.copy()
    references = {}
    if original.calc is not None and not original.calc.check_state(original):
        references = {key: np.array(value, copy=True) for key, value in
                      original.calc.results.items() if key in all_properties and value is not None}
    # Metadata has precedence over calculator storage, as in reference reads.
    for key in all_properties:
        location = atoms.arrays if key in ('forces', 'charges', 'magmoms', 'stresses', 'energies') else atoms.info
        if key in location and location[key] is not None:
            references[key] = np.array(location[key], copy=True)
    for values, location in ((info, atoms.info), (arrays, atoms.arrays)):
        for key, value in values.items():
            if key in all_properties:
                references[key] = value
            elif location is atoms.arrays:
                atoms.set_array(key, np.array(value, copy=True))
            else:
                location[key] = np.array(value, copy=True)
    for key in references:
        atoms.info.pop(key, None)
        atoms.arrays.pop(key, None)
    if ('energy' in info or 'forces' in arrays) and 'free_energy' not in info:
        references.pop('free_energy', None)
        atoms.info.pop('free_energy', None)
    if 'stress' in references and np.shape(references['stress']) == (3, 3):
        references['stress'] = full_3x3_to_voigt_6_stress(references['stress'])
    if references:
        atoms.calc = SinglePointCalculator(atoms, **references)
    return atoms
