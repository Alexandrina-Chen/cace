# the CACE calculator for ASE

from typing import Union, List

import numpy as np 
import torch

from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from ..tools import torch_geometric, torch_tools, to_numpy
from ..data import AtomicData
from ..modules import Forces, DirectForces
from ..models import NeuralNetworkPotential, CombinePotential
 
__all__ = ["CACECalculator"]

class CACECalculator(Calculator):
    """CACE ASE Calculator
    args:
        model_path: str or nn.module, path to a trusted pickled model or model object
        device: str, device to run on (cuda or cpu)
        compute_stress: bool, whether to compute stress
        energy_key: str, key for energy in model output
        forces_key: str, key for forces in model output
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
        atomic_energies: dict, dictionary of atomic energies to add to model output
    """

    def __init__(
        self,
        model_path: Union[str, torch.nn.Module],
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        electric_field_unit: float = 1.0,
        compute_stress = False,
        energy_key: str = 'energy',
        forces_key: str = 'forces',
        stress_key: str = 'stress',
        charge_key: str = None,
        charge_unit: float = 1.0/(90.0474)**0.5, # the standard normal factor in accordance with the cace convention used in ewald.py
        bec_key: str = 'bec',
        data_key: dict = None,
        external_field: Union[float,List[float]] = None,
        keep_neutral: bool = True, # to keep BEC sum to be neutral
        atomic_energies: dict = None,
        output_index: int = None, # only used for multi-output models
        **kwargs,
        ):

        Calculator.__init__(self, **kwargs)
        self.implemented_properties = [
            "energy",
            "forces",
        ]

        if compute_stress:
            self.implemented_properties.append("stress")

        if charge_key is not None:
            self.implemented_properties.extend(
                [
                    "charges",
                ]
            )

        self.results = {}

        if isinstance(model_path, str):
            self.model = torch.load(f=model_path, map_location=device, weights_only=False)
        elif isinstance(model_path, torch.nn.Module):
            self.model = model_path
        else:
            raise ValueError("model_path must be a string or nn.Module")
        self.model.to(device)
        if external_field is None and _conservative_energy(self.model, energy_key, forces_key):
            self.implemented_properties.append("free_energy")

        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.electric_field_unit = electric_field_unit
        self.charge_unit = charge_unit

        try:
            self.cutoff = self.model.representation.cutoff
        except AttributeError:
            self.cutoff = self.model.models[0].representation.cutoff

        self.atomic_energies = atomic_energies

        self.compute_stress = compute_stress
        self.energy_key = energy_key 
        self.forces_key = forces_key
        self.stress_key = stress_key
        self.charge_key = charge_key
        self.bec_key = bec_key
        self.data_key = data_key
        self.keep_neutral = keep_neutral

        if external_field is not None:
            if isinstance(external_field, float):
                self.external_field = external_field
            else:
                self.external_field = np.array(external_field)
        else:
            self.external_field = None      
        self.output_index = output_index
        
        for param in self.model.parameters():
            param.requires_grad = False

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        if not hasattr(self, "output_index"):
            self.output_index = None

        # Atoms.copy() omits the attached calculator, avoiding recursive reads.
        inference_atoms = atoms.copy()
        # prepare data
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                AtomicData.from_atoms(
                    inference_atoms, cutoff=self.cutoff,
                    data_key=self.data_key,
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        batch_base = next(iter(data_loader)).to(self.device)
        batch = batch_base.clone()
        output = self.model(batch.to_dict(), training=True, compute_stress=self.compute_stress, output_index=self.output_index)
        energy_output = to_numpy(output[self.energy_key])
        forces_output = to_numpy(output[self.forces_key])
        if self.external_field is not None and self.bec_key is not None:
            bec_output = to_numpy(output[self.bec_key])
        # subtract atomic energies if available
        if self.atomic_energies:
            e0 = sum(self.atomic_energies.get(Z, 0) for Z in atoms.get_atomic_numbers())
        else:
            e0 = 0.0
        self.results["energy"] = (energy_output + e0) * self.energy_units_to_eV
        self.results["forces"] = forces_output * self.energy_units_to_eV / self.length_units_to_A
        if self.external_field is not None:        
            if isinstance(self.external_field, float):
                if self.keep_neutral:
                    correction = -np.average(bec_output, axis=0)
                    bec_output += correction
                forces_bec = bec_output * self.external_field * self.electric_field_unit
                self.results["forces"] += forces_bec # bec_output * self.external_field * self.electric_field_unit
            else:
                if self.keep_neutral:
                    correction = -np.average(bec_output, axis=0)
                    bec_output += correction
                forces_bec = bec_output @ self.external_field * self.electric_field_unit
                self.results["forces"] += forces_bec # bec_output * self.external_field * self.electric_field_unit
            
        if self.compute_stress and output[self.stress_key] is not None:
            stress = to_numpy(output[self.stress_key])
            # stress has units eng / len^3:
            self.results["stress"] = (
                stress * (self.energy_units_to_eV / self.length_units_to_A**3)
            )[0]
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])

        if self.charge_key is not None:
            charge_output = to_numpy(output[self.charge_key])
            if charge_output.shape not in ((len(atoms),), (len(atoms), 1)):
                raise ValueError("Charge output must contain one scalar per atom")
            self.results["charges"] = charge_output.reshape(len(atoms)) * self.charge_unit

        self.results["energy"] = float(np.asarray(self.results["energy"], dtype=np.float64).item())     # python float
        self.results["forces"] = np.asarray(self.results["forces"], dtype=np.float64)            # (N,3) float64

        if "free_energy" in self.implemented_properties:
            self.results["free_energy"] = self.results["energy"]

        return self.results


def _conservative_energy(model, energy_key, forces_key):
    """Recognize native energy derivatives and their linear combinations."""
    if isinstance(model, CombinePotential):
        return (model.operation == model.default_operation and bool(model.models)
                and all(energy_key in keys and forces_key in keys
                        and _conservative_energy(child, keys[energy_key], keys[forces_key])
                        for child, keys in zip(model.models, model.potential_keys)))
    if not isinstance(model, NeuralNetworkPotential) or model.do_postprocessing:
        return False
    if any(isinstance(module, DirectForces) for module in model.modules()):
        return False
    # Later energy/force writers would invalidate the derivative relationship.
    for module in reversed(model.output_modules):
        outputs = getattr(module, 'model_outputs', ())
        if energy_key in outputs or forces_key in outputs:
            return (isinstance(module, Forces) and module.calc_forces
                    and module.energy_key == energy_key and module.forces_key == forces_key)
    return False
