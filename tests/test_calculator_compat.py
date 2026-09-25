import numpy as np
import pytest
import torch
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.calculator import PropertyNotImplementedError
from cace.calculators import CACECalculator
from cace.modules import DirectForces
from compat_helpers import atoms, model


def test_energy_force_stress_units_free_energy_and_charges():
    a = atoms()
    a.calc = CACECalculator(model(), 'cpu', compute_stress=True,
                            charge_key='charges', charge_unit=2,
                            energy_units_to_eV=2, length_units_to_A=3,
                            atomic_energies={1: 4})
    expected = (np.square(a.positions).sum() / 2 + 8) * 2
    assert isinstance(a.get_potential_energy(), float)
    assert a.get_potential_energy(force_consistent=True) == pytest.approx(expected)
    np.testing.assert_allclose(a.get_forces(), -a.positions * 2 / 3, rtol=1e-6)
    expected_stress = a.positions.T @ a.positions / a.get_volume() * 2 / 27
    np.testing.assert_allclose(a.get_stress(voigt=False), expected_stress, rtol=1e-6)
    assert a.get_charges().shape == (len(a),)
    np.testing.assert_allclose(a.get_charges(), .5)


def test_disabled_stress_and_nonconservative_properties():
    a = atoms()
    a.calc = CACECalculator(model(), 'cpu')
    assert 'stress' not in a.calc.implemented_properties
    with pytest.raises(PropertyNotImplementedError):
        a.get_stress()
    a.calc = CACECalculator(model(), 'cpu', external_field=0.1)
    assert 'free_energy' not in a.calc.implemented_properties
    direct = model()
    direct.output_modules.append(DirectForces())
    calc = CACECalculator(direct, 'cpu')
    assert 'free_energy' not in calc.implemented_properties


def test_calculator_free_graph_and_cache_invalidation(monkeypatch):
    a = atoms()
    calc = CACECalculator(model(), 'cpu', compute_stress=True,
                          data_key={'energy': 'energy', 'forces': 'forces'})
    a.calc = calc
    calls = []
    from cace.data import AtomicData
    original = AtomicData.from_atoms
    def graph(atoms, *args, **kwargs):
        assert atoms.calc is None
        calls.append(atoms.positions.copy())
        return original(atoms, *args, **kwargs)
    monkeypatch.setattr(AtomicData, 'from_atoms', graph)
    a.get_forces()
    a.get_potential_energy()
    a.get_stress()
    assert len(calls) == 1
    a.positions[0, 0] += .001
    a.get_forces()
    assert len(calls) == 2
    a.set_cell(a.cell * 1.0001, scale_atoms=True)
    a.get_stress()
    assert len(calls) == 3


def test_trusted_model_path(tmp_path):
    path = tmp_path / 'model.pt'
    torch.save(model(), path)
    a = atoms()
    a.calc = CACECalculator(str(path), 'cpu')
    assert np.isfinite(a.get_potential_energy())


def test_free_energy_requires_conservative_composition():
    from cace.models import CombinePotential
    from cace.modules import FeatureAdd
    keys = [{'energy':'energy', 'forces':'forces'}]*2
    nonlinear = CombinePotential([model(), model()], keys, operation=lambda values: torch.stack(values).square().sum(0))
    assert 'free_energy' not in CACECalculator(nonlinear, 'cpu').implemented_properties
    linear = CombinePotential([model(), model()], keys)
    assert 'free_energy' in CACECalculator(linear, 'cpu').implemented_properties
    changed_energy = model()
    changed_energy.output_modules.append(FeatureAdd(['energy','energy'], 'energy'))
    assert 'free_energy' not in CACECalculator(changed_energy, 'cpu').implemented_properties
