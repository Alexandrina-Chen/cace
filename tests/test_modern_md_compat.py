"""Short interface operation only; no stability or density claim."""
import numpy as np
import pytest
import torch
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from cace.calculators import CACECalculator
from compat_helpers import DEVICE, compat_model, synthetic_atoms
from cace.data import AtomicData
from cace.tools import torch_geometric

nhc = pytest.importorskip('ase.md.nose_hoover_chain')


@pytest.mark.parametrize('ensemble', ['nvt','npt'])
@pytest.mark.parametrize('long_range', [False, True])
def test_modern_md(ensemble, long_range):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(1234567)
        a = synthetic_atoms()
        m = compat_model(long_range)
        b = next(iter(torch_geometric.DataLoader([AtomicData.from_atoms(a,6)], batch_size=1)))
        m(b.to(DEVICE).to_dict(), training=True, compute_stress=True)
        a.calc = CACECalculator(m, DEVICE, compute_stress=True)
        MaxwellBoltzmannDistribution(a, temperature_K=300, rng=np.random.RandomState(1234567))
        initial_cell = a.cell.array.copy()
        if ensemble == 'nvt':
            dyn = nhc.NoseHooverChainNVT(a, timestep=.1*units.fs, temperature_K=300, tdamp=10*units.fs)
        else:
            dyn = nhc.IsotropicMTKNPT(a, timestep=.1*units.fs, temperature_K=300,
                                    pressure_au=1e-4, tdamp=10*units.fs, pdamp=100*units.fs)
        assert np.isfinite(dyn.get_conserved_energy())
        dyn.run(3)
        assert np.isfinite(dyn.get_conserved_energy())
        for value in (a.positions, a.get_momenta(), a.cell.array, a.get_forces(), a.get_stress()):
            assert np.isfinite(value).all()
        np.testing.assert_allclose(a.cell.array-np.diag(np.diag(a.cell.array)), 0, atol=1e-12)
        ratios = np.diag(a.cell.array)/np.diag(initial_cell)
        np.testing.assert_allclose(ratios, np.repeat(ratios[0],3), rtol=1e-12)
    finally:
        torch.set_default_dtype(previous)
