"""Scalar external LES integration; no compiled/multipolar validation claim."""
import copy
import numpy as np
import pytest
import torch
from cace.data import AtomicData
from cace.tools import torch_geometric
from cace.calculators import CACECalculator
from cace.tasks import EvaluateTask
from cace.modules import LesWrapper
from compat_helpers import DEVICE, compat_model, synthetic_atoms

pytest.importorskip('les')


@pytest.fixture(autouse=True)
def precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1234567)
    yield
    torch.set_default_dtype(previous)


def graph(a, device=DEVICE):
    return next(iter(torch_geometric.DataLoader([AtomicData.from_atoms(a, 6)], batch_size=1))).to(device)


@pytest.mark.parametrize('long_range', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_fresh_model_optimizer_and_roundtrip(long_range, dtype, tmp_path):
    torch.set_default_dtype(dtype)
    a = synthetic_atoms()
    m = compat_model(long_range)
    # Initialize all lazy heads before constructing the optimizer.
    out = m(graph(a).to_dict(), training=True, compute_stress=True)
    for key in ('energy','forces','stress','descriptor'):
        assert torch.isfinite(out[key]).all()
    if long_range:
        torch.testing.assert_close(out['energy'], out['sr_energy']+out['lr_energy'])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    assert {id(p) for p in m.parameters() if p.requires_grad} == {id(p) for g in opt.param_groups for p in g['params'] if p.requires_grad}
    tracked = dict(m.named_parameters())
    before = {k:v.detach().clone() for k,v in tracked.items() if 'les.atomwise' in k}
    loss = sum(out[k].square().mean() for k in ('energy','forces','stress'))
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    if long_range:
        assert before
        assert all(tracked[k].grad is not None and torch.isfinite(tracked[k].grad).all() for k in before)
    opt.step()
    if long_range:
        assert any(not torch.equal(before[k], tracked[k]) for k in before)
    path = tmp_path/'model.pt'
    torch.save(m, path)
    expected = EvaluateTask(copy.deepcopy(m), device=DEVICE)(a, compute_stress=True)
    actual = EvaluateTask(str(path), device=DEVICE)(a, compute_stress=True)
    for k in ('energy','forces','stress'):
        np.testing.assert_allclose(actual[k], expected[k], rtol=1e-12, atol=1e-12)
    a.calc = CACECalculator(str(path), DEVICE, compute_stress=True)
    np.testing.assert_allclose(a.get_potential_energy(force_consistent=True), expected['energy'][0], rtol=1e-12)
    np.testing.assert_allclose(a.get_forces(), expected['forces'], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a.get_stress(voigt=False), expected['stress'][0], rtol=1e-12, atol=1e-12)


def test_external_force_stress_central_differences():
    a = synthetic_atoms()
    m = compat_model(True)
    m(graph(a).to_dict(), training=True, compute_stress=True)
    calc = CACECalculator(m, DEVICE, compute_stress=True)
    a.calc = calc
    forces = a.get_forces()
    stress = a.get_stress(voigt=False)
    h = 1e-5
    for i in range(len(a)):
        for j in range(3):
            plus, minus = a.copy(), a.copy()
            plus.positions[i,j] += h
            minus.positions[i,j] -= h
            plus.calc = calc
            ep = plus.get_potential_energy()
            minus.calc = calc
            em = minus.get_potential_energy()
            np.testing.assert_allclose(forces[i,j], -(ep-em)/(2*h), rtol=2e-5, atol=2e-7)
    for i,j in [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]:
        strain = np.zeros((3,3))
        strain[i,j] = 1 if i == j else .5
        strain[j,i] = strain[i,j]
        plus, minus = a.copy(), a.copy()
        plus.set_cell(a.cell.array@(np.eye(3)+h*strain), scale_atoms=True)
        minus.set_cell(a.cell.array@(np.eye(3)-h*strain), scale_atoms=True)
        plus.calc = calc
        ep = plus.get_potential_energy()
        minus.calc = calc
        em = minus.get_potential_energy()
        np.testing.assert_allclose(np.sum(stress*strain), (ep-em)/(2*h*a.get_volume()), rtol=2e-5, atol=2e-8)


def test_wrapper_descriptor_list_supplied_charges_and_response():
    a = synthetic_atoms()
    g = graph(a).to_dict()
    g['positions'].requires_grad_(True)
    g['a'] = torch.randn(len(a), 2, device=DEVICE)
    g['b'] = torch.randn(len(a), 3, device=DEVICE)
    wrapper = LesWrapper(feature_key=['a','b'], compute_bec=True).to(DEVICE)
    out = wrapper(dict(g))
    assert torch.isfinite(out['LES_energy']).all()
    assert torch.isfinite(out['LES_BEC']).all()
    direct = LesWrapper(feature_key=None, compute_bec=True).to(DEVICE)
    provided = dict(g, LES_charge=out['LES_charge'])
    result = direct(provided)
    torch.testing.assert_close(result['LES_energy'], out['LES_energy'])
    torch.testing.assert_close(result['LES_BEC'], out['LES_BEC'])


def test_optional_response_arguments_match_direct_les():
    a = synthetic_atoms()
    g = graph(a).to_dict()
    g['positions'].requires_grad_(True)
    n = len(a)
    g.update(q=torch.linspace(-.2,.2,n, device=DEVICE).reshape(n,1),
             u=torch.full((n,1,3), .001, device=DEVICE), quad=torch.zeros(n,3,3, device=DEVICE),
             alpha=torch.full((n,1), .001, device=DEVICE), kappa=torch.full((n,1), .001, device=DEVICE))
    wrapper = LesWrapper(feature_key=None, charge_key='q', dipole_key='u',
                         quad_key='quad', alpha_key='alpha', kappa_key='kappa',
                         atomic_number_key='atomic_numbers', compute_bec=True).to(DEVICE)
    direct = wrapper.les(positions=g['positions'], cell=g['cell'].reshape(-1,3,3),
                         batch=g['batch'], latent_charges=g['q'], latent_dipoles=g['u'],
                         latent_quads=g['quad'], latent_alphas=g['alpha'], latent_kappas=g['kappa'],
                         atomic_numbers=g['atomic_numbers'], compute_energy=True, compute_bec=True)
    result = wrapper(dict(g))
    torch.testing.assert_close(result['LES_energy'], direct['E_lr'])
    torch.testing.assert_close(result['LES_BEC'], direct['BEC'])
    assert torch.isfinite(result['LES_energy']).all()
    assert torch.isfinite(result['LES_BEC']).all()
