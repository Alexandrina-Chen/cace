import numpy as np
import pytest
import torch
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from cace.data import AtomicData
from cace.tasks import EvaluateTask, get_dataset_from_xyz, load_data_loader
from cace.tools import torch_geometric, batch_to_atoms
from compat_helpers import atoms, model


def labeled():
    a = atoms()
    a.calc = SinglePointCalculator(a, energy=-10., forces=np.ones((2, 3)), stress=np.arange(6.) / 10)
    return a


def batch(a):
    return next(iter(torch_geometric.DataLoader([AtomicData.from_atoms(a, 3)], batch_size=1)))


@pytest.mark.parametrize('form', ['atoms', 'list', 'batch', 'loader'])
def test_forms_offsets_stress_no_output(form, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = labeled()
    b = batch(a)
    inputs = {'atoms': a, 'list': [a], 'batch': b,
              'loader': torch_geometric.DataLoader([AtomicData.from_atoms(a, 3)], batch_size=1)}
    evaluator = EvaluateTask(model(), atomic_energies={1: 4}, energy_units_to_eV=2,
                             length_units_to_A=3)
    # Assert forwarding at model boundary, while still executing the real model.
    requested = []
    evaluator.model.register_forward_pre_hook(lambda m, args, kwargs: requested.append(kwargs['compute_stress']), with_kwargs=True)
    result = evaluator(inputs[form], compute_stress=True)
    assert requested == [True]
    np.testing.assert_allclose(result['energy'], [(np.square(a.positions).sum()/2+8)*2], rtol=1e-6)
    np.testing.assert_allclose(result['forces'], -a.positions*2/3, rtol=1e-6)
    np.testing.assert_allclose(result['stress'][0], a.positions.T@a.positions/a.get_volume()*2/27, rtol=1e-6)
    assert not list(tmp_path.iterdir())
    assert a.calc.results['energy'] == -10


@pytest.mark.parametrize('custom', [False, True])
def test_serialization_preserves_references_and_units(custom, tmp_path):
    a = labeled()
    m = model()
    if custom:
        # Rename the outputs at the interface, preserving their physical meaning.
        def renamed(module, args, output):
            return {'pred_'+k: v for k, v in output.items()}
        m.register_forward_hook(renamed)
    keys = {k+'_key': ('pred_'+k if custom else k) for k in ('energy','forces','stress')}
    evaluator = EvaluateTask(m, energy_units_to_eV=2, length_units_to_A=3, **keys)
    path = tmp_path/'result.xyz'
    result = evaluator([a], compute_stress=True, xyz_output=str(path))
    restored = read(path)
    if custom:
        assert restored.get_potential_energy() == -10
        np.testing.assert_allclose(restored.get_forces(), 1)
        np.testing.assert_allclose(restored.get_stress(), np.arange(6.)/10)
        np.testing.assert_allclose(restored.info['pred_energy'], result['energy'][0], rtol=1e-6)
        np.testing.assert_allclose(restored.arrays['pred_forces'], result['forces'], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(restored.info['pred_stress']).reshape(3,3), result['stress'][0], rtol=1e-6)
    else:
        np.testing.assert_allclose(restored.get_potential_energy(), result['energy'][0], rtol=1e-6)
        np.testing.assert_allclose(restored.get_forces(), result['forces'], rtol=1e-6)
        np.testing.assert_allclose(restored.get_stress(voigt=False), result['stress'][0], rtol=1e-6)
    assert a.calc.results['energy'] == -10
    assert not any(k.startswith('pred_') for k in a.info)


def test_batch_without_offsets_collects_energy():
    a = atoms()
    result = EvaluateTask(model())(batch(a))
    np.testing.assert_allclose(result['energy'], [np.square(a.positions).sum()/2], rtol=1e-6)


def test_output_requires_batch_one(tmp_path):
    with pytest.raises(ValueError, match='Batch size'):
        EvaluateTask(model())([atoms(), atoms()], batch_size=2, xyz_output=str(tmp_path/'bad.xyz'))


@pytest.mark.parametrize('custom', [False, True])
def test_graph_to_atoms_without_predictions(custom, tmp_path):
    a = atoms()
    a.info['energy'] = -1.
    a.arrays['forces'] = np.ones((2,3))
    b = next(iter(torch_geometric.DataLoader([AtomicData.from_atoms(a, 3, data_key={'energy':'energy','forces':'forces'})], batch_size=1)))
    kwargs = {}
    if custom:
        b['pred_energy'] = b['energy'] + 3
        b['pred_forces'] = b['forces'] * 2
        kwargs = {'cace_energy_key':'pred_energy', 'cace_forces_key':'pred_forces'}
    path = tmp_path/'graph.xyz'
    out = batch_to_atoms(b, output_file=str(path), **kwargs)
    assert len(out) == 1
    restored = read(path)
    assert restored.get_potential_energy() == -1.
    np.testing.assert_allclose(restored.get_forces(), 1)
    if custom:
        assert restored.info['pred_energy'] == 2.
        np.testing.assert_allclose(restored.arrays['pred_forces'], 2)


def test_dataset_standard_labels(tmp_path):
    path = tmp_path/'data.xyz'
    write(path, [labeled(), labeled()])
    collection = get_dataset_from_xyz(str(path), 3, valid_path=str(path),
                                      data_key={'energy':'energy','forces':'forces','stress':'stress'})
    b = next(iter(load_data_loader(collection, 'valid', 2)))
    np.testing.assert_allclose(b.energy, -10)
    np.testing.assert_allclose(b.forces, 1)
    assert b.stress.shape == (2,3,3)


@pytest.mark.parametrize('form', ['atoms', 'batch', 'loader'])
def test_output_other_input_forms(form, tmp_path):
    a = labeled()
    inputs = {'atoms': a, 'batch': batch(a),
              'loader': torch_geometric.DataLoader([AtomicData.from_atoms(a, 3)], batch_size=1)}
    path = tmp_path/'prediction.xyz'
    expected = EvaluateTask(model())(inputs[form], xyz_output=str(path))
    restored = read(path)
    np.testing.assert_allclose(restored.get_potential_energy(), expected['energy'][0], rtol=1e-6)
    np.testing.assert_allclose(restored.get_forces(), expected['forces'], rtol=1e-6)


def test_batch_input_is_unchanged_and_stress_false_forwarded():
    b = batch(atoms())
    positions, cell = b.positions.clone(), b.cell.clone()
    evaluator = EvaluateTask(model())
    requested = []
    evaluator.model.register_forward_pre_hook(lambda m, args, kwargs: requested.append(kwargs['compute_stress']), with_kwargs=True)
    result = evaluator(b, compute_stress=False)
    assert requested == [False]
    assert result['stress'] is None
    torch.testing.assert_close(b.positions, positions)
    torch.testing.assert_close(b.cell, cell)
    assert not b.positions.requires_grad


def test_graph_to_atoms_geometry_only():
    restored = batch_to_atoms(batch(atoms()))[0]
    np.testing.assert_allclose(restored.positions, atoms().positions, rtol=1e-6)
    assert 'energy' not in restored.info
    assert 'forces' not in restored.arrays


def test_standard_predictions_drop_reference_free_energy(tmp_path):
    a = labeled()
    a.calc.results['free_energy'] = -11.
    path = tmp_path/'prediction.xyz'
    EvaluateTask(model())([a], xyz_output=str(path))
    restored = read(path)
    assert 'free_energy' not in restored.calc.results
    assert a.calc.results['free_energy'] == -11.
