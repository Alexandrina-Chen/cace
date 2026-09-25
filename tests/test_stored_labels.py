import numpy as np
import pytest
import pickle
from types import SimpleNamespace
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from cace.data.atomic_data import AtomicData
from cace.tools.utils import compute_average_E0s


def water():
    return Atoms("HO", positions=[[0, 0, 0], [0, 0, 1]])


def test_reader_uses_exact_keys_and_requested_metadata_locations():
    from cace.tools.stored_labels import read_stored_label
    atoms = water()
    atoms.info["ref_energy"] = 3.0
    atoms.info["ref_forces"] = np.full((2, 3), 9.0)
    atoms.arrays["ref_forces"] = np.full((2, 3), 2.0)
    atoms.info["dipole"] = np.array([1.0, 2.0, 3.0])
    atoms.arrays["dipole"] = np.full((2, 3), 8.0)

    assert read_stored_label(atoms, "energy") is None
    assert read_stored_label(atoms, "ref_energy") == 3.0
    np.testing.assert_array_equal(
        read_stored_label(atoms, "ref_forces", locations=("arrays",)),
        np.full((2, 3), 2.0),
    )
    np.testing.assert_array_equal(
        read_stored_label(atoms, "dipole", locations=("info", "arrays")),
        [1.0, 2.0, 3.0],
    )
    atoms.info["dipole"] = None
    np.testing.assert_array_equal(
        read_stored_label(atoms, "dipole", locations=("info", "arrays")),
        np.full((2, 3), 8.0),
    )


def test_reader_copies_stored_arrays():
    from cace.tools.stored_labels import read_stored_label
    atoms = water()
    atoms.arrays["ref_forces"] = np.ones((2, 3))
    value = read_stored_label(atoms, "ref_forces", locations=("arrays",))
    value[0, 0] = 10.0
    assert atoms.arrays["ref_forces"][0, 0] == 1.0


def test_reader_uses_valid_calculator_results_without_evaluation():
    from cace.tools.stored_labels import read_stored_label
    atoms = water()
    atoms.calc = SinglePointCalculator(
        atoms, energy=4.0, forces=np.full((2, 3), 2.0), stress=np.arange(6.0)
    )
    assert read_stored_label(atoms, "energy") == 4.0
    assert read_stored_label(atoms, "free_energy") is None
    np.testing.assert_array_equal(
        read_stored_label(atoms, "forces", locations=("arrays",)),
        np.full((2, 3), 2.0),
    )
    np.testing.assert_array_equal(read_stored_label(atoms, "stress"), np.arange(6.0))


def test_reader_rejects_stale_requested_calculator_result():
    from cace.tools.stored_labels import read_stored_label
    atoms = water()
    atoms.calc = SinglePointCalculator(atoms, energy=4.0)
    atoms.positions[0, 0] += 0.1
    with pytest.raises(ValueError, match="[Ss]tale.*energy"):
        read_stored_label(atoms, "energy")
    assert read_stored_label(atoms, "forces", locations=("arrays",)) is None


def test_reader_metadata_wins_over_stale_calculator():
    from cace.tools.stored_labels import read_stored_label
    atoms = water()
    atoms.calc = SinglePointCalculator(atoms, energy=4.0)
    atoms.positions[0, 0] += 0.1
    atoms.info["energy"] = 7.0
    assert read_stored_label(atoms, "energy") == 7.0


def test_reader_does_not_calculate_missing_result():
    from cace.tools.stored_labels import read_stored_label

    class ExplodingCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, *args, **kwargs):
            raise AssertionError("calculator evaluated")

    atoms = water()
    atoms.calc = ExplodingCalculator()
    assert read_stored_label(atoms, "energy") is None
    assert read_stored_label(atoms, "forces", locations=("arrays",)) is None


def test_atomic_data_custom_mapping_is_independent_and_preserves_sources():
    atoms = water()
    atoms.info["my_energy"] = 5.0
    atoms.arrays["my_forces"] = np.full((2, 3), 2.0)
    atoms.info["my_stress"] = np.arange(6.0)
    atoms.info["my_dipole"] = np.array([1.0, 2.0, 3.0])
    atoms.arrays["initial_charges"] = np.array([0.2, -0.2])
    atoms.set_constraint(FixAtoms(indices=[0]))
    custom = AtomicData.from_atoms(
        atoms, cutoff=2.0,
        data_key={"energy": "my_energy", "forces": "my_forces",
                  "stress": "my_stress", "dipole": "my_dipole",
                  "charges": "charges"},
        atomic_energies={1: 1.0, 8: 2.0},
    )
    assert custom.energy.item() == pytest.approx(2.0)
    assert custom.forces[0, 0].item() == pytest.approx(2.0)
    assert custom.stress.shape == (1, 3, 3)
    assert custom.dipole.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert custom.charges is None
    assert atoms.info["my_energy"] == 5.0

    default = AtomicData.from_atoms(atoms, cutoff=2.0)
    assert default.energy is None
    assert default.forces is None
    initial = AtomicData.from_atoms(
        atoms, cutoff=2.0, data_key={"charges": "initial_charges"}
    )
    assert initial.charges.tolist() == pytest.approx([0.2, -0.2])
    assert atoms.arrays["my_forces"][0, 0] == 2.0


def test_atomic_data_reads_matching_calculator_results():
    atoms = water()
    atoms.calc = SinglePointCalculator(
        atoms, energy=5.0, forces=np.ones((2, 3)), stress=np.arange(6.0)
    )
    atoms.set_constraint(FixAtoms(indices=[0]))
    data = AtomicData.from_atoms(
        atoms, cutoff=2.0,
        data_key={"energy": "energy", "forces": "forces"},
    )
    assert data.energy.item() == pytest.approx(5.0)
    assert data.forces[0, 0].item() == pytest.approx(1.0)
    assert data.stress.shape == (1, 3, 3)


def test_e0_uses_exact_energy_key_and_requires_each_energy():
    hydrogen = Atoms("H", positions=[[0, 0, 0]])
    oxygen = Atoms("O", positions=[[0, 0, 0]])
    hydrogen.calc = SinglePointCalculator(hydrogen, energy=2.0)
    oxygen.info["target"] = 8.0
    with pytest.raises(ValueError, match="target"):
        compute_average_E0s([hydrogen, oxygen], zs=[1, 8], energy_key="target")
    hydrogen.info["target"] = 3.0
    assert compute_average_E0s([hydrogen, oxygen], zs=[1, 8], energy_key="target") == pytest.approx({1: 3.0, 8: 8.0})
    oxygen.calc = SinglePointCalculator(oxygen, energy=7.0)
    assert compute_average_E0s([hydrogen, oxygen], zs=[1, 8], energy_key="energy") == pytest.approx({1: 2.0, 8: 7.0})


def test_train_cli_computes_e0_for_configured_key_even_with_old_cache(monkeypatch, tmp_path):
    import scripts.train as train

    class StopAfterDataset(Exception):
        pass

    hydrogen = Atoms("H", positions=[[0, 0, 0]], info={"target": 3.0})
    oxygen = Atoms("O", positions=[[0, 0, 0]], info={"target": 8.0})
    with (tmp_path / "avge0.pkl").open("wb") as handle:
        pickle.dump({1: -100.0, 8: -100.0}, handle)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "parse_arguments", lambda: SimpleNamespace(
        train_path="frames.xyz", zs=[1, 8], energy_key="target",
        forces_key="forces", prefix="test", use_device="cpu",
        valid_fraction=0.1, cutoff=2.0,
    ))
    monkeypatch.setattr(train, "setup_logger", lambda **kwargs: None)
    monkeypatch.setattr(train, "init_device", lambda device: device)
    monkeypatch.setattr(train.ase.io, "read", lambda *args: [hydrogen, oxygen])

    def check_dataset(**kwargs):
        assert kwargs["atomic_energies"] == pytest.approx({1: 3.0, 8: 8.0})
        raise StopAfterDataset

    monkeypatch.setattr(train.cace.tasks, "get_dataset_from_xyz", check_dataset)
    with pytest.raises(StopAfterDataset):
        train.main()
