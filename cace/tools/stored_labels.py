"""Read stored ASE labels without asking a calculator to evaluate."""

from copy import deepcopy


def read_stored_label(atoms, key, *, locations=("info",)):
    """Return a copy of an exact stored label, or None when it is absent.

    Metadata locations are checked in order. A calculator result is used only
    when that same key is present and the calculator still matches the atoms.
    """
    for location in locations:
        values = getattr(atoms, location)
        if key in values and values[key] is not None:
            return deepcopy(values[key])

    calc = atoms.calc
    if calc is not None and key in calc.results:
        if calc.check_state(atoms):
            raise ValueError(f"Stale calculator result for requested label {key!r}")
        return deepcopy(calc.results[key])
    return None
