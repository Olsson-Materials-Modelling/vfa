"""Total-energy bookkeeping of a charge-trapping event from three calculations of one cell."""
import numpy as np

HARTREE_EV = 27.211386245988


def trapping_energies(e_neutral, e_charged_relaxed, e_charged_vertical=np.nan, in_hartree=True):
    """Addition (relaxed charged minus neutral), vertical addition (charged at the neutral geometry minus neutral) and relaxation (vertical minus relaxed) energies in eV; the relaxation energy is alignment-free, the addition energies need a finite-size correction."""
    s = HARTREE_EV if in_hartree else 1.0; add = (e_charged_relaxed - e_neutral) * s; vert = (e_charged_vertical - e_neutral) * s
    return dict(addition_eV=float(add), vertical_addition_eV=float(vert), relaxation_eV=float(vert - add))


def closes_cycle(r, tol=1e-9):
    """addition - vertical addition = -relaxation, within tol."""
    return bool(abs(r['addition_eV'] - r['vertical_addition_eV'] + r['relaxation_eV']) < tol)
