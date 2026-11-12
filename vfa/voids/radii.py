"""Radii per species, named by their convention and saved beside the data; standard atomic weights and mass density."""
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ..units import U_PER_A3_TO_G_PER_CM3

DATA = Path(__file__).resolve().parents[1] / 'data'


def _known(table, symbols, name):
    symbols = list(symbols); missing = sorted({str(s) for s in symbols if s not in table})
    if missing: raise ValueError(f'{name}: no value for species {missing}')
    return symbols


@dataclass(frozen=True)
class Radii:
    """Radius (A) of each species and the name of the convention: centre, covalent, ionic, or your own."""
    table: dict
    name: str = 'radii'

    def of(self, symbols):
        """Radius (A) of every atom."""
        return np.array([self.table[s] for s in _known(self.table, symbols, f'Radii.of ({self.name})')], float)

    @classmethod
    def centre(cls, symbols):
        """Zero radii: clearance to the atom centres."""
        return cls({s: 0.0 for s in sorted(set(symbols))}, 'centre')

    @classmethod
    def covalent(cls, symbols):
        """Cordero covalent radii (data/covalent_radii.json)."""
        t = json.loads((DATA / 'covalent_radii.json').read_text())['radii']; return cls({s: t[s] for s in sorted(set(_known(t, symbols, 'Radii.covalent')))}, 'covalent')

    @classmethod
    def ionic(cls, symbols):
        """Shannon radii in the charge and coordination of data/ionic_radii.json."""
        t = json.loads((DATA / 'ionic_radii.json').read_text())['radii']; return cls({s: t[s]['radius_A'] for s in sorted(set(_known(t, symbols, 'Radii.ionic')))}, 'ionic')

    def save(self, path):
        """The table and its name as JSON."""
        Path(path).write_text(json.dumps({'name': self.name, 'radii_A': self.table}, indent=1) + '\n')

    @classmethod
    def load(cls, path):
        """Radii from a JSON file written by save."""
        d = json.loads(Path(path).read_text()); return cls(d['radii_A'], d['name'])


def masses(symbols):
    """Standard atomic weight (u) of every atom (data/masses.json)."""
    t = json.loads((DATA / 'masses.json').read_text())['masses']; return np.array([t[s] for s in _known(t, symbols, 'masses')], float)


def mass_density(symbols, cell):
    """Mass density (g cm-3) of a cell from the standard atomic weights."""
    return float(masses(symbols).sum() * U_PER_A3_TO_G_PER_CM3 / abs(np.linalg.det(np.asarray(cell, float))))
