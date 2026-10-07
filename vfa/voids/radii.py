"""Radii per species: the one setting of the void analysis, named by its convention and saved beside the data."""
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / 'data'


@dataclass(frozen=True)
class Radii:
    table: dict          # species -> radius (A)
    name: str = 'radii'  # centre, covalent, ionic, or your own

    def of(self, symbols):
        """Radius of every atom."""
        return np.array([self.table[s] for s in symbols], float)

    @classmethod
    def centre(cls, symbols):
        """Zero radii: clearance to the atom centres."""
        return cls({s: 0.0 for s in sorted(set(symbols))}, 'centre')

    @classmethod
    def covalent(cls, symbols):
        """Cordero covalent radii: the space outside the bonded network (data/covalent_radii.json)."""
        t = json.loads((DATA / 'covalent_radii.json').read_text())['radii']; return cls({s: t[s] for s in sorted(set(symbols))}, 'covalent')

    @classmethod
    def ionic(cls, symbols):
        """Shannon radii in the charge and coordination of data/ionic_radii.json; give your own table for another state."""
        t = json.loads((DATA / 'ionic_radii.json').read_text())['radii']; return cls({s: t[s]['radius_A'] for s in sorted(set(symbols))}, 'ionic')

    def save(self, path):
        Path(path).write_text(json.dumps({'name': self.name, 'radii_A': self.table}, indent=1) + '\n')

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text()); return cls(d['radii_A'], d['name'])


def masses(symbols):
    """Standard atomic weight (u) of every atom (data/masses.json)."""
    t = json.loads((DATA / 'masses.json').read_text())['masses']; return np.array([t[s] for s in symbols], float)


def mass_density(symbols, cell):
    """Mass density (g cm-3) of a cell from the standard atomic weights."""
    return float(masses(symbols).sum() * 1.66053906660 / abs(np.linalg.det(np.asarray(cell, float))))
