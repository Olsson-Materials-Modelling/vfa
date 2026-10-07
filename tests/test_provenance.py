"""Sidecars carry the hashes they claim."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vfa.provenance import RunContext, write_csv, sha256_file, canonical_hash


def test_sidecar_round_trip(tmp_path):
    ctx = RunContext().with_config(dict(b=1, a=2)); t = write_csv(pd.DataFrame(dict(x=[1.0, 2.5])), tmp_path / 'out' / 't.csv', ctx, extra=dict(note='n'))
    m = json.loads((tmp_path / 'out' / 't.csv.meta.json').read_text()); assert m['file_sha256'] == sha256_file(t) and m['configuration_hash'] == canonical_hash(dict(a=2, b=1)) and m['note'] == 'n' and len(m['code_sha256']) == 64
