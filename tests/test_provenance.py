"""Sidecars carry the hashes, versions and nothing of the machine's paths."""
import hashlib, json, shutil, subprocess, sys
from importlib import metadata
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import vfa
from vfa.provenance import RunContext, write_csv, write_parquet, canonical_hash, code_fingerprint


def test_sidecar_round_trip(tmp_path):
    ctx = RunContext().with_config(dict(b=1, a=2)); t = write_csv(pd.DataFrame(dict(x=[1.0, 2.5])), tmp_path / 'out' / 't.csv', ctx, extra=dict(note='n'))
    m = json.loads((tmp_path / 'out' / 't.csv.meta.json').read_text()); assert m['file_sha256'] == hashlib.sha256(t.read_bytes()).hexdigest() and m['configuration_hash'] == canonical_hash(dict(a=2, b=1)) and m['note'] == 'n' and len(m['code_sha256']) == 64


def test_inputs_hash_ignores_order_and_spelling(tmp_path, monkeypatch):
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'; a.write_text('1'); b.write_text('2'); ctx = RunContext(); h = ctx.with_inputs([a, b]).inputs_hash
    monkeypatch.chdir(tmp_path); assert h == ctx.with_inputs([b, a]).inputs_hash == ctx.with_inputs(['b.json', Path('.') / 'a.json']).inputs_hash != ctx.with_inputs([a]).inputs_hash
    b.write_text('3'); assert ctx.with_inputs([a, b]).inputs_hash != h


def test_code_fingerprint_covers_the_data_files(tmp_path):
    pkg = tmp_path / 'vfa'; shutil.copytree(Path(vfa.__file__).parent, pkg, ignore=shutil.ignore_patterns('__pycache__')); f0 = code_fingerprint(pkg); assert f0 == code_fingerprint(Path(vfa.__file__).parent)
    t = json.loads((pkg / 'data/covalent_radii.json').read_text()); t['radii']['Si'] = 9.99; (pkg / 'data/covalent_radii.json').write_text(json.dumps(t)); assert code_fingerprint(pkg) != f0


def test_context_records_versions_and_no_paths(tmp_path):
    src = tmp_path / 'input.xyz'; src.write_text('1\n\nA 0 0 0\n'); ctx = RunContext().with_inputs([src.resolve()]); write_csv(pd.DataFrame(dict(x=[1.0])), tmp_path / 't.csv', ctx)
    text = (tmp_path / 't.csv.meta.json').read_text(); m = json.loads(text); assert m['vfa_version'] == vfa.__version__ and str(tmp_path) not in text and str(Path.home()) not in text and not [k for k, v in m.items() if str(v).startswith('/')]
    for lib in ('numpy', 'scipy', 'scikit-image', 'pandas'):
        try: assert metadata.version(lib) == m[f"{lib.replace('-', '_')}_version"]
        except metadata.PackageNotFoundError: assert f"{lib.replace('-', '_')}_version" not in m


def test_library_imports_no_pandas():
    out = subprocess.run([sys.executable, '-c', f'import sys; sys.path.insert(0, {str(ROOT)!r}); import vfa.provenance; print("pandas" in sys.modules)'], capture_output=True, text=True, check=True); assert out.stdout.strip() == 'False'


def test_write_csv_takes_any_table(tmp_path):
    class Table:
        def to_csv(self, path, index, float_format): Path(path).write_text('x\n1\n')
    t = write_csv(Table(), tmp_path / 't.csv', RunContext()); assert t.read_text() == 'x\n1\n' and json.loads((tmp_path / 't.csv.meta.json').read_text())['file_sha256'] == hashlib.sha256(t.read_bytes()).hexdigest()


def test_write_parquet_names_pyarrow_when_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyarrow', None); monkeypatch.setitem(sys.modules, 'pyarrow.parquet', None)
    with pytest.raises(ImportError, match='write_parquet: needs pyarrow'): write_parquet(pd.DataFrame(dict(x=[1.0])), tmp_path / 't.parquet', RunContext())


def test_write_parquet_carries_the_context(tmp_path):
    pq = pytest.importorskip('pyarrow.parquet'); ctx = RunContext(); t = write_parquet(pd.DataFrame(dict(x=[1.0, 2.0])), tmp_path / 't.parquet', ctx)
    meta = pq.read_schema(t).metadata; assert meta[b'code_sha256'].decode() == ctx.code_sha256 and meta[b'numpy_version'] and (tmp_path / 't.parquet.meta.json').exists()


def test_the_configuration_itself_is_recorded(tmp_path):
    cfg = dict(cutoff_A=2.08, radii='covalent', radii_A={'A': 1.1, 'B': 0.7}); t = write_csv(pd.DataFrame(dict(x=[1.0])), tmp_path / 't.csv', RunContext().with_config(cfg))
    m = json.loads((tmp_path / 't.csv.meta.json').read_text()); assert m['configuration'] == cfg and m['configuration_hash'] == canonical_hash(cfg) and t.exists()
    write_csv(pd.DataFrame(dict(x=[1.0])), tmp_path / 'u.csv', RunContext()); assert json.loads((tmp_path / 'u.csv.meta.json').read_text())['configuration'] is None


def test_numpy_values_hash_by_value():
    assert canonical_hash(dict(r=np.array([1.0, 2.0]), n=np.int64(24))) == canonical_hash(dict(r=[1.0, 2.0], n=24))
    assert canonical_hash(dict(r=np.array([1.0, 2.0]))) != canonical_hash(dict(r=np.array([1.0, 2.0 + 1e-9])))


def test_configurations_must_be_strict_json():
    for bad in (dict(r=float('nan')), dict(r=np.inf), {('A', 'B'): 2.0}, dict(f=object())):
        with pytest.raises(ValueError, match='with_config'): RunContext().with_config(bad)


def test_sidecar_extras_must_be_strict_json(tmp_path):
    with pytest.raises(ValueError, match='write_sidecar'): write_csv(pd.DataFrame(dict(x=[1.0])), tmp_path / 't.csv', RunContext(), extra=dict(stat=float('nan')))


def test_hash_keys_must_be_strings_and_extra_fields_keep_provenance(tmp_path):
    with pytest.raises(ValueError, match='canonical_hash'): canonical_hash({1: 'a'})
    with pytest.raises(ValueError, match='may not replace provenance fields'): write_csv(pd.DataFrame(dict(x=[1.0])), tmp_path / 't.csv', RunContext(), extra=dict(file_sha256='x'))


def test_write_csv_and_parquet_write_named_index_levels_as_columns(tmp_path):
    df = pd.DataFrame(dict(cell=['c1', 'c1', 'c2'], y=[1.0, 2.0, 5.0])); g = df.groupby('cell').mean()
    assert write_csv(g, tmp_path / 'g.csv', RunContext()).read_text().splitlines() == ['cell,y', 'c1,1.5', 'c2,5'] and write_csv(df[df.y > 1], tmp_path / 'f.csv', RunContext()).read_text().splitlines()[0] == 'cell,y'
    for t in (df.set_index('cell', drop=False), df.set_index(['cell', df.index])): assert write_csv(t, tmp_path / 't.csv', RunContext()).read_text().splitlines()[0] == 'cell,y'
    assert write_csv(df.groupby('cell').cell.count(), tmp_path / 's.csv', RunContext()).read_text().splitlines() == ['cell', '2', '1']
    pytest.importorskip('pyarrow'); assert pd.read_parquet(write_parquet(g, tmp_path / 'g.parquet', RunContext())).columns.tolist() == ['cell', 'y']
