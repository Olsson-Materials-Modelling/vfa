"""Provenance of written tables: a run context in a sidecar beside each file and in Parquet schema metadata."""
import hashlib, json, platform, sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from importlib import metadata as _metadata
from pathlib import Path
import numpy as np
from . import __version__

LIBRARIES = ('numpy', 'scipy', 'scikit-image', 'pandas')


def sha256_file(path, chunk=1 << 20):
    """Streamed SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''): h.update(block)
    return h.hexdigest()


def _plain(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, Path): return o.as_posix()
    raise TypeError(f'{type(o).__name__} is not JSON-serialisable')


def _string_keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if not isinstance(k, str): raise TypeError(f'key {k!r} is not a string')
            _string_keys(v)
    elif isinstance(o, (list, tuple)):
        for v in o: _string_keys(v)


def _json(obj, name, **kw):
    try: _string_keys(obj); return json.dumps(obj, sort_keys=True, allow_nan=False, default=_plain, **kw)
    except (TypeError, ValueError) as e: raise ValueError(f'{name}: JSON values with string keys and finite numbers are needed ({e})') from None


def canonical_hash(obj):
    """Order-independent hash of a configuration of JSON values and numpy numbers or arrays."""
    return hashlib.sha256(_json(obj, 'canonical_hash', separators=(',', ':')).encode()).hexdigest()


def code_fingerprint(package=Path(__file__).resolve().parent):
    """SHA-256 over the relative names and contents of the modules and data files of the package."""
    package = Path(package); h = hashlib.sha256(); files = {f.relative_to(package).as_posix(): f for f in [*package.rglob('*.py'), *(package / 'data').glob('*')] if f.is_file()}
    for name in sorted(files): h.update(name.encode()); h.update(sha256_file(files[name]).encode())
    return h.hexdigest()


def library_versions(names=LIBRARIES):
    """Installed version of each named distribution, sorted by name; absent ones are left out."""
    out = []
    for name in sorted(names):
        try: out.append((name, _metadata.version(name)))
        except _metadata.PackageNotFoundError: pass
    return tuple(out)


@dataclass(frozen=True)
class RunContext:
    """Code fingerprint, configuration and its hash, input hash, library versions, time and platform of one run."""
    code_sha256: str = field(default_factory=code_fingerprint)
    config_hash: str = ''
    config: str = ''
    inputs_hash: str = ''
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec='seconds'))
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    vfa_version: str = __version__
    libraries: tuple = field(default_factory=library_versions)

    def with_config(self, config):
        """The same context with a JSON-serialisable configuration and its hash."""
        text = _json(config, 'with_config'); return replace(self, config_hash=canonical_hash(config), config=text)

    def with_inputs(self, paths):
        """The same context with one hash over the given input files by name and content."""
        return replace(self, inputs_hash=canonical_hash(sorted((Path(p).name, sha256_file(p)) for p in paths)))

    def metadata(self):
        """The context as a dict: hashes, the configuration, the time and one version entry per recorded library."""
        return dict(code_sha256=self.code_sha256, configuration=json.loads(self.config) if self.config else None, configuration_hash=self.config_hash, inputs_hash=self.inputs_hash, created_utc=self.created_utc, python_version=self.python, platform=self.platform, vfa_version=self.vfa_version, **{f"{name.replace('-', '_')}_version": v for name, v in self.libraries})


def _index_as_columns(df):
    cols = getattr(df, 'columns', None); cols = (getattr(df, 'name', None),) if cols is None else cols
    names = [n for n in getattr(getattr(df, 'index', None), 'names', None) or () if n is not None and n not in cols]
    return df.reset_index(level=names) if names else df


def write_sidecar(target, ctx, extra=None):
    """<target>.meta.json: the context, the file's hash and size, and any extra fields."""
    target = Path(target); meta = dict(ctx.metadata(), file=target.name, file_sha256=sha256_file(target), file_bytes=target.stat().st_size); clash = sorted(set(extra or {}) & set(meta))
    if clash: raise ValueError(f'write_sidecar: extra fields may not replace provenance fields {clash}')
    meta.update(extra or {})
    side = target.with_suffix(target.suffix + '.meta.json'); side.write_text(_json(meta, 'write_sidecar', indent=2)); return side


def write_csv(df, target, ctx, extra=None, float_format='%.17g'):
    """CSV of a table, named index levels kept, and a provenance sidecar."""
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True); _index_as_columns(df).to_csv(target, index=False, float_format=float_format); write_sidecar(target, ctx, extra); return target


def write_parquet(df, target, ctx, extra=None):
    """Parquet of a DataFrame, named index levels kept, provenance in its metadata and a sidecar."""
    try: import pyarrow as pa, pyarrow.parquet as pq
    except ImportError as e: raise ImportError('write_parquet: needs pyarrow, which is not installed (pip install pyarrow)') from e
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True); t = pa.Table.from_pandas(_index_as_columns(df), preserve_index=False); meta = {k: v if isinstance(v, str) else _json(v, 'write_parquet') for k, v in dict(ctx.metadata(), **(extra or {})).items()}
    t = t.replace_schema_metadata({**(t.schema.metadata or {}), **{k.encode(): v.encode() for k, v in meta.items()}}); pq.write_table(t, target, compression='zstd'); write_sidecar(target, ctx, extra); return target
