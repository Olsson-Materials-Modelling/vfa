"""Provenance of every written table: a run context (code fingerprint, configuration hash, input manifest hash, time, platform) in a sidecar beside each CSV and in the schema metadata of each Parquet file."""
import hashlib, json, platform, sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


def sha256_file(path, chunk=1 << 20):
    """Streamed SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''): h.update(block)
    return h.hexdigest()


def canonical_hash(obj):
    """Order-independent hash of a JSON-serialisable configuration."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def code_fingerprint(package=Path(__file__).resolve().parent):
    """SHA-256 over the sorted contents of every module of the package: changes whenever the code changes."""
    h = hashlib.sha256()
    for py in sorted(package.rglob('*.py')): h.update(py.relative_to(package).as_posix().encode()); h.update(sha256_file(py).encode())
    return h.hexdigest()


@dataclass(frozen=True)
class RunContext:
    code_sha256: str = field(default_factory=code_fingerprint)
    config_hash: str = ''
    inputs_hash: str = ''
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec='seconds'))
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)

    def with_config(self, config):
        """The same context with the hash of a configuration."""
        return RunContext(self.code_sha256, canonical_hash(config), self.inputs_hash, self.created_utc, self.python, self.platform)

    def with_inputs(self, paths):
        """The same context with one hash over the given input files."""
        return RunContext(self.code_sha256, self.config_hash, canonical_hash([(str(p), sha256_file(p)) for p in paths]), self.created_utc, self.python, self.platform)

    def metadata(self):
        return dict(code_sha256=self.code_sha256, configuration_hash=self.config_hash, inputs_hash=self.inputs_hash, created_utc=self.created_utc, python_version=self.python, platform=self.platform)


def write_sidecar(target, ctx, extra=None):
    """<target>.meta.json: the context, the file's hash and size, and any extra fields."""
    target = Path(target); meta = dict(ctx.metadata(), file=target.name, file_sha256=sha256_file(target), file_bytes=target.stat().st_size, **(extra or {}))
    side = target.with_suffix(target.suffix + '.meta.json'); side.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str)); return side


def write_csv(df, target, ctx, extra=None, float_format='%.17g'):
    """CSV with full-precision floats and a provenance sidecar."""
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True); df.to_csv(target, index=False, float_format=float_format); write_sidecar(target, ctx, extra); return target


def write_parquet(df, target, ctx, extra=None):
    """Parquet with the provenance in the schema metadata and a sidecar."""
    import pyarrow as pa, pyarrow.parquet as pq
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True); t = pa.Table.from_pandas(df, preserve_index=False); meta = {k: str(v) for k, v in dict(ctx.metadata(), **(extra or {})).items()}
    t = t.replace_schema_metadata({**(t.schema.metadata or {}), **{k.encode(): v.encode() for k, v in meta.items()}}); pq.write_table(t, target, compression='zstd'); write_sidecar(target, ctx, extra); return target
