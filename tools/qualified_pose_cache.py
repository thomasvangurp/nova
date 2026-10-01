"""Scientific pose memoization for an isolated retrospective rank process.

Keys include the implementation/dependency identity, all pose parameters and
every retained structure's path, size, mtime and ctime. No credentials or raw
process output are stored. Corrupt records are recomputed, never trusted.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class PoseCache:
    def __init__(self, function, directory, implementation_identity):
        self.function = function
        self.directory = Path(directory)
        self.identity = implementation_identity
        self.hits = self.misses = 0

    def key(self, sequence, paths, parameters):
        unique = list(dict.fromkeys(path.resolve() for path in paths if path.is_file()))
        fingerprints = []
        for path in unique:
            stat = path.stat()
            fingerprints.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        params = {name: sorted(value) if isinstance(value, set) else value
                  for name, value in parameters.items()}
        return hashlib.sha256(canonical({"sequence": sequence, "structures": fingerprints,
                                        "parameters": params, "identity": self.identity}).encode()).hexdigest()

    def __call__(self, sequence, paths, **parameters):
        key = self.key(sequence, paths, parameters)
        path = self.directory / f"{key}.json"
        try:
            record = json.loads(path.read_text())
            summary = record["summary"]
            if (record["key"] == key and isinstance(summary, dict)
                    and record["summary_sha256"] == hashlib.sha256(canonical(summary).encode()).hexdigest()):
                self.hits += 1
                return summary
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self.misses += 1
        summary = self.function(sequence, paths, **parameters)
        if self.key(sequence, paths, parameters) != key:
            raise ValueError("structure changed during pose evaluation")
        record = {"key": key, "summary": summary,
                  "summary_sha256": hashlib.sha256(canonical(summary).encode()).hexdigest()}
        self.directory.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".pose_", suffix=".json", dir=self.directory)
        try:
            with os.fdopen(handle, "w") as output:
                output.write(canonical(record))
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return summary
