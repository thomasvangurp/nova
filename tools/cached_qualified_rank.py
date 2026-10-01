"""Process-local cache adapter; never patches the live ranker's source."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor

from qualified_pose_cache import PoseCache


_worker_cache = None


def precompute(task):
    global _worker_cache
    root, directory, identity, sequence, paths, parameters = task
    if _worker_cache is None:
        sys.path.insert(0, str(root))
        for name, expected in identity["sources"].items():
            if hashlib.sha256((root / "nanobody_opt" / name).read_bytes()).hexdigest() != expected:
                raise ValueError("worker pose implementation changed")
        from nanobody_opt.pose_stability import summarize_sequence_structures
        for name, expected in identity["sources"].items():
            if hashlib.sha256((root / "nanobody_opt" / name).read_bytes()).hexdigest() != expected:
                raise ValueError("worker pose implementation changed during import")
        _worker_cache = PoseCache(summarize_sequence_structures, directory, identity)
    hits, misses = _worker_cache.hits, _worker_cache.misses
    _worker_cache(sequence, [Path(path) for path in paths], **parameters)
    return _worker_cache.hits - hits, _worker_cache.misses - misses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-script", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--pose-workers", type=int, default=1)
    args, rest = parser.parse_known_args()
    if not rest or rest[0] != "--":
        parser.error("ranker arguments must follow --")
    root = args.rank_script.parent.parent
    sys.path.insert(0, str(root))
    identity = {
        "sources": {name: hashlib.sha256((root / "nanobody_opt" / name).read_bytes()).hexdigest()
                    for name in ("pose_stability.py", "regions.py")},
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "gemmi")},
        "cache_schema": 1,
    }
    spec = importlib.util.spec_from_file_location("retrospective_qualified_rank", args.rank_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if any(hashlib.sha256((root / "nanobody_opt" / name).read_bytes()).hexdigest() != expected
           for name, expected in identity["sources"].items()):
        raise ValueError("pose source changed while importing")
    cache = PoseCache(module.summarize_sequence_structures, args.cache_dir, identity)
    module.summarize_sequence_structures = cache
    sys.argv = [str(args.rank_script), *rest[1:]]
    # Prefetch only the candidates the original cheap gates already accepted.
    # Every worker calls the unchanged scientific function. The ranker's own
    # filtering and ordering still run in the parent, in their original order.
    workers = max(1, min(args.pose_workers, int(os.cpu_count() - os.getloadavg()[0] - 8)))
    statistics = {"prefetch_hits": 0, "prefetch_misses": 0, "workers": workers}
    if workers > 1:
        import yaml
        from nanobody_opt.pose_stability import exact_structure_paths, residue_numbers
        target = rest[rest.index("--target") + 1]
        exact = [Path(rest[index + 1]) for index, value in enumerate(rest) if value == "--exact"]
        challenge = yaml.safe_load(Path(rest[rest.index("--pose-config") + 1]).read_text())["protein_selection"]["nanobody"]
        parameters = {"target_chain": str(challenge["structure_chain"]), "nanobody_chain": "B",
                      "epitope": residue_numbers(str(challenge["structure_binding_site"]))}
        confidence = exact_structure_paths(exact, target, pose_kind="confidence")
        writer = exact_structure_paths(exact, target, pose_kind="writer")
        limit = int(rest[rest.index("--limit") + 1])
        original_order = module._order_candidates
        calls = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            def ordered(candidates, selection_order, threshold, *extra, **kwargs):
                nonlocal calls
                result = original_order(candidates, selection_order, threshold, *extra, **kwargs)
                calls += 1
                rows, paths = [], None
                if calls == 1 and not module._pose_order_is_subset_invariant(result, selection_order, threshold):
                    rows, paths = result, confidence
                elif calls == 2 and "--paired-consensus-shortlist" not in rest:
                    rows, paths = result[:limit], writer
                tasks = [(root, args.cache_dir, identity, str(row["sequence"]),
                          [str(path) for path in paths.get(str(row["sequence"]), [])], parameters)
                         for row in rows]
                for hits, misses in pool.map(precompute, tasks, chunksize=8):
                    statistics["prefetch_hits"] += hits
                    statistics["prefetch_misses"] += misses
                return result
            module._order_candidates = ordered
            module.main()
    else:
        module.main()
    output = Path(rest[rest.index("--output") + 1])
    payload = json.loads(output.read_text())
    payload["pose_cache"] = {"hits": cache.hits, "misses": cache.misses,
                             **statistics,
                             "implementation_identity": identity}
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
