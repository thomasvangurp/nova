"""Read host and visible cgroup-v2 memory bounds without inspecting credentials."""
from __future__ import annotations
import os
import re
from pathlib import Path, PurePosixPath


def _unescape(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _absolute(value):
    path = PurePosixPath(_unescape(value))
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("invalid cgroup path")
    return path


def available_memory(*, proc_root=Path("/proc"), pid=None):
    """Return conservative visible headroom; callers must require resolution.

    A PID membership check disambiguates mount-root versus cgroup namespace
    paths. Every visible numeric ancestor limit contributes, not just the leaf.
    Unresolved or malformed cgroup metadata never masquerades as host-only
    confirmation. Hidden ancestors outside the mount are explicitly unproved.
    """
    pid = os.getpid() if pid is None else pid
    proc_root = Path(proc_root)
    lines = (proc_root / "meminfo").read_text().splitlines()
    fields = next((line.split() for line in lines if line.startswith("MemAvailable:")), None)
    if not fields or len(fields) != 3 or fields[2] != "kB" or not fields[1].isdecimal():
        raise ValueError("host available memory unavailable")
    values = {"host_available_bytes": int(fields[1]) * 1024,
              "cgroup_resolution": "unavailable", "cgroup_limit_unavailable": True,
              "applicable_cgroups": [], "ancestor_scope": "visible_cgroup_mount_only"}
    try:
        groups = [line[3:] for line in (proc_root / "self/cgroup").read_text().splitlines()
                  if line.startswith("0::")]
        if len(groups) != 1:
            raise ValueError("one unified cgroup record required")
        group = _absolute(groups[0])
        mounts = []
        for line in (proc_root / "self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            if right.split()[0] == "cgroup2":
                parts = left.split()
                root, mount = _absolute(parts[3]), Path(str(_absolute(parts[4])))
                mounts.append((root, mount))
        # Prefer the broadest visible hierarchy, so an accessible parent limit
        # is not lost by choosing a narrower bind mount of the same hierarchy.
        for root, mount in sorted(mounts, key=lambda item: (len(item[0].parts), str(item[1]))):
            candidates = []
            if group.is_relative_to(root):
                candidates.append(mount / str(group.relative_to(root)))
            candidates.append(mount / str(group).lstrip("/"))
            current = None
            for candidate in dict.fromkeys(candidates):
                if not candidate.is_relative_to(mount):
                    continue
                try:
                    members = (candidate / "cgroup.procs").read_text().split()
                except FileNotFoundError:
                    continue
                if str(pid) in members:
                    current = candidate
                    break
            if current is None:
                continue
            values.update(cgroup_mount=str(mount), cgroup_mount_root=str(root),
                          process_cgroup=str(group), resolved_process_cgroup=str(current))
            observed_limits = 0
            while True:
                maximum_path = current / "memory.max"
                if maximum_path.exists():
                    maximum = maximum_path.read_text().strip()
                    if maximum != "max" and not maximum.isdecimal():
                        raise ValueError("invalid cgroup maximum")
                    used_text = (current / "memory.current").read_text().strip()
                    if not used_text.isdecimal():
                        raise ValueError("invalid cgroup usage")
                    used = int(used_text)
                    values["applicable_cgroups"].append({"path": str(current), "memory_max": maximum,
                        "memory_current_bytes": used,
                        "headroom_bytes": None if maximum == "max" else max(0, int(maximum) - used)})
                    observed_limits += 1
                else:
                    # Some hierarchy roots do not expose memory.max. This is
                    # recorded, never invented as a numeric unlimited limit.
                    values["applicable_cgroups"].append({"path": str(current),
                        "memory_controller_limit_present": False, "headroom_bytes": None})
                if current == mount:
                    break
                current = current.parent
            if not observed_limits:
                raise ValueError("no applicable memory controller metadata")
            values.update(cgroup_resolution="resolved_visible_ancestors", cgroup_limit_unavailable=False)
            break
    except (OSError, ValueError, IndexError):
        values.update(cgroup_resolution="unavailable", cgroup_limit_unavailable=True)
    headrooms = [row["headroom_bytes"] for row in values["applicable_cgroups"]
                 if row["headroom_bytes"] is not None]
    if headrooms:
        values["cgroup_headroom_bytes"] = min(headrooms)
    values["available_bytes"] = min([values["host_available_bytes"]] + headrooms)
    return values
