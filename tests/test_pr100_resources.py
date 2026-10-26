"""Linux metadata fixtures: parent constraints and namespace ambiguity matter."""
from pathlib import Path
import pytest
from nanobody_opt.pr100_resources import available_memory


def setup(tmp_path, group="/team/job", root="/"):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text("MemAvailable: 1000 kB\n")
    (proc / "self/cgroup").write_text("0::" + group + "\n")
    mount = tmp_path / "cgroup"
    mount.mkdir()
    (proc / "self/mountinfo").write_text(f"29 23 0:26 {root} {mount} ro - cgroup2 cgroup rw\n")
    return proc, mount


def node(path, maximum="max", used=0, member=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "memory.max").write_text(str(maximum))
    (path / "memory.current").write_text(str(used))
    (path / "cgroup.procs").write_text(str(member) if member is not None else "")


def test_parent_limit_can_be_stricter_than_leaf(tmp_path):
    proc, mount = setup(tmp_path)
    node(mount)
    node(mount / "team", 700_000, 600_000)
    node(mount / "team/job", 900_000, 100_000, 42)
    got = available_memory(proc_root=proc, pid=42)
    assert got["cgroup_resolution"] == "resolved_visible_ancestors"
    assert got["available_bytes"] == 100_000
    assert len(got["applicable_cgroups"]) == 3


def test_bound_mount_uses_relative_hierarchy_and_membership(tmp_path):
    proc, mount = setup(tmp_path, "/host/container/job", "/host/container")
    node(mount, 600_000, 100_000)
    node(mount / "job", 400_000, 50_000, 42)
    got = available_memory(proc_root=proc, pid=42)
    assert got["available_bytes"] == 350_000
    assert got["resolved_process_cgroup"] == str(mount / "job")


def test_namespace_root_can_differ_from_mount_root(tmp_path):
    proc, mount = setup(tmp_path, "/", "/host/container")
    node(mount, 900_000, 850_000, 42)
    got = available_memory(proc_root=proc, pid=42)
    assert got["available_bytes"] == 50_000
    assert got["cgroup_resolution"] == "resolved_visible_ancestors"


def test_wrong_pid_never_certifies_host_only_fallback(tmp_path):
    proc, mount = setup(tmp_path)
    node(mount)
    node(mount / "team/job", 900_000, 100_000, 99)
    got = available_memory(proc_root=proc, pid=42)
    assert got["cgroup_resolution"] == "unavailable"
    assert got["cgroup_limit_unavailable"] is True


def test_malformed_parent_invalidates_partial_leaf_evidence(tmp_path):
    proc, mount = setup(tmp_path)
    node(mount)
    node(mount / "team", "invalid", 10)
    node(mount / "team/job", 1000, 900, 42)
    got = available_memory(proc_root=proc, pid=42)
    assert got["cgroup_resolution"] == "unavailable"
    assert got["cgroup_limit_unavailable"] is True


def test_host_limit_still_bounds_unlimited_cgroup(tmp_path):
    proc, mount = setup(tmp_path, "/")
    node(mount, "max", 10, 42)
    got = available_memory(proc_root=proc, pid=42)
    assert got["available_bytes"] == 1_024_000
    assert got["cgroup_resolution"] == "resolved_visible_ancestors"


def test_missing_hierarchy_root_max_preserves_real_child_limit(tmp_path):
    proc, mount = setup(tmp_path, "/job")
    node(mount / "job", 1000, 200, 42)
    got = available_memory(proc_root=proc, pid=42)
    assert got["available_bytes"] == 800
    assert got["cgroup_resolution"] == "resolved_visible_ancestors"
    assert got["applicable_cgroups"][-1]["memory_controller_limit_present"] is False


def test_missing_controller_metadata_is_explicitly_unavailable(tmp_path):
    proc, mount = setup(tmp_path, "/")
    (mount / "cgroup.procs").write_text("42")
    got = available_memory(proc_root=proc, pid=42)
    assert got["cgroup_resolution"] == "unavailable"


def test_unknown_host_memory_is_not_guessed(tmp_path):
    proc, mount = setup(tmp_path, "/")
    (proc / "meminfo").write_text("MemFree: 1000 kB\n")
    with pytest.raises(ValueError):
        available_memory(proc_root=proc, pid=42)
