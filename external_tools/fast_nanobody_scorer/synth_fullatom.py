"""Synthesize the `design` step's output for a new nanobody sequence, full atom.

Why this exists
---------------
The design stage burns 24 s/sequence on 500 diffusion steps whose only product
is a .cif + .npz for the folding stage. A coordinate-perturbation trace showed
folding consumes, from that .cif, both chains' SEQUENCES and the TARGET chain's
geometry (as an SE(3)-invariant template); the designed chain's coordinates
reach no module the folding checkpoint contains. And chain A is reproducible to
0.13 A across independent designs, so it can be copied rather than regenerated.

An earlier attempt (synth_design.py) kept only chain B's backbone and lost
0.09 design_iiptm. Dropping atoms is NOT the same perturbation as moving them:
`parse_polymer` walks `const.ref_atoms[res_name]` and sets is_present=False for
every name it cannot find, which changes atom_resolved_mask and the whole
atom-level featurization. So chain B must carry the exact CCD heavy-atom set for
its residue type. This module builds that by SIDE-CHAIN GRAFTING.

Method
------
  * backbone (N, CA, C, O) is copied verbatim from the reference chain B --
    backbone geometry is residue-type independent, so it stays a real, strain-
    free design backbone whatever the new sequence says;
  * side chains come from a rotamer library harvested from real design outputs
    and are rigidly superimposed onto each target backbone frame by Kabsch on
    (N, CA, C);
  * candidates are scored against everything already placed and the least
    clashing one is kept, then a second pass re-places each side chain now that
    all its neighbours exist. One pass alone leaves a few hard overlaps, because
    the first residue of a packed core pair chooses blind.

Every table the parser can read residue identity from is rewritten together:
`_entity_poly`, `_entity_poly_seq`, `_atom_site`, `_pdbx_poly_seq_scheme` and
`_ma_qa_metric_local`. That is not cosmetic -- `parse_polymer` takes the
sequence from `_entity_poly_seq` and then asserts it against `_atom_site`.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
THREE_TO_ONE = {v: k for k, v in ONE_TO_THREE.items()}
BACKBONE = ("N", "CA", "C", "O")

# The atom set and ORDER the parser walks. Taken from boltzgen's own constants
# when importable so the two can never drift; the literal is the same table.
try:
    from boltzgen.data import const as _bg_const
    REF_ATOMS = {k: list(v) for k, v in _bg_const.ref_atoms.items()
                 if k in THREE_TO_ONE}
except Exception:  # standalone use, e.g. from a plain python
    REF_ATOMS = {
        "ALA": ["N", "CA", "C", "O", "CB"],
        "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
        "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
        "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
        "CYS": ["N", "CA", "C", "O", "CB", "SG"],
        "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
        "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
        "GLY": ["N", "CA", "C", "O"],
        "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
        "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
        "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
        "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
        "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
        "SER": ["N", "CA", "C", "O", "CB", "OG"],
        "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
        "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2",
                "CE3", "CZ2", "CZ3", "CH2"],
        "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2",
                "CZ", "OH"],
        "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    }


# --------------------------------------------------------------------------
# minimal mmCIF handling
#
# These files are machine written by boltzgen's own writer: one data block, no
# multi-line values, no quoted whitespace inside loop rows. A full CIF parser
# would round-trip the file through gemmi and reformat chain A, which we do not
# want -- chain A must survive byte-for-byte.
# --------------------------------------------------------------------------
class CifDoc:
    def __init__(self, text: str):
        self.lines = text.splitlines()
        self.loops: dict[str, tuple[list[str], int, int]] = {}
        i = 0
        while i < len(self.lines):
            if self.lines[i].strip() == "loop_":
                tags, j = [], i + 1
                while j < len(self.lines) and self.lines[j].startswith("_"):
                    tags.append(self.lines[j].strip())
                    j += 1
                start = j
                while j < len(self.lines) and self.lines[j].strip() and \
                        not self.lines[j].startswith(("_", "loop_", "#")):
                    j += 1
                cat = tags[0].split(".", 1)[0]
                self.loops[cat] = ([t.split(".", 1)[1] for t in tags], start, j)
                i = j
            else:
                i += 1

    def rows(self, cat: str) -> list[list[str]]:
        _, s, e = self.loops[cat]
        return [self.lines[k].split() for k in range(s, e)]

    def col(self, cat: str, tag: str) -> int:
        return self.loops[cat][0].index(tag)

    def replace_rows(self, cat: str, new_rows: Iterable[list[str]]) -> None:
        """Splice rewritten rows in, keeping every other line untouched."""
        tags, s, e = self.loops[cat]
        body = [" ".join(r) for r in new_rows]
        self.lines[s:e] = body
        shift = len(body) - (e - s)
        if shift:
            for c, (t, a, b) in list(self.loops.items()):
                if a >= e:
                    self.loops[c] = (t, a + shift, b + shift)
        self.loops[cat] = (tags, s, s + len(body))

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


class Residue:
    __slots__ = ("name", "seq_id", "asym", "names", "xyz", "raw", "source")

    def __init__(self, name, seq_id, asym, source):
        self.name, self.seq_id, self.asym, self.source = name, seq_id, asym, source
        self.names: list[str] = []
        self.xyz: list[tuple[float, float, float]] = []
        self.raw: list[list[str]] = []

    def get(self, atom: str) -> Optional[np.ndarray]:
        try:
            return np.asarray(self.xyz[self.names.index(atom)], float)
        except ValueError:
            return None

    def frame(self) -> Optional[np.ndarray]:
        pts = [self.get(a) for a in ("N", "CA", "C")]
        return None if any(p is None for p in pts) else np.stack(pts)

    def coords(self) -> np.ndarray:
        return np.asarray(self.xyz, float)


def read_residues(doc: CifDoc, source: str, asym: Optional[str] = None) -> list[Residue]:
    c = {t: doc.col("_atom_site", t) for t in
         ("label_atom_id", "label_comp_id", "label_asym_id", "label_seq_id",
          "Cartn_x", "Cartn_y", "Cartn_z")}
    out, cur, key = [], None, None
    for r in doc.rows("_atom_site"):
        a = r[c["label_asym_id"]]
        if asym is not None and a != asym:
            continue
        k = (a, r[c["label_seq_id"]])
        if k != key:
            cur = Residue(r[c["label_comp_id"]], r[c["label_seq_id"]], a, source)
            out.append(cur)
            key = k
        cur.names.append(r[c["label_atom_id"]])
        cur.xyz.append((float(r[c["Cartn_x"]]), float(r[c["Cartn_y"]]),
                        float(r[c["Cartn_z"]])))
        cur.raw.append(list(r))
    return out


def _entity_of_asym(doc: CifDoc, asym: str) -> str:
    """_entity_poly.pdbx_strand_id is '?' in these files; _struct_asym is not."""
    c_id = doc.col("_struct_asym", "id")
    c_ent = doc.col("_struct_asym", "entity_id")
    for r in doc.rows("_struct_asym"):
        if r[c_id] == asym:
            return r[c_ent]
    raise SystemExit(f"no _struct_asym row for chain {asym}")


def sequence_of(cif: Path, asym: str = "B") -> str:
    """Recover a design's chain sequence from its own _entity_poly block."""
    doc = CifDoc(cif.read_text())
    entity_id = _entity_of_asym(doc, asym)
    c_ent = doc.col("_entity_poly", "entity_id")
    c_seq1 = doc.col("_entity_poly", "pdbx_seq_one_letter_code")
    for r in doc.rows("_entity_poly"):
        if r[c_ent] == entity_id:
            return r[c_seq1]
    raise SystemExit(f"{cif}: no _entity_poly row for entity {entity_id}")


# --------------------------------------------------------------------------
# grafting
# --------------------------------------------------------------------------
class TemplateLibrary:
    """Side-chain conformers per residue type, harvested from real designs.

    Conformers of one type all have the same atom count, so they are stacked
    into (K, 3, 3) frames and (K, n, 3) side chains and the whole candidate scan
    for a position becomes one batched Kabsch plus one distance tensor.
    """

    def __init__(self):
        self._raw: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        self.by_type: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}

    @classmethod
    def from_cifs(cls, paths: Iterable[Path], exclude_source: Optional[str] = None):
        lib = cls()
        for p in paths:
            src = Path(p).name
            if exclude_source and src == exclude_source:
                continue
            doc = CifDoc(Path(p).read_text())
            for res in read_residues(doc, src):
                lib.add(res)
        lib.finalize()
        return lib

    def add(self, res: Residue) -> None:
        want = REF_ATOMS.get(res.name)
        if want is None or len(want) == 4:      # unknown type, or GLY: nothing to graft
            return
        frame = res.frame()
        if frame is None:
            return
        try:
            sc = np.stack([res.coords()[res.names.index(a)] for a in want[4:]])
        except ValueError:
            return                              # incomplete side chain, unusable
        self._raw.setdefault(res.name, []).append((frame, sc))

    def finalize(self) -> None:
        self.by_type = {t: (REF_ATOMS[t][4:],
                            np.stack([f for f, _ in v]),
                            np.stack([c for _, c in v]))
                        for t, v in self._raw.items()}

    def coverage(self) -> dict[str, int]:
        return {t: len(self._raw.get(t, ())) for t in sorted(REF_ATOMS)}

    def missing(self) -> list[str]:
        return [t for t in sorted(REF_ATOMS)
                if len(REF_ATOMS[t]) > 4 and not self._raw.get(t)]


CLASH_CUTOFF = 3.4          # A; soft, quadratic below this
NEIGH_RADIUS = 10.0         # A from CA; TRP reaches ~6.5 A, plus the cutoff


def kabsch_batch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(K,3,3) -> one (3,3) target: rigid transforms, no reflection."""
    pc, qc = P.mean(1), Q.mean(0)
    H = np.einsum("kji,jl->kil", P - pc[:, None, :], Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("kji,klj->kil", Vt, U)))
    D = np.zeros((len(P), 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = np.einsum("kji,kjl,kml->kim", Vt, D, U)
    return R, qc - np.einsum("kij,kj->ki", R, pc)


def soft_clash(placed: np.ndarray, neigh: np.ndarray) -> np.ndarray:
    """Quadratic penalty inside CLASH_CUTOFF, for a (K, n, 3) batch of side chains.

    Expanded as |a|^2 + |b|^2 - 2a.b so the only large temporary is (K, n, M)
    and the work lands in one BLAS call; the naive broadcast builds a
    (K, n, M, 3) difference tensor and dominated the whole synthesis.
    """
    if not len(neigh):
        return np.zeros(len(placed))
    K, n, _ = placed.shape
    flat = placed.reshape(K * n, 3)
    d2 = ((flat * flat).sum(1)[:, None] + (neigh * neigh).sum(1)[None, :]
          - 2.0 * (flat @ neigh.T))
    np.clip(d2, 0.0, None, out=d2)
    pen = np.clip(CLASH_CUTOFF - np.sqrt(d2), 0.0, None)
    return np.einsum("ij,ij->i", pen, pen).reshape(K, n).sum(1)


def graft(lib: TemplateLibrary, res_type: str, target_frame: np.ndarray,
          neigh: np.ndarray, rng: np.random.Generator, n_candidates: int = 24
          ) -> tuple[list[str], np.ndarray, float]:
    """Place `res_type`'s side chain on `target_frame`, least-clashing rotamer.

    All candidates are superimposed and scored at once, against a `neigh` set
    that already excludes the atoms genuinely bonded to this residue.
    """
    entry = lib.by_type.get(res_type)
    if entry is None:
        raise KeyError(f"no side-chain template for {res_type}")
    names, frames, scs = entry
    if len(frames) > n_candidates:
        sel = rng.choice(len(frames), n_candidates, replace=False)
        frames, scs = frames[sel], scs[sel]

    R, t = kabsch_batch(frames, target_frame)
    placed = np.einsum("kij,knj->kni", R, scs) + t[:, None, :]
    score = soft_clash(placed, neigh)
    k = int(np.argmin(score))
    return names, placed[k], float(score[k])


class _Packing:
    """Every atom of the model in one preallocated array.

    Side-chain slots are sized from the residue types up front (a re-graft never
    changes the atom count), so a neighbour query is a single masked norm over
    ~2250 rows -- cheaper than a spatial index at this size, and it survives a
    side chain moving with no bookkeeping. Unplaced slots sit at a sentinel far
    outside the radius filter.
    """

    FIXED = -10 ** 6        # chain A and side chains: never bonded to a backbone
    FAR = 1e6

    def __init__(self, fixed_xyz: np.ndarray, bb_xyz: np.ndarray,
                 bb_owner: np.ndarray, sc_len: list[int]):
        n_fix, n_bb = len(fixed_xyz), len(bb_xyz)
        self.sc_start = n_fix + n_bb + np.concatenate([[0], np.cumsum(sc_len)])[:-1]
        self.sc_len = np.asarray(sc_len)
        total = n_fix + n_bb + int(self.sc_len.sum())
        self.xyz = np.full((total, 3), self.FAR)
        self.owner = np.full(total, self.FIXED)
        self.xyz[:n_fix] = fixed_xyz
        self.xyz[n_fix:n_fix + n_bb] = bb_xyz
        self.owner[n_fix:n_fix + n_bb] = bb_owner

    def side_chain(self, i: int) -> np.ndarray:
        s = self.sc_start[i]
        return self.xyz[s:s + self.sc_len[i]]

    def set_side_chain(self, i: int, xyz: np.ndarray) -> None:
        s = self.sc_start[i]
        self.xyz[s:s + self.sc_len[i]] = xyz

    def neighbours(self, i: int, center: np.ndarray) -> np.ndarray:
        """Atoms within NEIGH_RADIUS of `center`, minus what is bonded to i:
        residue i's own side chain and the i-1/i/i+1 backbone."""
        m = np.einsum("ij,ij->i", self.xyz - center, self.xyz - center) < NEIGH_RADIUS ** 2
        m &= np.abs(self.owner - i) > 1
        s = self.sc_start[i]
        m[s:s + self.sc_len[i]] = False
        return self.xyz[m]


# --------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------
def _fmt(v: float) -> str:
    # matches the writer's precision; enough digits that the float32 round-trips
    return f"{v:.9g}"


def synthesize(ref_cif: Path, sequence: str, out_cif: Path,
               lib: TemplateLibrary, design_asym: str = "B",
               seed: int = 0, passes: int = 2) -> dict:
    doc = CifDoc(ref_cif.read_text())
    ref_res = read_residues(doc, ref_cif.name, asym=design_asym)
    if len(ref_res) != len(sequence):
        raise SystemExit(
            f"{ref_cif.name}: chain {design_asym} has {len(ref_res)} residues but "
            f"the sequence has {len(sequence)}. Synthesis grafts onto a real "
            f"design backbone, so it needs a same-length reference; pick a "
            f"reference design of length {len(sequence)}."
        )
    three = [ONE_TO_THREE[c] for c in sequence]
    rng = np.random.default_rng(seed)

    ac = doc.loops["_atom_site"][0]
    c_atom, c_comp = ac.index("label_atom_id"), ac.index("label_comp_id")
    c_asym, c_id, c_sym = ac.index("label_asym_id"), ac.index("id"), ac.index("type_symbol")
    c_x, c_y, c_z = ac.index("Cartn_x"), ac.index("Cartn_y"), ac.index("Cartn_z")

    # everything outside the designed chain -- all of chain A -- is carried
    # through untouched apart from the serial renumber
    other_rows = [r for r in doc.rows("_atom_site") if r[c_asym] != design_asym]
    fixed_xyz = np.array([[float(r[c_x]), float(r[c_y]), float(r[c_z])]
                          for r in other_rows], float).reshape(-1, 3)

    # backbones are all placed before any side chain, so a rotamer is scored
    # against the finished backbone rather than only the part emitted so far
    bb_rows: list[list[list[str]]] = []
    bb_xyz, bb_owner = [], []
    for i, (res, comp) in enumerate(zip(ref_res, three)):
        proto = dict(zip(res.names, res.raw))
        rows = []
        for a in BACKBONE:
            if a not in proto:
                raise SystemExit(f"reference residue {res.seq_id} lacks backbone {a}")
            row = list(proto[a])
            row[c_comp] = comp
            rows.append(row)
            bb_xyz.append(res.get(a))
            bb_owner.append(i)
        bb_rows.append(rows)
    sc_len = [len(REF_ATOMS[c]) - 4 for c in three]
    pack = _Packing(fixed_xyz, np.stack(bb_xyz), np.asarray(bb_owner), sc_len)

    frames = [r.frame() for r in ref_res]
    side_names: list[Optional[list[str]]] = [None] * len(three)
    scores = np.zeros(len(three))
    for p in range(passes):
        for i, comp in enumerate(three):
            if len(REF_ATOMS[comp]) == 4:        # GLY
                continue
            neigh = pack.neighbours(i, frames[i][1])
            if p:
                # a later neighbour may have landed on a side chain that was
                # clash-free when it was placed, so rescore rather than trust
                # the earlier score -- but only rescan rotamers if it is now bad
                scores[i] = soft_clash(pack.side_chain(i)[None], neigh)[0]
                if scores[i] == 0.0:
                    continue
            names, placed, scores[i] = graft(lib, comp, frames[i], neigh, rng)
            side_names[i] = names
            pack.set_side_chain(i, placed)

    new_b: list[list[str]] = []
    for i, rows in enumerate(bb_rows):
        new_b.extend(rows)
        if side_names[i] is None:
            continue
        tmpl = rows[0]
        for name, p in zip(side_names[i], pack.side_chain(i)):
            row = list(tmpl)
            row[c_atom] = name
            row[c_sym] = name[0]
            row[c_x], row[c_y], row[c_z] = _fmt(p[0]), _fmt(p[1]), _fmt(p[2])
            new_b.append(row)

    # splice chain B back where it was, then renumber every serial
    out_rows, inserted = [], False
    for r in doc.rows("_atom_site"):
        if r[c_asym] == design_asym:
            if not inserted:
                out_rows.extend(new_b)
                inserted = True
        else:
            out_rows.append(r)
    if not inserted:
        out_rows.extend(new_b)
    for i, r in enumerate(out_rows, 1):
        r[c_id] = str(i)
    doc.replace_rows("_atom_site", out_rows)

    _rewrite_sequence_tables(doc, sequence, three, design_asym)
    out_cif.write_text(doc.text())
    return {"n_residues": len(three), "n_atoms": len(out_rows),
            "n_design_atoms": len(new_b), "max_clash_score": float(scores.max()),
            "n_clashing_residues": int((scores > 1.0).sum())}


def _rewrite_sequence_tables(doc: CifDoc, sequence: str, three: list[str],
                             design_asym: str) -> None:
    """Keep every table the parser can read residue identity from consistent.

    `parse_polymer` takes the sequence from `_entity_poly_seq` and then asserts
    it against `_atom_site`, so those two disagreeing is a crash rather than a
    silent mismatch. The other two are metadata, but are cheap to keep honest.
    """
    entity_id = _entity_of_asym(doc, design_asym)
    ep = doc.rows("_entity_poly")
    c_ent = doc.col("_entity_poly", "entity_id")
    c_seq1 = doc.col("_entity_poly", "pdbx_seq_one_letter_code")
    for r in ep:
        if r[c_ent] == entity_id:
            r[c_seq1] = sequence
    doc.replace_rows("_entity_poly", ep)

    eps = doc.rows("_entity_poly_seq")
    c_e = doc.col("_entity_poly_seq", "entity_id")
    c_mon = doc.col("_entity_poly_seq", "mon_id")
    k = 0
    for r in eps:
        if r[c_e] == entity_id:
            r[c_mon] = three[k]
            k += 1
    if k != len(three):
        raise SystemExit(f"_entity_poly_seq has {k} rows for entity {entity_id}, "
                         f"sequence has {len(three)}")
    doc.replace_rows("_entity_poly_seq", eps)

    if "_pdbx_poly_seq_scheme" in doc.loops:
        rows = doc.rows("_pdbx_poly_seq_scheme")
        tags = doc.loops["_pdbx_poly_seq_scheme"][0]
        c_a = tags.index("asym_id")
        cols = [tags.index(t) for t in ("mon_id", "pdb_mon_id", "auth_mon_id")
                if t in tags]
        k = 0
        for r in rows:
            if r[c_a] == design_asym:
                for c in cols:
                    r[c] = three[k]
                k += 1
        doc.replace_rows("_pdbx_poly_seq_scheme", rows)

    if "_ma_qa_metric_local" in doc.loops:
        rows = doc.rows("_ma_qa_metric_local")
        tags = doc.loops["_ma_qa_metric_local"][0]
        c_a, c_c = tags.index("label_asym_id"), tags.index("label_comp_id")
        k = 0
        for r in rows:
            if r[c_a] == design_asym:
                r[c_c] = three[k]
                k += 1
        doc.replace_rows("_ma_qa_metric_local", rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref-cif", required=True)
    ap.add_argument("--ref-npz", required=True)
    ap.add_argument("--sequence", help="one-letter; default: the reference's own")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--library", nargs="+", required=True,
                    help="design .cif files to harvest side chains from")
    ap.add_argument("--design-asym", default="B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--exclude-ref-from-library", action="store_true",
                    help="drop the reference file from the library, so a "
                         "same-sequence comparison cannot cheat by reusing the "
                         "reference's own side chains")
    a = ap.parse_args()

    ref_cif, out = Path(a.ref_cif), Path(a.out_prefix)
    seq = a.sequence or sequence_of(ref_cif, a.design_asym)

    t0 = time.perf_counter()
    lib = TemplateLibrary.from_cifs(
        [Path(p) for p in a.library],
        exclude_source=ref_cif.name if a.exclude_ref_from_library else None)
    t_lib = time.perf_counter() - t0
    missing = lib.missing()
    if missing:
        raise SystemExit(f"library has no template for: {', '.join(missing)}")

    t0 = time.perf_counter()
    info = synthesize(ref_cif, seq, out.with_suffix(".cif"), lib,
                      design_asym=a.design_asym, seed=a.seed, passes=a.passes)
    t_syn = time.perf_counter() - t0
    # folding reads only ss_type / binding_type out of the npz and the model
    # consumes neither, so the reference copy is byte-valid for any sequence
    shutil.copyfile(a.ref_npz, out.with_suffix(".npz"))

    print(f"library: {sum(lib.coverage().values())} side chains, {t_lib*1e3:.0f} ms")
    print(f"synth:   {info['n_residues']} res / {info['n_design_atoms']} design "
          f"atoms / {info['n_atoms']} total, {t_syn*1e3:.1f} ms  "
          f"(residues still clashing: {info['n_clashing_residues']})")
    print(f"wrote {out.with_suffix('.cif')} and {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
