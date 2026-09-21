"""Run the VALIDATOR'S OWN validity code against our release, before sending it.

Exit status is the point: 0 when every uid would score, 1 when any would be
skipped. A submission that cannot score is worse than none -- it spends the
commitment slot and the epoch. At 25248 all ten uids were skipped for one
molecule and nothing said so until the leaderboard did, hours later.

Not a reimplementation. This imports neurons/validator/molecule_validity.py
from metanova-labs/nova and calls it, with the two network-bound helpers
pointed at our local copy of the same public archive the validator downloads.
"""
import json, sys, types
from pathlib import Path

VALIDATOR = Path("/opt/nova-submit/nova_validator")
sys.path.insert(0, str(VALIDATOR))

import pandas as pd
import yaml
from rdkit import Chem

ARCHIVE = Path("/opt/nova-submit/P40261_molecules.csv")
frame = pd.read_csv(ARCHIVE)
INCHI = set(frame["InChI_Key"].dropna().astype(str))

# utils/__init__ pulls in protein and nanobody machinery the molecule gate
# never touches. Stub whatever is missing rather than installing a stack we do
# not exercise -- but stub ONLY missing modules, so nothing real is shadowed.
def _import_gate(attempts=12):
    for _ in range(attempts):
        try:
            import neurons.validator.molecule_validity as module
            return module
        except ModuleNotFoundError as missing:
            name = missing.name
            if name.startswith("neurons") or name.startswith("utils"):
                raise
            # Register every parent too, and give each a __path__ so dotted
            # imports resolve; a bare module makes "import a.b" raise a
            # confusing TypeError instead of ModuleNotFoundError.
            parts = name.split(".")
            for depth in range(1, len(parts) + 1):
                dotted = ".".join(parts[:depth])
                if dotted in sys.modules:
                    continue
                stub = types.ModuleType(dotted)
                stub.__path__ = []
                stub.__getattr__ = lambda attr: type(attr, (), {})
                sys.modules[dotted] = stub
            print(json.dumps(dict(stubbed=name)), file=sys.stderr)
    raise RuntimeError("could not import the validator gate")


mv = _import_gate()

mv.get_historical_submissions = lambda target, kind, revision=None: frame.copy()
mv.entry_unique_for_protein_hf = lambda protein, smiles, kind='molecules': (
    Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)) not in INCHI
    if Chem.MolFromSmiles(smiles) else False)

cfg = yaml.safe_load((VALIDATOR / "config/config.yaml").read_text())
# The gate reads keys from SEVERAL config sections. Passing only
# molecule_requirements made config.get("random_valid_reaction") None, which
# sent it down the allowed-reactions branch and iterated a key that was not
# there -- a harness bug that looked exactly like a molecule failure.
config = dict(cfg["molecule_requirements"])
config.update(cfg.get("reaction_filtering", {}))
config["small_molecule_target"] = ["P40261"]

fleet = json.loads(Path(sys.argv[1]).read_text())
reaction = int(fleet["reaction"])
uid_to_data, score_dict = {}, {}
for sub in fleet["submissions"]:
    uid = int(sub["uid"])
    uid_to_data[uid] = {"molecules": [m["name"] for m in sub["molecules"]]}
    score_dict[uid] = {}

valid = mv.validate_molecules_and_calculate_entropy(
    uid_to_data=uid_to_data, score_dict=score_dict, config=config,
    allowed_reaction=f"rxn:{reaction}")

would_score = len(valid)
print(json.dumps(dict(
    epoch=fleet.get("epoch"), reaction=reaction,
    uids_submitted=len(uid_to_data),
    UIDS_THAT_WOULD_SCORE=len(valid),
    molecules_per_uid={str(u): len(v["smiles"]) for u, v in valid.items()},
    all_twenty=all(len(v["smiles"]) == config["num_molecules"] for v in valid.values()),
    entropy={str(u): round(float(d.get("entropy", 0)), 4)
             for u, d in score_dict.items() if d.get("entropy") is not None})))

# Non-zero if ANY uid would be skipped, so a caller can refuse to ship.
sys.exit(0 if would_score and would_score == len(uid_to_data) else 1)
