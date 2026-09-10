import pandas as pd

from nanobody_opt.seed_mutation_database import snapshot_observations
from nanobody_opt.simulate_round import METRICS


def test_snapshot_rows_become_stably_identified_public_observations(tmp_path):
    snapshot = tmp_path / "fields.csv.gz"
    pd.DataFrame([
        {
            "epoch": 25001,
            "sequence": "AAAA",
            "final_score": 42,
            **{name: float(index) for index, name in enumerate(METRICS)},
        }
    ]).to_csv(snapshot, index=False)

    observations = snapshot_observations(
        snapshot,
        target="P20809",
        api_root="https://example.test",
        workers=2,
    )

    assert observations == [{
        "observation_id": "public-validator:epoch25001:AAAA",
        "sequence": "AAAA",
        "score": 42.0,
        "score_kind": "validator_final_rank_sum",
        "source": "public_validator_api",
        "epoch": 25001,
        "target": "P20809",
        "observation_count": 1,
        "metrics": {
            name: float(index) for index, name in enumerate(METRICS)
        },
        "api_root": "https://example.test",
    }]
