"""Read-only audit of frozen nanobody commit forecasts against public outcomes.

No local inference, live miner mutation, fitted selection policy, or credentials.
SSH uses an opaque key locator; subprocess stderr and HTTP error bodies are never
printed. Artifacts contain scientific allowlisted fields, not full responses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shlex
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

METRICS = {
    "design_iiptm": "max", "design_ptm": "max",
    "design_to_target_iptm": "max", "min_design_to_target_pae": "min",
    "interaction_pae": "min", "plip_hbonds_refolded": "max",
    "plip_saltbridge_refolded": "max", "delta_sasa_refolded": "max",
    "liability_score": "min", "liability_num_violations": "min",
}
CATEGORIES = {
    "confidence_rank": list(METRICS)[:5],
    "physical_interaction_rank": list(METRICS)[5:8],
    "developability_rank": list(METRICS)[8:],
}
API = "https://compound-api.metanova-labs.ai"


def digest(sequence):
    return hashlib.sha256(sequence.strip().upper().encode()).hexdigest()


def average_ranks(values):
    ranks = {}
    ordered = sorted(set(values))
    offset = 0
    for value in ordered:
        count = values.count(value)
        ranks[value] = offset + (count + 1) / 2
        offset += count
    return [ranks[value] for value in values]


def spearman(x, y):
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return statistics.correlation(average_ranks(x), average_ranks(y))


def concordance(rows, predictor, outcomes=None):
    """Higher predictor should imply lower validator score; neutral ties = .5."""
    outcomes = outcomes or [row["actual_score"] for row in rows]
    values = [row[predictor] for row in rows]
    correct = 0.0
    count = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            product = (values[i] - values[j]) * (outcomes[i] - outcomes[j])
            correct += 1.0 if product < 0 else 0.5 if product == 0 else 0.0
            count += 1
    return correct / count if count else None


def dense_metric_ranks(metrics, field):
    return {
        key: 1 + len({other[key] for other in field
                      if (other[key] > value if direction == "max" else other[key] < value)})
        for key, direction in METRICS.items()
        for value in [metrics[key]]
    }


def replay_field(field):
    for row in field:
        ranks = dense_metric_ranks(row["metrics"], [other["metrics"] for other in field])
        categories = {key: sum(ranks[name] for name in names)
                      for key, names in CATEGORIES.items()}
        if categories != row["categories"] or sum(ranks.values()) != row["actual_score"]:
            raise ValueError("validator_dense_rank_parity_failed")
        row["metric_ranks"] = ranks


def remote_history(epochs, host, port, key):
    # This remote program reads only named scientific history artifacts.
    code = '''import json,hashlib,pathlib
epochs=EPOCHS
out=[]
for epoch in epochs:
 p=pathlib.Path('/root/artifacts/nanobody_history')/('epoch%d.json'%epoch)
 raw=p.read_bytes(); d=json.loads(raw); submissions=[]
 for s in d['submissions']:
  pred=s.get('prediction_at_commit') or {}; sel=s.get('selection') or {}
  seq=str(s.get('sequence') or '').strip().upper()
  h=hashlib.sha256(seq.encode()).hexdigest()
  if h!=s.get('sequence_sha256'): raise ValueError('history_sequence_digest_mismatch')
  submissions.append({'uid':s.get('uid'),'hotkey':s.get('hotkey'),'sequence_sha256':h,
    'commit_block':s.get('commit_block'),'publication_stage':s.get('publication_stage'),
    'selection':{k:sel.get(k) for k in ['score','score_median','score_p90','win_probability']},
    'prediction_at_commit':{k:pred.get(k) for k in ['event','probability','threshold',
     'score_median','observation_count','selection_metric_policy','selection_metric_weights',
     'consensus_ranking_policy']},
    'reference_epochs':(pred.get('reference') or {}).get('epochs',[])})
 out.append({'epoch':epoch,'target':d.get('target'),'history_sha256':hashlib.sha256(raw).hexdigest(),
   'submissions':submissions})
print(json.dumps(out))
'''.replace("EPOCHS", repr(epochs))
    result = subprocess.run([
        "ssh", "-i", key, "-p", str(port), "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes", "-o", "ControlMaster=no",
        "-o", "ControlPath=none", "-o", "ConnectTimeout=10", host,
        "python3 -c " + shlex.quote(code),
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=40)
    if result.returncode:
        raise RuntimeError("history_ssh_failed")
    return json.loads(result.stdout)


def public_field(epoch, target):
    url = f"{API}/api/competitions/leaderboard/{epoch}/nanobodies?limit=1000&sequence_detail=full"
    request = Request(url, headers={"User-Agent": "nova-frozen-order-audit/1"})
    with urlopen(request, timeout=30) as response:
        raw = response.read()
    payload = json.loads(raw)
    if payload.get("success") is not True:
        raise ValueError("leaderboard_unavailable")
    field = []
    for submission in payload["data"]:
        for nb in submission.get("nanobodies", []):
            if nb.get("protein_name") != target:
                continue
            metrics = {key: float(nb[key]) for key in METRICS}
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError("nonfinite_validator_metric")
            if not nb.get("sequence"):
                raise ValueError("full_sequence_unavailable")
            h = digest(nb["sequence"])
            if nb.get("sequence_hash") and nb["sequence_hash"] != h:
                raise ValueError("api_sequence_digest_mismatch")
            field.append({"uid": submission["uid"], "hotkey": submission["hotkey"],
                          "sequence_sha256": h, "metrics": metrics,
                          "categories": {key: float(nb[key]) for key in CATEGORIES},
                          "actual_score": float(nb["final_score"]),
                          "actual_rank": int(submission["nanobody_leaderboard_rank"])})
    if not field:
        raise ValueError("empty_validator_field")
    replay_field(field)
    return {"epoch": epoch, "url": url, "response_sha256": hashlib.sha256(raw).hexdigest(),
            "field": field}


def joined_rows(history, leaderboard):
    lookup = {}
    for actual in leaderboard["field"]:
        identity = (actual["hotkey"], actual["sequence_sha256"])
        if identity in lookup:
            raise ValueError("ambiguous_hotkey_and_sequence_identity")
        lookup[identity] = actual
    rows = []
    for submission in history["submissions"]:
        prediction = submission["prediction_at_commit"]
        if (prediction.get("event") != "selection_rank_sum_below_fixed_threshold"
                or prediction.get("selection_metric_policy") != "historical-drivers"
                or prediction.get("consensus_ranking_policy") != "off"):
            raise ValueError("incompatible_frozen_prediction_policy")
        if any(epoch >= history["epoch"] for epoch in submission["reference_epochs"]):
            raise ValueError("prediction_reference_lookahead")
        probability = float(prediction["probability"])
        if not 0 <= probability <= 1:
            raise ValueError("invalid_probability")
        if probability != float(submission["selection"]["win_probability"]):
            raise ValueError("inconsistent_frozen_probability")
        actual = lookup.get((submission["hotkey"], submission["sequence_sha256"]))
        row = {key: submission[key] for key in ["uid", "sequence_sha256", "commit_block", "publication_stage"]}
        row.update({"epoch": history["epoch"], "field_size": len(leaderboard["field"]),
                    "prediction": prediction, "reference_epochs": submission["reference_epochs"],
                    "probability": probability, "negative_median": -float(prediction["score_median"]),
                    "negative_p90": -float(submission["selection"]["score_p90"]),
                    "scored": actual is not None})
        if actual:
            weights = prediction["selection_metric_weights"]
            if set(weights) != set(METRICS):
                raise ValueError("unknown_frozen_metric_weights")
            weighted_score = sum(float(weights[key]) * actual["metric_ranks"][key] for key in METRICS)
            row.update({key: actual[key] for key in ["actual_score", "actual_rank", "metrics", "metric_ranks", "categories"]})
            row.update({"uid_matches": actual["uid"] == submission["uid"],
                        "observed_selection_score": weighted_score,
                        "predicted_event_occurred": weighted_score < float(prediction["threshold"]),
                        "top10": actual["actual_rank"] <= 10, "winner": actual["actual_rank"] == 1})
        rows.append(row)
    for predictor in ["probability", "negative_median", "negative_p90"]:
        ordered = sorted(rows, key=lambda row: (-row[predictor], row["sequence_sha256"]))
        for position, row in enumerate(ordered, 1):
            row[f"{predictor}_order_published"] = position
        scored = [row for row in ordered if row["scored"]]
        for position, row in enumerate(scored, 1):
            row[f"{predictor}_order_scored"] = position
    return rows


def quantile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def epoch_bootstrap(values, rng, count):
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(count)]
    return [quantile(means, .025), quantile(means, .975)]


def summarize(rows, repeats=10000):
    rng = random.Random(25172)
    grouped = {epoch: [row for row in rows if row["epoch"] == epoch]
               for epoch in sorted({row["epoch"] for row in rows})}
    scored = [row for row in rows if row["scored"]]
    result = {"epochs": list(grouped), "published": len(rows), "scored": len(scored),
              "unique_sequences": len({row["sequence_sha256"] for row in rows}),
              "unscored": len(rows) - len(scored), "wins": sum(row["winner"] for row in scored),
              "top10": sum(row["top10"] for row in scored), "predictors": {}}
    for predictor in ["probability", "negative_median", "negative_p90"]:
        correlations, accuracies, weighted_accuracies = [], [], []
        for group in grouped.values():
            subset = [row for row in group if row["scored"]]
            rho = spearman([row[predictor] for row in subset], [-row["actual_score"] for row in subset])
            if rho is not None:
                correlations.append(rho)
            accuracy = concordance(subset, predictor)
            if accuracy is not None:
                accuracies.append(accuracy)
                weighted_accuracies.append(concordance(subset, predictor, [row["observed_selection_score"] for row in subset]))
        mean_accuracy = statistics.mean(accuracies)
        permuted_means = []
        for _ in range(repeats):
            permuted = []
            for group in grouped.values():
                subset = [row for row in group if row["scored"]]
                outcomes = [row["actual_score"] for row in subset]
                rng.shuffle(outcomes)
                permuted.append(concordance(subset, predictor, outcomes))
            permuted_means.append(statistics.mean(permuted))
        predictor_result = {
            "mean_epoch_spearman": statistics.mean(correlations),
            "spearman_epoch_bootstrap_95ci": epoch_bootstrap(correlations, rng, repeats),
            "mean_epoch_pairwise_accuracy": mean_accuracy,
            "pairwise_epoch_bootstrap_95ci": epoch_bootstrap(accuracies, rng, repeats),
            "weighted_target_pairwise_accuracy": statistics.mean(weighted_accuracies),
            "permutation_two_sided_p": (1 + sum(abs(value - .5) >= abs(mean_accuracy - .5)
                                                  for value in permuted_means)) / (repeats + 1),
            "top_k": {},
        }
        for k in [1, 3]:
            selected = [row for group in grouped.values()
                        for row in sorted(group, key=lambda row: row[f"{predictor}_order_published"])[:k]]
            predictor_result["top_k"][str(k)] = {
                "selected": len(selected), "scored": sum(row["scored"] for row in selected),
                "top10": sum(row.get("top10", False) for row in selected),
                "wins": sum(row.get("winner", False) for row in selected),
                "random_order_expected_top10": sum(min(k, len(group)) * sum(row.get("top10", False) for row in group) / len(group)
                                                    for group in grouped.values()),
                "random_order_expected_wins": sum(min(k, len(group)) * sum(row.get("winner", False) for row in group) / len(group)
                                                   for group in grouped.values()),
            }
        result["predictors"][predictor] = predictor_result
    probabilities = [row["probability"] for row in scored]
    labels = [int(row["predicted_event_occurred"]) for row in scored]
    base_rate = statistics.mean(labels)
    result["event_calibration"] = {
        "event": "weighted_dense_rank_sum_strictly_below_frozen_threshold",
        "thresholds": sorted({row["prediction"]["threshold"] for row in scored}),
        "observed": sum(labels), "predicted_expected": sum(probabilities),
        "brier": statistics.mean((p - y) ** 2 for p, y in zip(probabilities, labels)),
        "empirical_constant_brier_in_sample_not_deployable": base_rate * (1 - base_rate),
        "zero_event_probability_if_independent_not_valid_uncertainty": math.prod(1 - p for p in probabilities),
        "bins": [],
    }
    for low, high in [(0, .02), (.02, .03), (.03, .04), (.04, .05), (.05, 1.01)]:
        subset = [row for row in scored if low <= row["probability"] < high]
        if subset:
            result["event_calibration"]["bins"].append({"lower": low, "upper": high,
                "count": len(subset), "mean_probability": statistics.mean(row["probability"] for row in subset),
                "observed_events": sum(row["predicted_event_occurred"] for row in subset)})
    result["winners"] = [{key: row[key] for key in ["epoch", "uid", "actual_score", "probability", "observed_selection_score",
                         "probability_order_published", "negative_median_order_published", "negative_p90_order_published"]}
                         for row in scored if row["winner"]]
    result["per_epoch"] = [{"epoch": epoch, "published": len(group),
        "scored": sum(row["scored"] for row in group), "field_size": group[0]["field_size"],
        "top10": sum(row.get("top10", False) for row in group), "wins": sum(row.get("winner", False) for row in group),
        "probability_pairwise_accuracy": concordance([row for row in group if row["scored"]], "probability"),
        "head_uid": min(group, key=lambda row: row["probability_order_published"])["uid"],
        "head_actual_rank": min(group, key=lambda row: row["probability_order_published"]).get("actual_rank")}
        for epoch, group in grouped.items()]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=25155)
    parser.add_argument("--last", type=int, default=25172)
    parser.add_argument("--host", default="root@159.112.204.190")
    parser.add_argument("--port", type=int, default=61170)
    parser.add_argument("--key", default="/Users/tvgdeena/.ssh/id_ed25519")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--reuse-input", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs_path = args.output / "inputs.json"
    if args.reuse_input:
        data = json.loads(inputs_path.read_text())
    else:
        histories = remote_history(list(range(args.first, args.last + 1)), args.host, args.port, args.key)
        with ThreadPoolExecutor(max_workers=6) as pool:
            leaderboards = list(pool.map(lambda history: public_field(history["epoch"], history["target"]), histories))
        rows, provenance, fields = [], [], []
        for history, leaderboard in zip(histories, leaderboards):
            rows.extend(joined_rows(history, leaderboard))
            provenance.append({"epoch": history["epoch"], "history_sha256": history["history_sha256"],
                               "api_sha256": leaderboard["response_sha256"], "url": leaderboard["url"]})
            fields.append({"epoch": history["epoch"], "field": [{key: value for key, value in item.items() if key != "hotkey"}
                                                                  for item in leaderboard["field"]]})
        data = {"schema_version": 1, "captured_at": datetime.now(timezone.utc).isoformat(),
                "match_key": "exact_archived_hotkey_and_sequence_sha256", "rows": rows,
                "provenance": provenance, "validator_fields": fields,
                "validator_replayed_rows": sum(len(item["field"]) for item in fields)}
        inputs_path.write_text(json.dumps(data, indent=2) + "\n")
    result = {"schema_version": 1, "primary": summarize(data["rows"], args.repeats),
              "post_tnp_fix": summarize([row for row in data["rows"] if row["epoch"] >= 25164], args.repeats),
              "recent_three": summarize([row for row in data["rows"] if row["epoch"] >= 25170], args.repeats),
              "validator_replayed_rows": data["validator_replayed_rows"],
              "limitations": ["Frozen forecast audit conditional on submissions reaching leaderboard; missing entries retained for top-k yield.",
                              "No counterfactual labels for unsubmitted candidates; cannot infer optimal full-pool selection.",
                              "Epochs may share evolutionary ancestry and conditions; bootstrap does not remove cross-epoch dependence.",
                              "Multiple diagnostic predictors; no fitted policy and no deployment.",
                              "History artifacts are retained records, not independently timestamp-signed forecasts.",
                              "Target event is weighted score < 60, not actual rank 1 or top 10."]}
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "published": result["primary"]["published"],
                      "scored": result["primary"]["scored"], "validator_replayed_rows": result["validator_replayed_rows"],
                      "output": str(args.output)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1)
