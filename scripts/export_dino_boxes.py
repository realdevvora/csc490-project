"""Add box IoU to saved DINO A/B/C evaluations, without models or retraining.

Accept full-page A directories and/or head evaluation directories. Head runs
reuse selection.json's already selected checkpoints; no new model selection.
Outputs go to a fresh directory and source artifacts remain untouched.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from regression_lens.data.records import load_records, operator_names
from regression_lens.data.splits import TEAM_SPLIT
from regression_lens.evaluation.dino_boxes import BOX_RULE, heatmap_box
from regression_lens.evaluation.scoring import load_predictions, score_predictions, validate_predictions

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def plans(evaluation):
    config = read(evaluation / "config.json")
    if not (evaluation / "complete.json").is_file():
        raise ValueError(f"Incomplete evaluation: {evaluation}")
    if config.get("smoke") or config.get("smoke_test"):
        raise ValueError(f"Smoke evaluations are not full validation results: {evaluation}")
    if "checkpoints" not in config:
        if config["output_scale"] != 2 or config["subset"] != "val":
            raise ValueError("Expected 2px full-page validation maps")
        return config, [("A", None, None, evaluation, "cosine", config["metadata_sha256"])]
    selection = read(evaluation / "selection.json")
    runs = []
    for condition in ("B", "C"):
        chosen = selection["selected"][condition]
        if sorted(chosen["seeds"]) != ["490", "491", "492"]:
            raise ValueError("Expected three original seeds per condition")
        for seed, entry in sorted(chosen["seeds"].items()):
            name = entry["name"]
            # Reject path traversal in a source manifest.
            if Path(name).name != name:
                raise ValueError("Invalid checkpoint name")
            checkpoint = config["checkpoints"][name]
            run = checkpoint["config"]
            if (checkpoint["step"] != chosen["step"] or run["condition"] != condition
                    or run["seed"] != int(seed)):
                raise ValueError("Checkpoint selection/configuration mismatch")
            runs.append((condition, int(seed), chosen["step"], evaluation/name, "location",
                         run["cache"]["manifest"]["metadata_sha256"]))
    return config, runs


def export(evaluations, data_root, split_path, out):
    records = load_records(data_root)
    by_id = {r["sample_id"]: r for r in records}
    operators = operator_names(records)
    split = read(split_path)
    expected = {r["sample_id"] for r in records if r["parent_id"] in split["val"]}
    metadata_hash = sha(data_root/"master_dataset_with_benign.json")
    jobs = []
    names = set()
    # Preflight missing maps/coverage before creating any output files.
    for evaluation in evaluations:
        config, runs = plans(evaluation)
        if config["split_sha256"] != sha(split_path):
            raise ValueError("Source split hash differs from requested split")
        ids = config["sample_ids"]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError("Evaluation config does not cover the full validation split")
        for condition, seed, step, directory, key, expected_hash in runs:
            if expected_hash != metadata_hash:
                raise ValueError("Source dataset fingerprint differs from current metadata")
            name = evaluation.name + ("__A" if seed is None else f"__{directory.name}")
            if name in names:
                raise ValueError("Evaluation directory names must be unique")
            names.add(name)
            source = directory/"predictions.jsonl"
            predictions = load_predictions(source)
            validate_predictions(predictions, by_id, operators, expected)
            maps = directory/"maps" if key == "cosine" else directory
            paths = {sample_id: maps/f"{i:05d}.npz" for i, sample_id in enumerate(ids)}
            missing = [p for p in paths.values() if not p.is_file()]
            if missing:
                raise ValueError(f"{len(missing)} missing maps; first: {missing[0]}. "
                                 "Use the original lab evaluation, not the map-free results bundle.")
            jobs.append((name, condition, seed, step, evaluation, config, source, predictions, key, paths))
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be empty; preserve previous results")
    # Never write inside source artifacts, even to a new child directory.
    if any(e.resolve() == out.resolve() or e.resolve() in out.resolve().parents for e in evaluations):
        raise ValueError("Output must be outside source evaluation directories")
    out.mkdir(parents=True, exist_ok=True)
    write(out/"config.json", dict(rule=BOX_RULE, map_threshold="0.5 * valid peak per image",
          selection="Reuse original selected checkpoints; no IoU tuning", scale=2,
          split_sha256=sha(split_path), metadata_sha256=metadata_hash,
          exporter_sha256=sha(Path(__file__)),
          box_code_sha256=sha(REPO/"src/regression_lens/evaluation/dino_boxes.py"),
          interpretation="Post-hoc validation localization extension, not an independent test"))
    results = []
    for name, condition, seed, step, evaluation, config, source, predictions, key, paths in jobs:
        directory = out/name
        directory.mkdir()
        original = score_predictions(predictions, records, operators, expected_ids=expected)
        rows, audits = [], []
        for prediction in predictions:
            sample_id = prediction["sample_id"]
            record = by_id[sample_id]
            # Dimensions only; never pass mutation boxes or labels to extraction.
            width, height = [max(record["size_ref"][i], record["size_cand"][i]) for i in (0, 1)]
            path = paths[sample_id]
            with np.load(path, allow_pickle=False) as archive:
                box, audit = heatmap_box(archive[key], archive["valid"], width, height)
            row = dict(prediction, box=box)
            # Preserve absent/None points too: prevent scorer's box-centre fallback
            # from changing original regression pointing coverage silently.
            if prediction.get("point") is None and record["label"] == "regression":
                raise ValueError("Missing original regression point; cannot preserve pointing coverage")
            rows.append(row)
            audits.append(dict(sample_id=sample_id, map_sha256=sha(path), box=box, **audit))
        metrics = score_predictions(rows, records, operators, threshold=original["threshold"],
                                    threshold_selection=original["threshold_selection"], expected_ids=expected)
        for field in original:
            if field != "localization" and original[field] != metrics[field]:
                raise ValueError(f"Non-localization metric changed: {field}")
        for field in ("pointing_accuracy", "n_point"):
            if original["localization"][field] != metrics["localization"][field]:
                raise ValueError("Pointing metric changed")
        for filename, entries in [("predictions.jsonl", rows), ("boxes.jsonl", audits)]:
            (directory/filename).write_text("".join(json.dumps(r)+"\n" for r in entries), encoding="utf-8")
        write(directory/"metrics.json", metrics)
        write(directory/"provenance.json", dict(source_predictions=str(source),
              source_predictions_sha256=sha(source), source_config_sha256=sha(evaluation/"config.json"),
              source_selection_sha256=sha(evaluation/"selection.json") if seed is not None else None,
              condition=condition, seed=seed, step=step, map_key=key,
              scores_points_and_detection_metrics_unchanged=True,
              zero_map_fallbacks=sum(r["zero_map_fallback"] for r in audits)))
        results.append(dict(evaluation=evaluation.name, run=name, condition=condition,
                            seed=seed, step=step, localization=metrics["localization"]))
        print(f"{name}: IoU {metrics['localization']['mean_iou']:.4f}; scores/pointing unchanged", flush=True)
    grouped = defaultdict(list)
    for row in results:
        grouped[(row["evaluation"], row["condition"])].append(row)
    aggregates = []
    lines = ["| Evaluation | Condition | Runs | Mean box IoU | Seed SD | Boxes/run |",
             "|---|---|---:|---:|---:|---:|"]
    for (evaluation, condition), rows in grouped.items():
        ious = [r["localization"]["mean_iou"] for r in rows]
        sd = float(np.std(ious, ddof=1)) if len(ious) > 1 else None
        aggregates.append(dict(evaluation=evaluation, condition=condition, mean_iou=float(np.mean(ious)),
                               seed_sd=sd, runs=rows))
        lines.append(f"| {evaluation} | {condition} | {len(rows)} | {np.mean(ious):.4f} | "
                     + (f"{sd:.4f}" if sd is not None else "—")
                     + f" | {rows[0]['localization']['n_box']} |")
    write(out/"summary.json", aggregates)
    (out/"comparison.md").write_text("# DINO heatmap-derived boxes: validation extension\n\n"
          + "\n".join(lines) + "\n\nFixed half-peak rule; no IoU-selected checkpoints/thresholds. "
          "Original scores and pointing preserved. IoU targets diff_bbox, including for reflow. "
          "These boxes are post-processed proposals, not segmentation masks.\n", encoding="utf-8")
    write(out/"complete.json", dict(runs=len(results), pairs_per_run=len(expected), rule=BOX_RULE))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, nargs="+", required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", type=Path, default=REPO/TEAM_SPLIT)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    try:
        export(args.evaluation, args.data_root, args.split, args.out)
    except (ValueError, FileNotFoundError, KeyError) as error:
        p.exit(1, f"Box export failed: {error}\n")


if __name__ == "__main__":
    main()
