"""Check archived maps and recalculate metrics without running any model.

Outputs are new audit evidence in the thesis directory. Historical results,
manifests, configurations and model sources are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

from figure_capitolo_4 import DATASET, MODELS, REPOSITORY, THESIS, load_archived_run

sys.path.insert(0, str(REPOSITORY / "PaDiM"))
from Evaluation.evaluate_experiment import calculate_au_pro_au_roc
from experiment_protocol import file_sha256


def verify_model(name):
    archived, maps_path, provenance = load_archived_run(name)
    gt_paths, prediction_paths = [], []
    diagnostics = []
    for image in sorted((DATASET / "test").glob("*/*.png")):
        relative = image.relative_to(DATASET / "test")
        prediction_path = maps_path / relative.with_suffix(".tiff")
        raw = tifffile.imread(prediction_path)
        if raw.shape != (1024, 1024) or raw.dtype != np.float32 or not np.isfinite(raw).all():
            raise ValueError(f"Invalid map {prediction_path}")
        y, x = np.unravel_index(np.argmax(raw), raw.shape)
        diagnostics.append({
            "image": relative.as_posix(),
            "maximum": float(raw.max()),
            "minimum": float(raw.min()),
            "maximum_yx": [int(y), int(x)],
            "max_border_distance": int(min(y, x, 1023 - y, 1023 - x)),
            "zero_pixel_count": int(np.count_nonzero(raw == 0)),
        })
        gt_paths.append(
            None if relative.parts[0] == "good" else
            str(DATASET / "ground_truth" / relative.parent / (relative.stem + "_mask.png"))
        )
        # The archived evaluator's TIFF reader takes the stem, not the suffix.
        prediction_paths.append(str(prediction_path.with_suffix("")))
    protocol = archived["evaluation_protocol"]
    metadata = json.loads((REPOSITORY / "results" / name / "leather").read_text())["_metadata"]
    pro, image_auc, _, _, pixel_auc, extra = calculate_au_pro_au_roc(
        gt_paths, prediction_paths, metadata["pro_integration_limit"], protocol
    )
    computed = {"au_pro": pro, "classification_au_roc": image_auc,
                "segmentation_au_roc": pixel_auc}
    checked_extra = ("f1_max_image", "f1_max_pixel", "precision_image", "recall_image",
                     "precision_pixel", "recall_pixel", "optimal_threshold_image",
                     "optimal_threshold_pixel", "fpr_at_tpr95_image", "score_min_raw",
                     "score_max_raw", "map_score_min_raw", "map_score_max_raw")
    computed.update({key: extra[key] for key in checked_extra})
    checks = {}
    for key, value in computed.items():
        checks[key] = {"archived": archived[key], "recomputed": float(value),
                       "absolute_difference": abs(float(value) - archived[key])}
        if not np.isclose(value, archived[key], atol=1e-10, rtol=0):
            raise ValueError(f"Metric mismatch {name}: {key}: {checks[key]}")
    result = {
        "model": name, "verified_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Shared archived evaluator on all 124 full-resolution maps; no model execution",
        "verifier_sha256": file_sha256(Path(__file__)),
        "provenance": provenance, "evaluation_protocol": protocol,
        "metric_checks": checks, "map_diagnostics": diagnostics,
    }
    if name == "CCW-SSIM":
        history = REPOSITORY / name / "Weights/new_weights/ccwssim_leather_history.csv"
        with history.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        result["history_analysis"] = {
            "path": str(history), "sha256": file_sha256(history), "rows": len(rows),
            "epoch_indices": [int(row["epoch"]) for row in rows],
            "first": rows[0], "last": rows[-1],
            "max_loss_minus_structural": max(float(row["loss"]) - float(row["cwssim_only_metric"]) for row in rows),
            "last_loss_minus_structural": float(rows[-1]["loss"]) - float(rows[-1]["cwssim_only_metric"]),
            "history_binding": "Configured CSV path; historical training manifest does not authenticate this CSV hash",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    output = THESIS / "verifiche"
    output.mkdir(exist_ok=True)
    shared_protocol = None
    for name in args.models:
        print(f"VERIFY START {name}", flush=True)
        result = verify_model(name)
        if shared_protocol is None:
            shared_protocol = result["evaluation_protocol"]
        elif result["evaluation_protocol"] != shared_protocol:
            raise ValueError("Different evaluation protocol")
        path = output / f"metriche_{name}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"VERIFY PASS {name}: {len(result['metric_checks'])} metrics; evidence {path}", flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
