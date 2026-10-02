"""Create thesis figures from the archived leather run without retraining.

This reads the existing metrics and raw maps. It checks their recorded hashes
and preserves differences from the original runtime in the output metadata.
It does not change the repository's strict comparability gate or its reports.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="tesi-mpl-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image

THESIS = Path(__file__).resolve().parents[1]
REPOSITORY = THESIS.parent / "repository_pw"
sys.path.insert(0, str(REPOSITORY))

from audit_comparability import (
    configured_manifest_path,
    configured_prediction_manifest_path,
    current_training_config_digest,
    evaluation_hash,
    implementation_source_snapshot,
    load_model_config,
)
from experiment_protocol import (
    config_snapshot,
    dataset_snapshot,
    environment_snapshot,
    file_sha256,
    stable_json_digest,
)

MODELS = ("CCW-SSIM", "PaDiM", "RD4AD", "FastFlow")
COLORS = ("#4053a4", "#008577", "#c77800", "#8f429c")
KINDS = ("color", "cut", "fold", "glue", "poke")
OUTPUT = THESIS / "figure" / "capitolo_4"
DATASET = REPOSITORY / "MVTec_Data" / "leather"


def require_equal(actual, expected, description):
    if actual != expected:
        raise RuntimeError(f"Archived artifact differs: {description}")


def load_archived_run(name):
    model = load_model_config(name)
    training_path = configured_manifest_path(model)
    prediction_path = configured_prediction_manifest_path(model)
    training = json.loads(training_path.read_text())
    prediction = json.loads(prediction_path.read_text())
    result_path = REPOSITORY / "results" / name / "leather"
    result = json.loads(result_path.read_text())
    metadata = result["_metadata"]
    maps_path = REPOSITORY / "results" / name / "Maps" / "MVTec_Data" / "leather" / "test"

    require_equal(dataset_snapshot(DATASET / "train" / "good"), training["dataset"], name + " training data")
    require_equal(dataset_snapshot(DATASET / "test"), prediction["test_dataset"], name + " test data")
    require_equal(
        metadata["dataset"]["leather"]["ground_truth"],
        dataset_snapshot(DATASET / "ground_truth"),
        name + " ground truth",
    )
    require_equal(
        current_training_config_digest(model),
        training["configuration_sha256"],
        name + " training configuration",
    )
    prediction_sections = ["MODEL", "PREDICTION"] if model.config.has_section("MODEL") else ["PREDICTION"]
    require_equal(
        stable_json_digest(config_snapshot(model.config, prediction_sections)),
        prediction["configuration_sha256"],
        name + " prediction configuration",
    )
    for phase, manifest in (("training", training), ("prediction", prediction)):
        require_equal(
            implementation_source_snapshot(model, phase),
            manifest["implementation_source_sha256"],
            name + " " + phase + " sources",
        )
        require_equal(
            file_sha256(REPOSITORY / "experiment_protocol.py"),
            manifest["protocol_source_sha256"],
            name + " provenance implementation",
        )
        require_equal(
            file_sha256(Path(manifest["checkpoint"]["path"])),
            manifest["checkpoint"]["sha256"],
            name + " checkpoint",
        )
    require_equal(file_sha256(training_path), prediction["training_manifest"]["sha256"], name + " training link")
    require_equal(
        file_sha256(prediction_path),
        metadata["provenance"]["prediction_manifest_sha256"],
        name + " evaluation link",
    )
    require_equal(metadata["provenance"]["status"], "verified", name + " historical provenance")
    require_equal(evaluation_hash(model), metadata["evaluator_source_sha256"], name + " evaluator")
    maps_snapshot = dataset_snapshot(maps_path)
    require_equal(maps_snapshot, prediction["prediction"]["anomaly_maps"], name + " saved maps")
    require_equal(maps_snapshot, metadata["anomaly_maps"]["leather"], name + " evaluated maps")

    runtime = environment_snapshot()
    changes = {}
    for phase, manifest in (("training", training), ("prediction", prediction), ("evaluation", metadata)):
        changes[phase] = {
            key: {"recorded": value, "current": runtime.get(key)}
            for key, value in manifest["environment"].items()
            if value != runtime.get(key)
        }
    return result["leather"], maps_path, {
        "result_sha256": file_sha256(result_path),
        "training_manifest_sha256": file_sha256(training_path),
        "prediction_manifest_sha256": file_sha256(prediction_path),
        "maps_sha256": maps_snapshot["sha256"],
        "runtime_differences": changes,
        "training_budget": training["training_budget"],
    }


def save_figure(fig, stem):
    fig.savefig(OUTPUT / (stem + ".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT / (stem + ".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_case(relative, results, map_paths):
    with Image.open(DATASET / "test" / relative) as image:
        original = np.asarray(image.convert("RGB"))
    kind = relative.parts[0]
    if kind == "good":
        mask = np.zeros(original.shape[:2], dtype=bool)
    else:
        with Image.open(DATASET / "ground_truth" / kind / (relative.stem + "_mask.png")) as image:
            mask = np.asarray(image.convert("L")) > 0

    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.1), layout="constrained")
    axes = axes.ravel()
    axes[0].imshow(original)
    axes[0].set_title("Immagine: " + relative.as_posix())
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")
    for axis, name in zip(axes[2:], MODELS):
        raw = tifffile.imread(map_paths[name] / relative.with_suffix(".tiff"))
        result = results[name]
        low, high = result["map_score_min_raw"], result["map_score_max_raw"]
        normalized = np.clip((raw.astype(np.float64) - low) / (high - low), 0, 1)
        heatmap = axis.imshow(normalized, cmap="magma", vmin=0, vmax=1)
        if mask.any():
            axis.contour(mask, levels=[0.5], colors=["#00e676"], linewidths=0.7)
        axis.set_title(name)
    for axis in axes:
        axis.axis("off")
    fig.colorbar(heatmap, ax=axes.tolist(), shrink=0.8, label="Punteggio normalizzato per modello")
    stem = "leather_" + kind + "_" + relative.stem
    save_figure(fig, stem)
    return stem


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    results, map_paths, evidence = {}, {}, {}
    for name in MODELS:
        results[name], map_paths[name], evidence[name] = load_archived_run(name)
    require_equal(
        results[MODELS[0]]["evaluation_protocol"],
        results[MODELS[-1]]["evaluation_protocol"],
        "shared metric protocol",
    )

    # Inspect the raw maps using the same sampled normalization as the evaluator.
    diagnostics = {}
    for name in MODELS:
        result = results[name]
        records = []
        sample_min, sample_max = float("inf"), float("-inf")
        for image_path in sorted((DATASET / "test").glob("*/*.png")):
            relative = image_path.relative_to(DATASET / "test")
            raw = tifffile.imread(map_paths[name] / relative.with_suffix(".tiff"))
            sample = raw[::4, ::4]
            sample_min = min(sample_min, float(sample.min()))
            sample_max = max(sample_max, float(sample.max()))
            records.append((relative, float(raw.max()), sample.copy()))
        groups = {kind: {"images": 0, "fn_image": 0, "fp_image": 0} for kind in (*KINDS, "good")}
        totals = {"tp": 0, "fp": 0, "fn": 0}
        false_positives, false_negatives = [], []
        image_low, image_high = result["score_min_raw"], result["score_max_raw"]
        for relative, maximum, sample in records:
            kind = relative.parts[0]
            groups[kind]["images"] += 1
            image_score = (maximum - image_low) / (image_high - image_low)
            predicted_image = image_score >= result["optimal_threshold_image"]
            if kind == "good":
                sampled_mask = np.zeros(sample.shape, dtype=bool)
                if predicted_image:
                    groups[kind]["fp_image"] += 1
                    false_positives.append(relative.as_posix())
            else:
                with Image.open(DATASET / "ground_truth" / kind / (relative.stem + "_mask.png")) as image:
                    sampled_mask = (np.asarray(image.convert("L")) > 0)[::4, ::4]
                if not predicted_image:
                    groups[kind]["fn_image"] += 1
                    false_negatives.append(relative.as_posix())
            normalized = (sample.astype(np.float64) - sample_min) / (sample_max - sample_min)
            predicted_pixel = normalized >= result["optimal_threshold_pixel"]
            totals["tp"] += int(np.count_nonzero(predicted_pixel & sampled_mask))
            totals["fp"] += int(np.count_nonzero(predicted_pixel & ~sampled_mask))
            totals["fn"] += int(np.count_nonzero(~predicted_pixel & sampled_mask))
        pixel_f1 = 2 * totals["tp"] / (2 * totals["tp"] + totals["fp"] + totals["fn"])
        if not np.isclose(pixel_f1, result["f1_max_pixel"], atol=1e-10):
            raise RuntimeError(f"Pixel F1 inconsistent for {name}: {pixel_f1}")
        diagnostics[name] = {
            "groups_at_global_image_test_oracle": groups,
            "false_positive_images": false_positives,
            "false_negative_images": false_negatives,
            "pixel_f1_recomputed_at_saved_threshold": pixel_f1,
            "sampled_map_min": sample_min,
            "sampled_map_max": sample_max,
        }

    selected = {kind: Path(kind) / "000.png" for kind in KINDS}
    # A deliberately problematic normal sample: highest global CCW-SSIM score.
    selected["good"] = max(
        (Path("good") / path.name for path in (DATASET / "test" / "good").glob("*.png")),
        key=lambda rel: float(tifffile.imread(map_paths["CCW-SSIM"] / rel.with_suffix(".tiff")).max()),
    )
    ccw_groups = diagnostics["CCW-SSIM"]["groups_at_global_image_test_oracle"]
    hardest_kind = max(KINDS, key=lambda kind: ccw_groups[kind]["fn_image"])
    selected["missed_defect"] = Path(next(
        rel for rel in diagnostics["CCW-SSIM"]["false_negative_images"]
        if Path(rel).parts[0] == hardest_kind
    ))
    panels = {kind: plot_case(relative, results, map_paths) for kind, relative in selected.items()}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), layout="constrained")
    for name, color in zip(MODELS, COLORS):
        result = results[name]
        axes[0].plot(
            result["classification_roc_curve_fpr"],
            result["classification_roc_curve_tpr"],
            label=name,
            color=color,
        )
        axes[1].plot(result["pro_curve_fpr"], result["pro_curve_pro"], label=name, color=color)
    axes[0].plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC a livello di immagine", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(xlabel="FPR", ylabel="PRO", title="Curva PRO", xlim=(0, 0.3), ylim=(0, 1.02))
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(loc="lower right")
    save_figure(fig, "curve_roc_pro")

    summary = {
        "scope": "Visualization and descriptive analysis of the archived leather run; no retraining.",
        "strict_current_audit": "Fails on the recorded versus current Linux platform; historical reports are not overwritten.",
        "selection_rule": (
            "000.png for each defect type; normal image with highest CCW-SSIM maximum; "
            "first CCW-SSIM false negative in the type with most false negatives."
        ),
        "selected_images": {kind: rel.as_posix() for kind, rel in selected.items()},
        "panels": panels,
        "evidence": evidence,
        "diagnostics": diagnostics,
    }
    (OUTPUT / "provenienza_figure.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_images": summary["selected_images"],
        "diagnostics": diagnostics,
        "output_directory": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
