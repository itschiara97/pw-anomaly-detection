"""User-approved, read-only CCW checkpoint diagnostic on twelve normal patches."""

from __future__ import annotations

import configparser
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from figure_capitolo_4 import DATASET, REPOSITORY, THESIS, load_archived_run

import cv2
import numpy as np
import tensorflow as tf

CCW = REPOSITORY / "CCW-SSIM"
sys.path.insert(0, str(CCW))
from ColorUtils import prepare_dataset_colorssim
from NNModels import Model_noise_skip_bigger_old
import Training
from experiment_protocol import file_sha256


def main():
    _, _, evidence = load_archived_run("CCW-SSIM")
    config = configparser.ConfigParser()
    config.read(CCW / "Config/Parameters.ini")
    scale = float(config["TRAINING"]["LabNormalizationMax"])
    tf.keras.backend.set_floatx("float64")
    tf.keras.utils.set_random_seed(1024)
    Training.patch_size = 128
    Training.window_size = int(config["LOSSFUNCTION"]["WindowSize"])
    Training.scales = int(config["LOSSFUNCTION"]["Scales"])
    Training.orients = int(config["LOSSFUNCTION"]["Orientations"])
    Training.pad_size = int(config["LOSSFUNCTION"]["pad_size"])
    images = sorted((DATASET / "test/good").glob("*.png"))[:3]
    positions = ((0, 0), (448, 448), (896, 896), (128, 640))
    patches, selection = [], []
    for path in images:
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        for y, x in positions:
            patches.append(image[y:y + 128, x:x + 128])
            selection.append({"image": path.relative_to(DATASET / "test").as_posix(),
                              "image_sha256": file_sha256(path), "top_left_yx": [y, x]})
    inputs = tf.constant(prepare_dataset_colorssim(np.asarray(patches)) / scale, dtype=tf.float64)
    model = Model_noise_skip_bigger_old(input_shape=(None, None, 3))
    weights = CCW / config["PREDICTION"]["WeightsFile"]
    model.load_weights(weights)
    predictions = model(inputs, training=False)
    x, raw = inputs.numpy(), predictions.numpy()
    corrected = raw.copy()
    corrected[..., 0] *= -1
    structural_raw = float(Training.cwssim_only_metric(inputs, predictions))
    structural_flipped = float(Training.cwssim_only_metric(inputs, tf.constant(corrected)))
    total_loss = float(Training.color_cwssim_loss(inputs, predictions))
    mse_ab = float(np.mean((x[..., 1:3] - raw[..., 1:3]) ** 2))
    records = []
    for i, sample in enumerate(selection):
        records.append({**sample,
            "input_L_mean_Lab_units": float(np.mean(x[i, ..., 0]) * scale),
            "output_L_mean_Lab_units": float(np.mean(raw[i, ..., 0]) * scale),
            "raw_L_rmse_Lab_units": float(np.sqrt(np.mean((x[i, ..., 0] - raw[i, ..., 0])**2)) * scale),
            "flipped_L_rmse_Lab_units": float(np.sqrt(np.mean((x[i, ..., 0] - corrected[i, ..., 0])**2)) * scale),
            "ab_rmse_normalized": float(np.sqrt(np.mean((x[i, ..., 1:3] - raw[i, ..., 1:3])**2))),
        })
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "User approved diagnostic only; no training or official prediction rerun",
        "scope": "12 patches 128x128 from first three normal test images; CPU only",
        "checkpoint_sha256": file_sha256(weights), "verifier_sha256": file_sha256(Path(__file__)),
        "tensorflow": tf.__version__, "artifact_evidence": evidence,
        "patches": records,
        "structural_loss_raw": structural_raw, "structural_loss_flipped_L": structural_flipped,
        "structural_sign_difference": abs(structural_raw - structural_flipped),
        "total_training_loss": total_loss, "normalized_ab_mse": mse_ab,
        "training_color_penalty_mse_div_45": mse_ab / 45,
        "inference_color_penalty_equivalent_10_mse_before_spatial_averaging": 10 * mse_ab,
        "weight_ratio_inference_to_training": 450,
        "total_minus_structural": total_loss - structural_raw,
        "limitations": "No full-image reconstruction, performance metric, gradient comparison or causal ablation",
    }
    output = THESIS / "verifiche"
    output.mkdir(exist_ok=True)
    path = output / "diagnostica_ccw_checkpoint.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("artifact_evidence", "patches")}, indent=2))
    for record in records:
        print(json.dumps(record))


if __name__ == "__main__":
    main()
