from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
import logging
import sys
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from audio_model import (
    BATCH_SIZE, DATA_DIR, EXTENSIONS, FEATURE_VERSION,
    LABELS, LOGGER, MODEL_DIR, MODEL_ID, MODEL_REVISION, SAMPLE_RATE, THRESHOLD,
    FeatureEncoder, read_audio, score_features
)

TEST_SIZE = 0.2
CV_FOLDS = 5
SEED = 42

@dataclass(frozen=True)
class Example:
    path: str
    label: int
    audio_hash: str
    duration: float

def load_dataset() -> tuple[list[Example], list[dict]]:
    examples, duplicates = [], []
    seen: dict[str, Example] = {}
    for label, name in LABELS.items():
        files = sorted(path for path in (DATA_DIR / name).rglob("*")
                       if path.is_file() and not path.name.startswith(".")
                       and path.suffix.lower() in EXTENSIONS)
        for path in files:
            audio = read_audio(path)
            digest = hashlib.sha256(audio.astype("<f4", copy=False).tobytes()).hexdigest()
            relative_path = str(path.relative_to(DATA_DIR))
            previous = seen.get(digest)
            if previous:
                duplicates.append({"file": relative_path, "duplicate_of": previous.path})
                continue
            item = Example(relative_path, label, digest, len(audio) / SAMPLE_RATE)
            seen[digest] = item
            examples.append(item)
    return examples, duplicates

def extract_features(examples: list[Example], encoder: FeatureEncoder) -> np.ndarray:
    features = []
    for start in range(0, len(examples), BATCH_SIZE):
        batch = examples[start:start + BATCH_SIZE]
        waveforms = [read_audio(DATA_DIR / example.path) for example in batch]
        features.extend(encoder.encode(waveforms))
        LOGGER.info("Extracted %d/%d clips", min(start + BATCH_SIZE, len(examples)), len(examples))
    return np.stack(features)

def write_csv(filename: str, rows: list[dict]) -> None:
    with (MODEL_DIR / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = (probabilities >= THRESHOLD).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision_good": float(average_precision_score(labels, probabilities)),
        "confusion_matrix_labels": ["bad", "good"],
        "confusion_matrix_rows_actual_columns_predicted": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(labels, predicted, labels=[0, 1],
                                target_names=["bad", "good"], output_dict=True, zero_division=0),
    }

def export_classifier(pipeline) -> dict:
    scale = pipeline.named_steps["scale"]
    head = pipeline.named_steps["classifier"]
    assert head.classes_.tolist() == [0, 1]
    return {
        "format_version": 1,
        "feature_version": FEATURE_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "classes": ["bad", "good"],
        "threshold": THRESHOLD,
        "scaler_mean": scale.mean_.tolist(),
        "scaler_scale": scale.scale_.tolist(),
        "coefficients": head.coef_[0].tolist(),
        "intercept": float(head.intercept_[0]),
    }

def train() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    examples, duplicates = load_dataset()
    labels = np.asarray([e.label for e in examples])
    train_indices, test_indices = train_test_split(
        np.arange(len(labels)), test_size=TEST_SIZE, random_state=SEED, stratify=labels)
    LOGGER.info("Unique clips: %d bad, %d good; %d train, %d test; %d duplicates removed",
                sum(labels == 0), sum(labels == 1), len(train_indices), len(test_indices), len(duplicates))
    encoder = FeatureEncoder()
    features = extract_features(examples, encoder)
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(solver="lbfgs", max_iter=3000, tol=1e-5, random_state=SEED)),
    ])
    search = GridSearchCV(pipeline, {
        "classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    }, scoring="balanced_accuracy",
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED),
        n_jobs=1, error_score="raise")
    search.fit(features[train_indices], labels[train_indices])
    classifier = export_classifier(search.best_estimator_)
    probabilities = score_features(features[test_indices], classifier)
    metrics = classification_metrics(labels[test_indices], probabilities)
    majority = int(np.argmax(np.bincount(labels[train_indices], minlength=2)))
    metrics["majority_baseline_accuracy"] = float(np.mean(labels[test_indices] == majority))
    metrics["majority_baseline_balanced_accuracy"] = 0.5
    report = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "feature_version": FEATURE_VERSION, "feature_count": features.shape[1],
        "seed": SEED, "threshold": THRESHOLD,
        "train_count": len(train_indices), "test_count": len(test_indices),
        "unique_class_counts": {name: int(sum(labels == label)) for label, name in LABELS.items()},
        "duplicates_removed": duplicates,
        "best_parameters": search.best_params_,
        "cv_folds": CV_FOLDS,
        "cv_candidates": [
            {"parameters": parameters,
             "mean_cv_balanced_accuracy": float(search.cv_results_["mean_test_score"][i]),
             "std_cv_balanced_accuracy": float(search.cv_results_["std_test_score"][i])}
            for i, parameters in enumerate(search.cv_results_["params"])],
        "best_cv_balanced_accuracy": float(search.best_score_),
        "held_out_test": metrics,
        "versions": {name: importlib.metadata.version(name) for name in
                     ["torch", "transformers", "numpy", "scipy", "scikit-learn", "soundfile"]},
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    encoder.save(MODEL_DIR / "encoder")
    for filename, contents in {"classifier.json": classifier, "report.json": report}.items():
        (MODEL_DIR / filename).write_text(json.dumps(contents, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved model: {MODEL_DIR.resolve()}")
    print(f"Best training CV balanced accuracy: {search.best_score_:.1%}")
    print(f"Held-out accuracy: {metrics['accuracy']:.1%}")
    print(f"Held-out balanced accuracy: {metrics['balanced_accuracy']:.1%}")
    print(f"Held-out ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"Best parameters: {search.best_params_}")

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        train()
    except (ValueError, OSError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())