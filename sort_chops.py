from __future__ import annotations
import json
import logging
import shutil
import sys
from itertools import count
from pathlib import Path
import numpy as np

from audio_model import (
    BATCH_SIZE, EXTENSIONS, LABELS, LOGGER, MODEL_DIR, THRESHOLD,
    FeatureEncoder, read_audio, score_features
)

def find_chops(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir()
                  if not path.name.startswith(".") and not path.is_symlink()
                  and path.is_file() and path.suffix.lower() in EXTENSIONS)

def move_chop(source: Path, folder: Path) -> Path:
    folder.mkdir(exist_ok=True)
    for number in count():
        suffix = f"_{number}" if number else ""
        destination = folder / f"{source.stem}{suffix}{source.suffix}"
        try:
            target = destination.open("xb")
        except FileExistsError:
            continue
        try:
            with target, source.open("rb") as original:
                shutil.copyfileobj(original, target)
            shutil.copystat(source, destination)
            source.unlink()
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        return destination

def sort_folder(folder: Path) -> dict:
    folder = folder.resolve()
    files = find_chops(folder)
    for label in LABELS.values():
        destination = folder / label
        if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
            raise ValueError(f"Output must be a regular directory: {destination}")
    summary = {"good": 0, "bad": 0, "errors": 0}
    if not files:
        LOGGER.info("No unsorted audio files in %s", folder)
        return summary
    classifier = json.loads((MODEL_DIR / "classifier.json").read_text(encoding="utf-8"))
    encoder = FeatureEncoder(MODEL_DIR / "encoder")
    LOGGER.info("Classifying %d audio files in %s", len(files), folder)
    decisions = []
    for start in range(0, len(files), BATCH_SIZE):
        readable, waveforms = [], []
        for path in files[start:start + BATCH_SIZE]:
            try:
                waveform = read_audio(path)
            except (ValueError, OSError, RuntimeError) as exc:
                LOGGER.error("Leaving %s in place: %s", path.name, exc)
                summary["errors"] += 1
                continue
            readable.append(path)
            waveforms.append(waveform)
        if not readable:
            continue
        probabilities = score_features(encoder.encode(waveforms), classifier)
        if len(probabilities) != len(readable) or not np.isfinite(probabilities).all():
            raise ValueError("Model returned invalid scores; no files have been moved")
        decisions.extend((path, LABELS[int(score >= THRESHOLD)], float(score))
                         for path, score in zip(readable, probabilities))
        LOGGER.info("Analyzed %d/%d files", min(start + BATCH_SIZE, len(files)), len(files))
    for path, label, score in decisions:
        try:
            destination = move_chop(path, folder / label)
        except (ValueError, OSError) as exc:
            LOGGER.error("Could not move %s: %s", path.name, exc)
            summary["errors"] += 1
            continue
        summary[label] += 1
        print(f"Moved: {path.name} -> {destination.relative_to(folder)} (good score: {score:.4f})")
    return summary

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) != 2:
        LOGGER.error('Usage: python3 sort_vocals.py "folder"')
        return 1
    try:
        result = sort_folder(Path(sys.argv[1]))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        LOGGER.error("%s", exc)
        return 1
    print(f"Sorted: {result['good']} good, {result['bad']} bad; {result['errors']} errors")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())