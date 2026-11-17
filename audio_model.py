from __future__ import annotations
import logging
import math
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly, stft
from scipy.special import expit
from transformers import ClapAudioModelWithProjection, ClapConfig, ClapFeatureExtractor

LOGGER = logging.getLogger("vocal-sorter")
MODEL_ID = "laion/clap-htsat-unfused"
MODEL_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
FEATURE_VERSION = 1
SAMPLE_RATE = 48000
EXTENSIONS = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3"}
LABELS = {0: "bad", 1: "good"}
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "artifacts" / "vocal-quality"
BATCH_SIZE = 4
THRESHOLD = 0.5

def read_audio(path: Path) -> np.ndarray:
    try:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, ValueError, sf.LibsndfileError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite samples in {path}")
    mono = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, rate // divisor)
    return np.ascontiguousarray(mono, dtype=np.float32)

def acoustic_features(audio: np.ndarray) -> np.ndarray:
    signal = resample_poly(audio, 1, 3).astype(np.float64)
    rate = SAMPLE_RATE // 3
    eps = 1e-12
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    edge = max(1, min(len(signal), int(0.01 * rate)))
    values = [
        np.log1p(len(audio) / SAMPLE_RATE),
        20 * np.log10(rms + eps),
        20 * np.log10(peak + eps),
        peak / (rms + eps),
        np.mean(np.abs(signal) >= 0.999),
        np.mean(np.abs(signal) < 0.001),
        np.mean(np.signbit(signal[1:]) != np.signbit(signal[:-1])) if len(signal) > 1 else 0,
        np.sqrt(np.mean(signal[:edge] ** 2)) / (rms + eps),
        np.sqrt(np.mean(signal[-edge:] ** 2)) / (rms + eps),
        abs(signal[0]) / (rms + eps),
        abs(signal[-1]) / (rms + eps),
    ]
    padded = np.pad(signal, (0, max(0, 512 - len(signal))))
    frequencies, _, spectrum = stft(padded, fs=rate, nperseg=512, noverlap=352, boundary=None, padded=False)
    power = np.abs(spectrum) ** 2
    total = power.sum(axis=0) + eps
    centroid = (frequencies[:, None] * power).sum(axis=0) / total
    bandwidth = np.sqrt(((frequencies[:, None] - centroid) ** 2 * power).sum(axis=0) / total)
    flatness = np.exp(np.mean(np.log(power + eps), axis=0)) / (np.mean(power, axis=0) + eps)
    cumulative = np.cumsum(power, axis=0)
    rolloff = frequencies[np.argmax(cumulative >= 0.85 * total, axis=0)]
    frame_rms = np.sqrt(np.mean(np.lib.stride_tricks.sliding_window_view(padded, 512)[::160] ** 2, axis=1))
    flux = np.sqrt(np.mean(np.diff(np.sqrt(power), axis=1) ** 2, axis=0))
    series = [centroid / rate, bandwidth / rate, flatness, rolloff / rate, 20 * np.log10(frame_rms + eps), flux]
    for sequence in series:
        values.extend([float(np.mean(sequence)) if sequence.size else 0,
                       float(np.std(sequence)) if sequence.size else 0,
                       float(np.percentile(sequence, 90)) if sequence.size else 0])
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite acoustic features")
    return result

def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class FeatureEncoder:
    def __init__(self, encoder_dir: Path | None = None):
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.device = choose_device()
        LOGGER.info("Audio encoder device: %s", self.device)
        if encoder_dir is None and (MODEL_DIR / "encoder").is_dir():
            encoder_dir = MODEL_DIR / "encoder"
        if encoder_dir is not None:
            self.processor = ClapFeatureExtractor.from_pretrained(encoder_dir, local_files_only=True)
            self.model = ClapAudioModelWithProjection.from_pretrained(
                encoder_dir, local_files_only=True, use_safetensors=True)
        else:
            options = dict(revision=MODEL_REVISION)
            LOGGER.info("Loading pretrained %s (first use downloads about 615 MB)", MODEL_ID)
            config = ClapConfig.from_pretrained(MODEL_ID, **options)
            self.processor = ClapFeatureExtractor.from_pretrained(MODEL_ID, **options)
            checkpoint = hf_hub_download(MODEL_ID, "pytorch_model.bin", **options)
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model = ClapAudioModelWithProjection(config.audio_config)
            audio_weights = {key: value for key, value in weights.items()
                             if key.startswith(("audio_model.", "audio_projection."))}
            self.model.load_state_dict(audio_weights, strict=True)
            del weights, audio_weights
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, waveforms: list[np.ndarray]) -> np.ndarray:
        inputs = self.processor(waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                                padding="repeatpad", truncation="rand_trunc")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        features = self.model(**inputs).audio_embeds
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        embeddings = features.cpu().numpy()
        acoustic = np.stack([acoustic_features(w) for w in waveforms])
        return np.concatenate([embeddings, acoustic], axis=1).astype(np.float32)

    def save(self, path: Path) -> None:
        self.model.save_pretrained(path, safe_serialization=True)
        self.processor.save_pretrained(path)

def score_features(features: np.ndarray, classifier: dict) -> np.ndarray:
    if classifier["format_version"] != 1 or classifier["feature_version"] != FEATURE_VERSION:
        raise ValueError("Unsupported model format or feature version; use its matching script.")
    coefficients = np.asarray(classifier["coefficients"], dtype=np.float64)
    if features.shape[1] != len(coefficients):
        raise ValueError("Feature size does not match this trained classifier.")
    standardized = (features - np.asarray(classifier["scaler_mean"])) / np.asarray(classifier["scaler_scale"])
    return expit(standardized @ coefficients + classifier["intercept"])