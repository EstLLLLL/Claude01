#!/usr/bin/env python3
"""On-device speaker voiceprint enrollment and meeting-label matching.

Audio is decoded with the local ffmpeg binary. Speaker embeddings are computed
on-device with sherpa-onnx. Derived profiles are stored in the user's private
iCloud Drive so the same library is available on their Macs. Raw audio, the
model, and the Python runtime stay local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sherpa_onnx


SAMPLE_RATE = 16000
DEFAULT_LOCAL_ROOT = Path.home() / ".codex" / "private" / "voiceprints"
DEFAULT_ICLOUD_PROFILES = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "com~apple~CloudDocs"
    / "Codex"
    / "Private"
    / "Voiceprints"
    / "profiles"
)
DEFAULT_MODEL_NAME = "3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx"


class VoiceprintError(RuntimeError):
    pass


def local_root() -> Path:
    override = os.environ.get("MINUTES_VOICEPRINT_LOCAL_DIR") or os.environ.get("MINUTES_VOICEPRINT_DIR")
    return Path(override).expanduser().resolve() if override else DEFAULT_LOCAL_ROOT.resolve()


def model_path() -> Path:
    override = os.environ.get("MINUTES_VOICEPRINT_MODEL")
    return Path(override).expanduser().resolve() if override else local_root() / "models" / DEFAULT_MODEL_NAME


def profiles_dir() -> Path:
    override = os.environ.get("MINUTES_VOICEPRINT_PROFILE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    legacy_root = os.environ.get("MINUTES_VOICEPRINT_DIR")
    if legacy_root:
        return Path(legacy_root).expanduser().resolve() / "profiles"
    return DEFAULT_ICLOUD_PROFILES.resolve()


def legacy_profiles_dir() -> Path:
    return DEFAULT_LOCAL_ROOT.resolve() / "profiles"


def safe_profile_name(name: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", name.strip(), flags=re.UNICODE).strip("._")
    if not value:
        raise VoiceprintError("Speaker name is empty after sanitization")
    return value


def decode_audio(path: Path, start_sec: float | None = None, duration_sec: float | None = None) -> np.ndarray:
    if not path.is_file():
        raise VoiceprintError(f"Audio file not found: {path}")
    command = ["ffmpeg", "-nostdin", "-v", "error"]
    if start_sec is not None:
        command.extend(["-ss", f"{max(0.0, start_sec):.3f}"])
    command.extend(["-i", str(path)])
    if duration_sec is not None:
        command.extend(["-t", f"{max(0.0, duration_sec):.3f}"])
    command.extend(["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"])
    result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")
        raise VoiceprintError(f"ffmpeg could not decode {path.name}: {detail[:300]}")
    samples = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=True)
    if samples.size == 0:
        raise VoiceprintError(f"Decoded audio is empty: {path.name}")
    return trim_edge_silence(samples)


def trim_edge_silence(samples: np.ndarray) -> np.ndarray:
    """Trim only quiet leading/trailing frames; preserve internal pauses."""
    frame = int(0.03 * SAMPLE_RATE)
    if samples.size < frame * 4:
        return samples
    usable = samples[: samples.size - (samples.size % frame)]
    frames = usable.reshape(-1, frame)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    threshold = min(-32.0, float(np.percentile(db, 20)) + 8.0)
    active = np.flatnonzero(db > threshold)
    if active.size == 0:
        return samples
    start = max(0, int(active[0]) * frame - int(0.15 * SAMPLE_RATE))
    end = min(samples.size, (int(active[-1]) + 1) * frame + int(0.15 * SAMPLE_RATE))
    return samples[start:end]


def audio_stats(samples: np.ndarray) -> dict[str, float]:
    duration = samples.size / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)) + 1e-12))
    peak = float(np.max(np.abs(samples)))
    return {
        "duration_sec": round(duration, 3),
        "rms_dbfs": round(20.0 * math.log10(max(rms, 1e-12)), 2),
        "peak_dbfs": round(20.0 * math.log10(max(peak, 1e-12)), 2),
    }


def validate_sample(samples: np.ndarray, label: str) -> dict[str, float]:
    stats = audio_stats(samples)
    if stats["duration_sec"] < 2.0:
        raise VoiceprintError(f"{label}: less than 2 seconds of usable audio")
    if stats["rms_dbfs"] < -42.0:
        raise VoiceprintError(f"{label}: audio is too quiet for reliable enrollment")
    return stats


def load_extractor() -> sherpa_onnx.SpeakerEmbeddingExtractor:
    model = model_path()
    if not model.is_file():
        raise VoiceprintError(f"Speaker model not found: {model}")
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model), num_threads=max(1, min(4, os.cpu_count() or 1)), debug=False, provider="cpu"
    )
    if not config.validate():
        raise VoiceprintError(f"Invalid speaker embedding model config: {model}")
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


def embedding(samples: np.ndarray, extractor: sherpa_onnx.SpeakerEmbeddingExtractor) -> np.ndarray:
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=samples)
    stream.input_finished()
    if not extractor.is_ready(stream):
        raise VoiceprintError("Speaker model did not accept the audio sample")
    value = np.asarray(extractor.compute(stream), dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0:
        raise VoiceprintError("Speaker model returned an invalid embedding")
    return value / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def profile_fingerprint(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable.pop("storage", None)
    encoded = json.dumps(comparable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temp, 0o600)
    temp.replace(path)
    os.chmod(path, 0o600)


def ensure_library_manifest() -> Path:
    path = profiles_dir().parent / "library.json"
    if not path.exists():
        write_private_json(
            path,
            {
                "schema_version": 1,
                "library": "minutes shared voiceprints",
                "profile_schema_version": 1,
                "model": DEFAULT_MODEL_NAME,
                "sample_rate": SAMPLE_RATE,
                "contains_raw_audio": False,
                "storage": "user-authorized private iCloud Drive",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    return path


def enroll(name: str, audio_files: list[Path], min_consistency: float) -> dict[str, Any]:
    if len(audio_files) < 2:
        raise VoiceprintError("Use at least two independent clips for enrollment")
    extractor = load_extractor()
    vectors: list[np.ndarray] = []
    samples_meta: list[dict[str, Any]] = []
    for path in audio_files:
        resolved = path.expanduser().resolve()
        samples = decode_audio(resolved)
        stats = validate_sample(samples, resolved.name)
        vectors.append(embedding(samples, extractor))
        samples_meta.append({
            "file_name": resolved.name,
            "sha256": sha256(resolved),
            **stats,
        })
    similarities = [
        cosine(vectors[i], vectors[j])
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    minimum = min(similarities)
    if minimum < min_consistency:
        raise VoiceprintError(
            f"Enrollment clips are not consistent enough: minimum cosine={minimum:.4f}, "
            f"required={min_consistency:.4f}"
        )
    prototype = np.mean(np.stack(vectors), axis=0)
    prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
    profile = {
        "schema_version": 1,
        "name": name.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_path().name,
        "sample_rate": SAMPLE_RATE,
        "embedding_dimension": int(prototype.size),
        "prototype": [round(float(x), 9) for x in prototype],
        "enrollment_embeddings": [[round(float(x), 9) for x in value] for value in vectors],
        "samples": samples_meta,
        "consistency": {
            "pairwise_cosine": [round(value, 6) for value in similarities],
            "minimum": round(minimum, 6),
            "mean": round(float(np.mean(similarities)), 6),
        },
        "raw_audio_stored": False,
        "storage": "icloud_shared_profile",
    }
    ensure_library_manifest()
    output = profiles_dir() / f"{safe_profile_name(name)}.json"
    write_private_json(output, profile)
    return {
        "status": "enrolled",
        "name": name.strip(),
        "profile_path": str(output),
        "sample_count": len(audio_files),
        "consistency": profile["consistency"],
        "samples": samples_meta,
        "raw_audio_stored": False,
        "profile_dir": str(profiles_dir()),
    }


def load_profiles(
    expected_model: str | None = None,
    expected_dim: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    loaded: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for path in sorted(profiles_dir().glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if int(data.get("schema_version", 0)) != 1:
                raise VoiceprintError("unsupported profile schema")
            if expected_model and str(data.get("model", "")) != expected_model:
                raise VoiceprintError(f"model mismatch: {data.get('model', 'missing')}")
            vector = np.asarray(data["prototype"], dtype=np.float32)
            if expected_dim and vector.size != expected_dim:
                raise VoiceprintError(f"embedding dimension {vector.size}, expected {expected_dim}")
            name = str(data["name"]).strip()
            if not name:
                raise VoiceprintError("empty speaker name")
            if name in seen_names:
                raise VoiceprintError(f"duplicate speaker name: {name}")
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            loaded.append({"name": name, "vector": vector, "path": str(path), "model": str(data.get("model", ""))})
            seen_names.add(name)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError, VoiceprintError) as exc:
            issues.append({"file": path.name, "error": str(exc)})
    return loaded, issues


def inventory() -> dict[str, Any]:
    profiles, issues = load_profiles(expected_model=DEFAULT_MODEL_NAME)
    return {
        "profile_dir": str(profiles_dir()),
        "storage": "icloud_shared" if profiles_dir() == DEFAULT_ICLOUD_PROFILES.resolve() else "override",
        "profiles": [profile["name"] for profile in profiles],
        "profile_files": [Path(profile["path"]).name for profile in profiles],
        "issues": issues,
        "local_model": str(model_path()),
        "local_model_available": model_path().is_file(),
        "raw_audio_stored": False,
    }


def migrate_local(source_dir: Path | None = None) -> dict[str, Any]:
    source = (source_dir or legacy_profiles_dir()).expanduser().resolve()
    destination = profiles_dir()
    destination.mkdir(parents=True, exist_ok=True)
    ensure_library_manifest()
    migrated: list[str] = []
    skipped: list[str] = []
    conflicts: list[dict[str, str]] = []
    incompatible: list[dict[str, str]] = []
    if source == destination:
        return {
            "status": "already_shared",
            "source": str(source),
            "destination": str(destination),
            "migrated": [],
            "skipped": [],
            "conflicts": [],
            "incompatible": [],
        }
    for path in sorted(source.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if int(data.get("schema_version", 0)) != 1:
                raise VoiceprintError("unsupported profile schema")
            if str(data.get("model", "")) != DEFAULT_MODEL_NAME:
                raise VoiceprintError(f"model mismatch: {data.get('model', 'missing')}")
            name = str(data.get("name", "")).strip()
            if not name:
                raise VoiceprintError("empty speaker name")
            target = destination / f"{safe_profile_name(name)}.json"
            if target.exists():
                existing = json.loads(target.read_text(encoding="utf-8"))
                if profile_fingerprint(data) == profile_fingerprint(existing):
                    skipped.append(name)
                else:
                    conflicts.append({"name": name, "source": str(path), "existing": str(target)})
                continue
            data["storage"] = "icloud_shared_profile"
            write_private_json(target, data)
            migrated.append(name)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError, VoiceprintError) as exc:
            incompatible.append({"file": path.name, "error": str(exc)})
    status = "migrated" if migrated else ("conflict" if conflicts else "nothing_to_migrate")
    return {
        "status": status,
        "source": str(source),
        "destination": str(destination),
        "migrated": migrated,
        "skipped": skipped,
        "conflicts": conflicts,
        "incompatible": incompatible,
    }


def _segment_audio(path: Path, rows: list[dict[str, Any]], max_sec: float = 30.0) -> np.ndarray:
    candidates: list[tuple[float, float]] = []
    for row in rows:
        start = max(0.0, float(row.get("start_time", 0)) / 1000.0)
        end = max(start, float(row.get("end_time", 0)) / 1000.0)
        if end - start >= 0.5:
            candidates.append((start, end))
    candidates.sort(key=lambda item: item[1] - item[0], reverse=True)
    chosen: list[np.ndarray] = []
    used_sec = 0.0
    for start, end in candidates[:16]:
        if used_sec >= max_sec:
            break
        piece_sec = min(end - start, max_sec - used_sec, 12.0)
        piece = decode_audio(path, start_sec=start, duration_sec=piece_sec)
        if piece.size:
            chosen.append(piece)
            used_sec += piece.size / SAMPLE_RATE
    if not chosen:
        return np.asarray([], dtype=np.float32)
    return trim_edge_silence(np.concatenate(chosen))


def _chunk_key(label: str) -> str:
    match = re.match(r"^(\d+)-", label)
    return match.group(1) if match else "all"


def match_utterances(audio_file: Path, utterances: list[dict[str, Any]], threshold: float, margin: float) -> dict[str, Any]:
    extractor = load_extractor()
    profiles, profile_issues = load_profiles(expected_model=model_path().name, expected_dim=extractor.dim)
    if not profiles:
        return {
            "status": "no_profiles",
            "aliases": {},
            "threshold": threshold,
            "labels": [],
            "profile_dir": str(profiles_dir()),
            "profile_issues": profile_issues,
        }
    resolved_audio = audio_file.expanduser().resolve()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in utterances:
        label = str(row.get("speaker", "")).strip()
        if label and str(row.get("text", "")).strip():
            grouped.setdefault(label, []).append(row)

    label_rows: list[dict[str, Any]] = []
    vectors: dict[str, np.ndarray] = {}
    for label, rows in grouped.items():
        speech = _segment_audio(resolved_audio, rows)
        usable = speech.size / SAMPLE_RATE
        if usable < 2.0:
            label_rows.append({"label": label, "usable_sec": round(usable, 3), "status": "too_short", "scores": {}})
            continue
        vector = embedding(speech, extractor)
        vectors[label] = vector
        scores = {profile["name"]: round(cosine(vector, profile["vector"]), 6) for profile in profiles}
        label_rows.append({"label": label, "usable_sec": round(usable, 3), "status": "scored", "scores": scores})

    aliases: dict[str, str] = {}
    rejected: list[dict[str, Any]] = []
    chunks = sorted({_chunk_key(label) for label in vectors})
    for chunk in chunks:
        chunk_labels = [label for label in vectors if _chunk_key(label) == chunk]
        candidates: list[tuple[float, str, str]] = []
        for profile in profiles:
            ranked = sorted(
                ((cosine(vectors[label], profile["vector"]), label) for label in chunk_labels),
                reverse=True,
            )
            if not ranked:
                continue
            best_score, best_label = ranked[0]
            second_score = ranked[1][0] if len(ranked) > 1 else -1.0
            other_profile_scores = sorted(
                (cosine(vectors[best_label], p["vector"]) for p in profiles if p["name"] != profile["name"]),
                reverse=True,
            )
            profile_margin = best_score - (other_profile_scores[0] if other_profile_scores else -1.0)
            label_margin = best_score - second_score
            if best_score < threshold:
                rejected.append({"chunk": chunk, "name": profile["name"], "reason": "below_threshold", "score": round(best_score, 6)})
                continue
            if len(ranked) > 1 and second_score >= threshold and label_margin < margin:
                rejected.append({"chunk": chunk, "name": profile["name"], "reason": "ambiguous_label", "score": round(best_score, 6), "margin": round(label_margin, 6)})
                continue
            if other_profile_scores and profile_margin < margin:
                rejected.append({"chunk": chunk, "name": profile["name"], "reason": "ambiguous_profile", "score": round(best_score, 6), "margin": round(profile_margin, 6)})
                continue
            candidates.append((best_score, best_label, profile["name"]))
        used_labels: set[str] = set()
        used_profiles: set[str] = set()
        for score, label, name in sorted(candidates, reverse=True):
            if label in used_labels or name in used_profiles:
                rejected.append({"chunk": chunk, "name": name, "reason": "assignment_conflict", "score": round(score, 6)})
                continue
            aliases[label] = name
            used_labels.add(label)
            used_profiles.add(name)
    return {
        "status": "matched" if aliases else "no_confident_match",
        "aliases": aliases,
        "threshold": threshold,
        "margin": margin,
        "profiles": [profile["name"] for profile in profiles],
        "profile_dir": str(profiles_dir()),
        "profile_issues": profile_issues,
        "labels": label_rows,
        "rejected": rejected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="On-device speaker voiceprint utility with iCloud-shared profiles")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("enroll", help="Enroll a speaker from two or more clips")
    add.add_argument("--name", required=True)
    add.add_argument("--audio", action="append", required=True)
    add.add_argument("--min-consistency", type=float, default=0.55)

    match = sub.add_parser("match-utterances", help="Match diarization labels to enrolled speakers")
    match.add_argument("--audio", required=True)
    match.add_argument("--utterances-json", default="-", help="JSON file or - for stdin")
    match.add_argument("--threshold", type=float, default=0.55)
    match.add_argument("--margin", type=float, default=0.04)

    migrate = sub.add_parser("migrate-local", help="Copy compatible legacy local profiles into the iCloud library")
    migrate.add_argument("--source-dir", default=None)

    sub.add_parser("list", help="List enrolled speaker names and the active shared library path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "enroll":
        result = enroll(args.name, [Path(value) for value in args.audio], args.min_consistency)
    elif args.command == "match-utterances":
        if args.utterances_json == "-":
            payload = json.load(sys.stdin)
        else:
            payload = json.loads(Path(args.utterances_json).read_text(encoding="utf-8"))
        utterances = payload.get("utterances", payload) if isinstance(payload, dict) else payload
        if not isinstance(utterances, list):
            raise VoiceprintError("Utterances input must be a JSON list or {\"utterances\": [...]} object")
        result = match_utterances(Path(args.audio), utterances, args.threshold, args.margin)
    elif args.command == "migrate-local":
        result = migrate_local(Path(args.source_dir) if args.source_dir else None)
    else:
        result = inventory()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VoiceprintError as exc:
        print(f"voiceprint error: {exc}", file=sys.stderr)
        raise SystemExit(2)
