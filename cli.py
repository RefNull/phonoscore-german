#!/usr/bin/env python3
"""PhonoScore command-line client: record or load German speech, then print a pronunciation report.

Talks to the services started by serve.py, or to llama-swap with --router.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np
import requests
import soundfile as sf

try:
    import sounddevice as sd
except (ImportError, OSError):  # OSError: PortAudio library missing
    sd = None


def _post_multipart(url: str, data: dict, files: dict, service_label: str, direct: bool, timeout: float) -> dict:
    """POST a multipart form and return the JSON body; raise with a hint on how to fix connection errors."""
    try:
        response = requests.post(url, files=files, data=data, timeout=timeout)
    except requests.exceptions.ConnectionError as err:
        if direct:
            message = f"Could not connect to the {service_label} server at {url}.\nStart the services with 'python serve.py'."
        else:
            message = (
                f"Could not connect to the llama-swap router at {url}.\n"
                "Start llama-swap with 'llama-swap --config llama-swap.yaml', or drop --router."
            )
        raise ConnectionRefusedError(message) from err
    except requests.exceptions.RequestException as err:
        raise RuntimeError(f"{service_label} request failed: {err}") from err

    if response.status_code != 200:
        raise RuntimeError(f"{service_label} request returned status {response.status_code}: {response.text}")
    return response.json()


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score German pronunciation from the microphone or an audio file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--text", default="Ich habe morgen einen Termin beim Arzt.", help="German sentence the speaker should say.")
    parser.add_argument("--audio-file", default=None, help="Score a .wav or .ogg file instead of recording.")
    parser.add_argument("--asr-url", default="http://localhost:8001", help="ASR service URL.")
    parser.add_argument("--pronounce-url", default="http://localhost:8002", help="Pronunciation service URL.")
    parser.add_argument("--router", action="store_true", help="Send both requests through llama-swap at --router-url.")
    parser.add_argument("--router-url", default="http://localhost:8080", help="llama-swap URL.")
    parser.add_argument("--samplerate", type=int, default=16000, help="Microphone sample rate in Hz.")
    return parser.parse_args(args)


def array_to_wav_bytes(data: np.ndarray, samplerate: int = 16000) -> bytes:
    """Encode samples as 16-bit PCM WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, data, samplerate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def record_audio(samplerate: int = 16000, input_fn=input) -> bytes:
    """Record from the microphone until Enter is pressed; return WAV bytes."""
    if sd is None:
        raise RuntimeError(
            "No microphone support: 'sounddevice' or the PortAudio library is missing.\n"
            "  Use --audio-file <file.wav>, or record in the web UI (http://<server>:8002).\n"
            "  To enable recording here: 'sudo apt install libportaudio2' or 'brew install portaudio'."
        )

    input_fn("Press Enter to start recording...")
    chunks: list[np.ndarray] = []
    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype="int16", callback=lambda data, *_: chunks.append(data.copy())):
            input_fn("Recording... Press Enter to stop.\n")
    except Exception as exc:
        raise RuntimeError(
            f"Microphone recording failed: {exc}.\n"
            "  Use --audio-file <file.wav>, or record in the web UI (http://<server>:8002)."
        ) from exc

    audio = np.concatenate(chunks) if chunks else np.zeros((0, 1), dtype=np.int16)
    return array_to_wav_bytes(audio, samplerate=samplerate)


def load_audio_file(file_path: str | Path) -> bytes:
    """Read an audio file, rejecting missing or empty files."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty (0 bytes): {file_path}")
    return path.read_bytes()


def _audio_upload(wav_bytes: bytes, filename: str) -> dict:
    mime_type = "audio/ogg" if filename.lower().endswith(".ogg") else "audio/wav"
    return {"file": (filename, wav_bytes, mime_type)}


def query_asr(
    wav_bytes: bytes,
    target_url: str,
    direct: bool = False,
    language: str = "de",
    filename: str = "audio.wav",
    timeout: float = 60.0,
) -> str:
    """Transcribe the audio; return the text."""
    data = {"language": language}
    if not direct:
        data["model"] = "qwen3-asr"  # llama-swap routes on this
    url = f"{target_url.rstrip('/')}/v1/audio/transcriptions"
    return _post_multipart(url, data, _audio_upload(wav_bytes, filename), "ASR", direct, timeout).get("text", "")


def query_pronounce(
    wav_bytes: bytes,
    target_url: str,
    direct: bool = False,
    expected_text: str = "",
    lang: str = "de",
    filename: str = "audio.wav",
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Score the audio against expected_text; return the /assess JSON."""
    path = "/assess" if direct else "/upstream/openpronounce/assess"
    url = f"{target_url.rstrip('/')}{path}"
    data = {"expected_text": expected_text, "lang": lang}
    return _post_multipart(url, data, _audio_upload(wav_bytes, filename), "Pronunciation", direct, timeout)


def format_score_bar(score: float, width: int = 20) -> str:
    """Render e.g. '[================>   ] 82.5 / 100'."""
    clamped = max(0.0, min(100.0, float(score)))
    filled = round(clamped / 100.0 * width)
    if 0 < filled < width:
        bar = "=" * (filled - 1) + ">" + " " * (width - filled)
    else:
        bar = "=" * filled + " " * (width - filled)
    return f"[{bar}] {clamped:.1f} / 100"


def format_report(target_text: str, asr_text: str, assessment: dict[str, Any]) -> str:
    """Build the terminal report: score, PER, the three transcriptions, and flagged words."""
    per = float(assessment.get("phoneme_error_rate", 0.0))
    errors = assessment.get("errors", [])
    lines = [
        "=" * 74,
        f"Score:                {format_score_bar(float(assessment.get('score', 0.0)))}",
        f"Phoneme error rate:   {per * 100:.1f}%",
        "-" * 74,
        f"Expected:             {target_text}",
        f"Heard (Qwen3-ASR):    {asr_text}",
        f"Heard (Wav2Vec2):     {assessment.get('transcription', '')}",
        "-" * 74,
    ]
    if not errors:
        lines.append("No mispronounced words.")
    else:
        lines.append(f"Mispronounced words ({len(errors)}):")
        lines.append(f"  {'Word':<16} {'Expected IPA':<18} {'Heard IPA':<18} {'Confidence':<10}")
        lines.append(f"  {'-' * 16} {'-' * 18} {'-' * 18} {'-' * 10}")
        for err in errors:
            confidence = f"{err.get('confidence', 0.0) * 100:.1f}%"
            lines.append(f"  {err.get('word', ''):<16} {err.get('expected_ipa', ''):<18} {err.get('actual_ipa', ''):<18} {confidence:<10}")
    lines.append("=" * 74)
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    direct = not args.router
    print(f"PhonoScore German\nSay: \"{args.text}\"\n")

    filename = "audio.wav"
    try:
        if args.audio_file:
            print(f"Reading {args.audio_file}")
            wav_bytes = load_audio_file(args.audio_file)
            filename = Path(args.audio_file).name
        else:
            wav_bytes = record_audio(samplerate=args.samplerate)
    except Exception as exc:
        what = "Failed to load audio file" if args.audio_file else "Audio recording failed"
        print(f"\n[Error] {what}: {exc}", file=sys.stderr)
        return 1

    asr_url = args.asr_url if direct else args.router_url
    pronounce_url = args.pronounce_url if direct else args.router_url
    try:
        print("Transcribing...")
        asr_text = query_asr(wav_bytes=wav_bytes, target_url=asr_url, direct=direct, language="de", filename=filename)
        print("Scoring pronunciation...")
        assessment = query_pronounce(
            wav_bytes=wav_bytes, target_url=pronounce_url, direct=direct, expected_text=args.text, lang="de", filename=filename
        )
    except ConnectionRefusedError as exc:
        print(f"\n[Connection Error] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[Error] {exc}", file=sys.stderr)
        return 1

    print()
    print(format_report(target_text=args.text, asr_text=asr_text, assessment=assessment))
    return 0


if __name__ == "__main__":
    sys.exit(main())
