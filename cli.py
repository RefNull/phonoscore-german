#!/usr/bin/env python3
"""PhonoScore command-line client: record or load German speech, then print a pronunciation report.

Talks to the services started by serve.py, or to llama-swap with --router.
"""

import argparse
import io
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# Outside a virtualenv, re-run with the project's .venv if there is one
if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
    repo_dir = Path(__file__).resolve().parent
    venv_python = (
        repo_dir / ".venv" / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else repo_dir / ".venv" / "bin" / "python"
    )
    if venv_python.is_file() and os.access(venv_python, os.X_OK):
        if os.environ.get("_PHONOSCORE_CLI_VENV_EXEC") != "1":
            os.environ["_PHONOSCORE_CLI_VENV_EXEC"] = "1"
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)

import numpy as np
import requests

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

import soundfile as sf


def _post_multipart(
    url: str,
    data: dict,
    files: dict,
    service_label: str = "ASR",
    script_name: str = "asr_server.py",
    default_port: int = 8001,
    direct: bool = False,
    timeout: float = 60.0,
) -> dict:
    """POST a multipart form and return the JSON body; raise with a hint on how to fix connection errors."""
    try:
        response = requests.post(url, files=files, data=data, timeout=timeout)
    except requests.exceptions.ConnectionError as err:
        parsed = urlparse(url)
        port = parsed.port or default_port
        base_url = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else url
        target_name = f"{service_label} server at {base_url}" if direct else f"llama-swap router at {base_url}"
        guidance = (
            f"Start the services with 'python serve.py' (or 'python {script_name} --port {port}')."
            if direct
            else "Start llama-swap with 'llama-swap --config llama-swap.yaml', or drop --router."
        )
        raise ConnectionRefusedError(f"Could not connect to the {target_name}.\n{guidance}") from err
    except requests.exceptions.RequestException as err:
        raise RuntimeError(f"{service_label} request failed: {err}") from err

    if response.status_code != 200:
        raise RuntimeError(
            f"{service_label} request returned status {response.status_code}: {response.text}"
        )

    return response.json()


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score German pronunciation from the microphone or an audio file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Ich habe morgen einen Termin beim Arzt.",
        help="German sentence the speaker should say.",
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Score a .wav or .ogg file instead of recording.",
    )
    parser.add_argument(
        "--asr-url",
        type=str,
        default="http://localhost:8001",
        help="ASR service URL.",
    )
    parser.add_argument(
        "--pronounce-url",
        type=str,
        default="http://localhost:8002",
        help="Pronunciation service URL.",
    )
    parser.add_argument(
        "--router",
        action="store_true",
        default=False,
        help="Send both requests through llama-swap at --router-url.",
    )
    parser.add_argument(
        "--router-url",
        type=str,
        default="http://localhost:8080",
        help="llama-swap URL.",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="Microphone sample rate in Hz.",
    )
    return parser.parse_args(args)


def array_to_wav_bytes(data: np.ndarray, samplerate: int = 16000) -> bytes:
    """Encode samples as 16-bit PCM WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, data, samplerate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def record_audio(
    samplerate: int = 16000,
    input_fn=input,
) -> bytes:
    """Record from the microphone until Enter is pressed; return WAV bytes."""
    if sd is None:
        raise RuntimeError(
            "No microphone support: 'sounddevice' or the PortAudio library is missing.\n"
            "  Use --audio-file <file.wav>, or record in the web UI (http://<server>:8002).\n"
            "  To enable recording here: 'sudo apt install libportaudio2' or 'brew install portaudio'."
        )

    input_fn("Press Enter to start recording...")
    chunks: List[np.ndarray] = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    try:
        stream = sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        with stream:
            input_fn("Recording... Press Enter to stop.\n")
    except Exception as exc:
        raise RuntimeError(
            f"Microphone recording failed: {exc}.\n"
            "  Use --audio-file <file.wav>, or record in the web UI (http://<server>:8002)."
        ) from exc

    if not chunks:
        audio_data = np.zeros((0, 1), dtype=np.int16)
    else:
        audio_data = np.concatenate(chunks, axis=0)

    return array_to_wav_bytes(audio_data, samplerate=samplerate)


def load_audio_file(file_path: str | Path) -> bytes:
    """Read an audio file, rejecting missing or empty files."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty (0 bytes): {file_path}")
    return path.read_bytes()


def query_asr(
    wav_bytes: bytes,
    target_url: str,
    direct: bool = False,
    language: str = "de",
    filename: str = "audio.wav",
    timeout: float = 60.0,
) -> str:
    """Transcribe the audio; return the text."""
    url = f"{target_url.rstrip('/')}/v1/audio/transcriptions"
    mime_type = "audio/ogg" if filename.lower().endswith(".ogg") else "audio/wav"
    files = {"file": (filename, wav_bytes, mime_type)}

    data: Dict[str, Any] = {"language": language}
    if not direct:
        data["model"] = "qwen3-asr"

    result = _post_multipart(
        url,
        data=data,
        files=files,
        service_label="ASR",
        script_name="asr_server.py",
        default_port=8001,
        direct=direct,
        timeout=timeout,
    )
    return result.get("text", "")


def query_pronounce(
    wav_bytes: bytes,
    target_url: str,
    direct: bool = False,
    expected_text: str = "",
    lang: str = "de",
    filename: str = "audio.wav",
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Score the audio against expected_text; return the /assess JSON."""
    url = f"{target_url.rstrip('/')}/assess" if direct else f"{target_url.rstrip('/')}/upstream/openpronounce/assess"
    mime_type = "audio/ogg" if filename.lower().endswith(".ogg") else "audio/wav"
    files = {"file": (filename, wav_bytes, mime_type)}
    data = {"expected_text": expected_text, "lang": lang}

    return _post_multipart(
        url,
        data=data,
        files=files,
        service_label="Pronunciation",
        script_name="pronounce_server.py",
        default_port=8002,
        direct=direct,
        timeout=timeout,
    )


def format_score_bar(score: float, width: int = 20) -> str:
    """Render e.g. '[================>   ] 82.5 / 100'."""
    clamped = max(0.0, min(100.0, float(score)))
    filled = int(round((clamped / 100.0) * width))
    if filled <= 0:
        bar = " " * width
    elif filled >= width:
        bar = "=" * width
    else:
        bar = "=" * (filled - 1) + ">" + " " * (width - filled)
    return f"[{bar}] {clamped:.1f} / 100"


def format_report(target_text: str, asr_text: str, assessment: Dict[str, Any]) -> str:
    """Build the terminal report: score, PER, the three transcriptions, and flagged words."""
    score = float(assessment.get("score", 0.0))
    w2v2_text = assessment.get("transcription", "")
    per = float(assessment.get("phoneme_error_rate", 0.0))
    errors = assessment.get("errors", [])

    score_bar = format_score_bar(score, width=20)
    per_pct = f"{per * 100:.1f}%"

    lines: List[str] = [
        "=" * 74,
        f"Score:                {score_bar}",
        f"Phoneme error rate:   {per_pct}",
        "-" * 74,
        f"Expected:             {target_text}",
        f"Heard (Qwen3-ASR):    {asr_text}",
        f"Heard (Wav2Vec2):     {w2v2_text}",
        "-" * 74,
    ]

    if not errors:
        lines.append("No mispronounced words.")
    else:
        lines.append(f"Mispronounced words ({len(errors)}):")
        lines.append(f"  {'Word':<16} {'Expected IPA':<18} {'Heard IPA':<18} {'Confidence':<10}")
        lines.append(f"  {'-'*16} {'-'*18} {'-'*18} {'-'*10}")
        for err in errors:
            word = str(err.get("word", ""))
            exp_ipa = str(err.get("expected_ipa") or err.get("expected", ""))
            act_ipa = str(err.get("actual_ipa") or err.get("actual", ""))
            conf = err.get("confidence", 0.0)
            if isinstance(conf, (int, float)):
                conf_str = f"{conf * 100:.1f}%" if conf <= 1.0 else f"{conf:.1f}%"
            else:
                conf_str = str(conf)
            lines.append(f"  {word:<16} {exp_ipa:<18} {act_ipa:<18} {conf_str:<10}")

    lines.append("=" * 74)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    direct = not args.router

    print(f"PhonoScore German\nSay: \"{args.text}\"\n")

    filename = "audio.wav"
    if args.audio_file:
        print(f"Reading {args.audio_file}")
        try:
            wav_bytes = load_audio_file(args.audio_file)
            filename = Path(args.audio_file).name
        except Exception as exc:
            print(f"\n[Error] Failed to load audio file: {exc}", file=sys.stderr)
            return 1
    else:
        try:
            wav_bytes = record_audio(samplerate=args.samplerate)
        except Exception as exc:
            print(f"\n[Error] Audio recording failed: {exc}", file=sys.stderr)
            return 1

    if not wav_bytes:
        print("\n[Error] Recorded audio buffer is empty.", file=sys.stderr)
        return 1

    asr_endpoint = args.asr_url if direct else args.router_url
    pronounce_endpoint = args.pronounce_url if direct else args.router_url

    print("Transcribing...")
    try:
        asr_text = query_asr(
            wav_bytes=wav_bytes,
            target_url=asr_endpoint,
            direct=direct,
            language="de",
            filename=filename,
        )
    except ConnectionRefusedError as exc:
        print(f"\n[Connection Error] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[Error] ASR step failed: {exc}", file=sys.stderr)
        return 1

    print("Scoring pronunciation...")
    try:
        assessment = query_pronounce(
            wav_bytes=wav_bytes,
            target_url=pronounce_endpoint,
            direct=direct,
            expected_text=args.text,
            lang="de",
            filename=filename,
        )
    except ConnectionRefusedError as exc:
        print(f"\n[Connection Error] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[Error] Pronunciation scoring failed: {exc}", file=sys.stderr)
        return 1

    print()
    report = format_report(
        target_text=args.text,
        asr_text=asr_text,
        assessment=assessment,
    )
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
