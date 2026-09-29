"""
Tests for pronounce_server.py.
"""

import io
import os
from pathlib import Path
import sys
import wave
from unittest.mock import MagicMock

# Mock openpronounce module in sys.modules if not installed in environment
if "openpronounce" not in sys.modules:
    mock_op = MagicMock()
    sys.modules["openpronounce"] = mock_op
import openpronounce

# Ensure repository root is in sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pytest
from fastapi.testclient import TestClient
import torch

from pronounce_server import (
    app,
    parse_args,
    resolve_device,
)


def make_synthetic_wav(duration_seconds: float = 0.1, sample_rate: int = 16000) -> bytes:
    """Generate synthetic 16kHz mono WAV bytes for testing without external audio files."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        num_frames = int(duration_seconds * sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return buffer.getvalue()


@pytest.fixture
def synthetic_wav_bytes():
    return make_synthetic_wav()


@pytest.fixture
def sample_german_result():
    """Trimmed openpronounce 0.3.x result with the phone recognizer enabled (its default)."""
    return {
        "score": 88.5,
        "transcribe": "Ich gehe heute in die Schule",
        "language": "de",
        "differences": {
            "phoneme_error_rate": 0.115,
            "word_error_rate": 0.0,
            "transcribe": "Ich gehe heute in die Schule",
            "errors": [
                {
                    "position": 5,
                    "word": "Schule",
                    "expected": ["ʃ", "uː", "l", "ə"],
                    "actual": ["s", "uː", "l", "ə"],
                    "distance": 1,
                    "phones": [{"expected": "ʃ", "heard": "s", "confidence": 0.92}],
                    "weighted_edits": 0.92,
                },
                {
                    "position": 1,
                    "word": "gehe",
                    "expected": ["ɡ", "eː", "ə"],
                    "actual": ["ɡ", "eː", "h", "ə"],
                    "distance": 1,
                    "phones": [
                        {"expected": "", "heard": "h", "confidence": 0.4},
                        {"expected": "ə", "heard": "ə", "confidence": 0.78},
                    ],
                    "weighted_edits": 1.18,
                },
            ],
        },
    }


@pytest.fixture
def mock_openpronounce(sample_german_result, monkeypatch):
    """Mock openpronounce compare and load functions."""
    mock_compare = MagicMock(return_value=sample_german_result)
    mock_load = MagicMock(return_value="mock_sound_tensor")
    monkeypatch.setattr(openpronounce, "compare_audio_with_text", mock_compare)
    monkeypatch.setattr(openpronounce, "load_audio", mock_load)
    monkeypatch.setattr("pronounce_server.openpronounce", openpronounce)
    return mock_compare, mock_load


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc


def test_root_serves_web_ui(client):
    """GET / should return the web UI."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>PhonoScore" in response.text


def test_health_endpoint(client):
    """GET /health should return 200 OK and expected service metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "openpronounce"
    assert "device" in data
    assert isinstance(data["device"], str)
    assert data["device"] in ("cuda", "mps", "cpu")


def test_assess_success_contract(client, mock_openpronounce, synthetic_wav_bytes):
    """
    POST /assess with mock assessor returning sample German differences.
    Verify returned JSON schema matches exact contract:
    - score: float
    - transcription: str
    - phoneme_error_rate: float
    - errors: list of dicts with word, expected_ipa, actual_ipa, confidence
    """
    files = {"file": ("test_recording.wav", synthetic_wav_bytes, "audio/wav")}
    form_data = {
        "expected_text": "Ich gehe heute in die Schule",
        "lang": "de",
    }
    response = client.post("/assess", files=files, data=form_data)

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data["score"], float)
    assert data["score"] == 88.5
    assert isinstance(data["transcription"], str)
    assert data["transcription"] == "Ich gehe heute in die Schule"
    assert isinstance(data["phoneme_error_rate"], float)
    assert data["phoneme_error_rate"] == 0.115

    errors = data["errors"]
    assert isinstance(errors, list)
    assert len(errors) == 2

    assert errors[0] == {
        "word": "Schule",
        "expected_ipa": "ʃuːlə",
        "actual_ipa": "suːlə",
        "confidence": 0.92,
    }
    assert errors[1] == {
        "word": "gehe",
        "expected_ipa": "ɡeːə",
        "actual_ipa": "ɡeːhə",
        "confidence": 0.78,
    }


def test_assess_ogg_format(client, mock_openpronounce):
    """POST /assess with .ogg audio file preserves audio format extension."""
    mock_compare, mock_load = mock_openpronounce
    ogg_bytes = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00synthetic_ogg_bytes"
    files = {"file": ("recording.ogg", ogg_bytes, "audio/ogg")}
    form_data = {"expected_text": "Guten Tag"}

    response = client.post("/assess", files=files, data=form_data)
    assert response.status_code == 200

    assert mock_load.called
    assert mock_load.call_args[0][0].endswith(".ogg")
    assert mock_compare.called
    call_args, call_kwargs = mock_compare.call_args
    assert call_args[1] == "Guten Tag"
    assert call_kwargs.get("lang") == "de"


def test_assess_empty_file(client):
    """Uploading an empty (0-byte) file should return 400 Bad Request."""
    files = {"file": ("empty.wav", b"", "audio/wav")}
    data = {"expected_text": "Hallo Welt"}
    response = client.post("/assess", files=files, data=data)

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_assess_missing_file(client):
    """Sending request without 'file' multipart parameter should return 422 Unprocessable Entity."""
    response = client.post("/assess", data={"expected_text": "Hallo Welt"})
    assert response.status_code == 422


def test_assess_missing_expected_text(client, synthetic_wav_bytes):
    """Missing expected_text parameter should return 422 Unprocessable Entity."""
    files = {"file": ("recording.wav", synthetic_wav_bytes, "audio/wav")}
    response = client.post("/assess", files=files)
    assert response.status_code == 422


def test_assess_empty_expected_text(client, synthetic_wav_bytes):
    """Empty or whitespace-only expected_text should return 422 Unprocessable Entity."""
    files = {"file": ("recording.wav", synthetic_wav_bytes, "audio/wav")}

    # Empty string
    res_empty = client.post("/assess", files=files, data={"expected_text": ""})
    assert res_empty.status_code == 422

    # Whitespace only
    files = {"file": ("recording.wav", synthetic_wav_bytes, "audio/wav")}
    res_ws = client.post("/assess", files=files, data={"expected_text": "   \t \n "})
    assert res_ws.status_code == 422
    assert "expected_text must not be empty" in str(res_ws.json()["detail"])


def test_assess_exception_unlinks_tempfile(client, monkeypatch, synthetic_wav_bytes):
    """Assessor exception should return 500 and guarantee temporary file is unlinked."""
    captured_paths = []

    def failing_load(audio_path):
        captured_paths.append(audio_path)
        assert os.path.exists(audio_path)
        raise RuntimeError("OpenPronounce comparison crash")

    monkeypatch.setattr(openpronounce, "load_audio", failing_load)
    monkeypatch.setattr("pronounce_server.openpronounce", openpronounce)

    files = {"file": ("recording.wav", synthetic_wav_bytes, "audio/wav")}
    response = client.post("/assess", files=files, data={"expected_text": "Guten Tag"})

    assert response.status_code == 500
    assert "Pronunciation assessment failed" in response.json()["detail"]
    assert len(captured_paths) == 1
    assert not os.path.exists(captured_paths[0])


def test_word_transcription_path_normalization(client, monkeypatch, synthetic_wav_bytes):
    """
    With the phone recognizer disabled, openpronounce derives errors from the word
    transcription: expected/actual are IPA strings and there is no per-phone confidence.
    """
    mock_compare = MagicMock(
        return_value={
            "score": 95.0,
            "transcribe": "Guten Morgen",
            "differences": {
                "phoneme_error_rate": 0.05,
                "word_error_rate": 0.0,
                "transcribe": "Guten Morgen",
                "errors": [
                    {
                        "position": 5,
                        "word": "Morgen",
                        "expected": "mɔʁɡn̩",
                        "actual": "mɔʁɡŋ",
                        "actual_word": "Morgen",
                    }
                ],
            },
        }
    )
    monkeypatch.setattr(openpronounce, "compare_audio_with_text", mock_compare)
    monkeypatch.setattr(openpronounce, "load_audio", lambda p: "mock_sound")
    monkeypatch.setattr("pronounce_server.openpronounce", openpronounce)

    files = {"file": ("audio.wav", synthetic_wav_bytes, "audio/wav")}
    response = client.post("/assess", files=files, data={"expected_text": "Guten Morgen"})
    assert response.status_code == 200
    data = response.json()
    assert data["transcription"] == "Guten Morgen"
    assert data["phoneme_error_rate"] == 0.05
    assert data["errors"] == [
        {
            "word": "Morgen",
            "expected_ipa": "mɔʁɡn̩",
            "actual_ipa": "mɔʁɡŋ",
            "confidence": 0.0,
        }
    ]


def test_device_resolution_and_env():
    """Test resolution of device and setting of OPENPRONOUNCE_DEVICE env var."""
    dev_cpu = resolve_device("cpu")
    assert dev_cpu == "cpu"
    assert os.environ.get("OPENPRONOUNCE_DEVICE") == "cpu"

    dev_cuda = resolve_device("cuda")
    assert dev_cuda == "cuda"
    assert os.environ.get("OPENPRONOUNCE_DEVICE") == "cuda"

    dev_mps = resolve_device("mps")
    assert dev_mps == "mps"
    assert os.environ.get("OPENPRONOUNCE_DEVICE") == "mps"

    dev_auto = resolve_device("auto")
    if torch.cuda.is_available():
        assert dev_auto == "cuda"
    elif torch.backends.mps.is_available():
        assert dev_auto == "mps"
    else:
        assert dev_auto == "cpu"
    assert os.environ.get("OPENPRONOUNCE_DEVICE") == dev_auto


def test_parse_args():
    """Test CLI arguments parsing with default and custom values."""
    defaults = parse_args([])
    assert defaults.host == "0.0.0.0"
    assert defaults.port == 8002
    assert defaults.device == "auto"

    custom = parse_args([
        "--host", "127.0.0.1",
        "--port", "8082",
        "--device", "cuda",
    ])
    assert custom.host == "127.0.0.1"
    assert custom.port == 8082
    assert custom.device == "cuda"


def test_cors_headers(client):
    """Verify CORS headers are present for cross-origin browser requests."""
    response = client.options(
        "/assess",
        headers={
            "Origin": "http://192.168.1.50:8000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://192.168.1.50:8000"
    assert response.headers.get("access-control-allow-credentials") == "true"




def test_tts_defaults_to_piper(client, monkeypatch):
    """The reference voice is local Piper unless --tts gtts is chosen."""
    monkeypatch.delenv("OPENPRONOUNCE_TTS", raising=False)
    assert parse_args([]).tts == "piper"
    assert parse_args(["--tts", "gtts"]).tts == "gtts"
    assert client.get("/health").json()["tts"] in ("piper", "gtts")
