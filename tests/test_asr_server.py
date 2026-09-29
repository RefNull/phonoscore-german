"""
Tests for asr_server.py.
"""

import io
from pathlib import Path
import sys
import wave
from unittest.mock import MagicMock

# Ensure repository root is in sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pytest
from fastapi.testclient import TestClient
import torch

from asr_server import (
    MODEL_ID,
    app,
    parse_args,
    resolve_device_and_dtype,
    resolve_model_id,
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
def mock_transcriber(monkeypatch):
    """Mock the (audio_bytes, language) -> text callable returned by get_transcriber."""
    mock = MagicMock(return_value="Ich habe morgen einen Termin beim Arzt.")
    monkeypatch.setattr("asr_server.get_transcriber", lambda: mock)
    return mock


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc


def test_health_endpoint(client):
    """GET /health should return 200 OK and expected service metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "asr"
    assert "device" in data
    assert isinstance(data["device"], str)
    assert data["device"] in ("cuda", "mps", "cpu")
    assert "model" in data
    assert data["model"] == MODEL_ID


def test_transcription_success(client, mock_transcriber, synthetic_wav_bytes):
    """POST /v1/audio/transcriptions should transcribe audio and return standard OpenAI-compatible format."""
    files = {"file": ("recording.wav", synthetic_wav_bytes, "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data == {"text": "Ich habe morgen einen Termin beim Arzt."}

    # Language is forwarded positionally to the transcriber, default "de"
    assert mock_transcriber.called
    call_args, _ = mock_transcriber.call_args
    assert call_args == (synthetic_wav_bytes, "de")


def test_transcription_custom_language(client, mock_transcriber, synthetic_wav_bytes):
    """POST /v1/audio/transcriptions with custom language parameter."""
    files = {"file": ("recording.wav", synthetic_wav_bytes, "audio/wav")}
    data = {"language": "en"}
    response = client.post("/v1/audio/transcriptions", files=files, data=data)

    assert response.status_code == 200
    assert mock_transcriber.call_args.args[1] == "en"


def test_transcription_blank_language_autodetects(client, mock_transcriber, synthetic_wav_bytes):
    """A blank language form field is passed as None so Qwen3-ASR auto-detects."""
    files = {"file": ("audio.wav", synthetic_wav_bytes, "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files, data={"language": " "})
    assert response.status_code == 200
    assert mock_transcriber.call_args.args[1] is None


def test_transcription_empty_file(client):
    """Uploading an empty (0-byte) file should return 400 Bad Request."""
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files)

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_transcription_missing_file(client):
    """Sending request without 'file' multipart parameter should return 422 Unprocessable Entity."""
    response = client.post("/v1/audio/transcriptions", data={"language": "de"})
    assert response.status_code == 422


def test_transcription_model_load_failure(client, monkeypatch):
    """A model that fails to load should return 500 with the reason."""
    def fail():
        raise OSError("model not found")
    monkeypatch.setattr("asr_server.get_transcriber", fail)
    files = {"file": ("test.wav", make_synthetic_wav(), "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files)
    assert response.status_code == 500
    assert "model not found" in response.json()["detail"]


def test_transcription_pipeline_failure(client, monkeypatch, synthetic_wav_bytes):
    """Pipeline failure should return 500 and clean up temporary files."""
    mock = MagicMock(side_effect=RuntimeError("Audio decoding failed"))
    monkeypatch.setattr("asr_server.get_transcriber", lambda: mock)
    files = {"file": ("corrupt.wav", synthetic_wav_bytes, "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files)

    assert response.status_code == 500
    assert "Transcription failed" in response.json()["detail"]


def test_device_and_dtype_resolution():
    """Test resolution of device and torch.dtype."""
    dev_cpu, dt_cpu = resolve_device_and_dtype("cpu")
    assert dev_cpu == "cpu"
    assert dt_cpu == torch.float32

    dev_cuda, dt_cuda = resolve_device_and_dtype("cuda")
    assert dev_cuda == "cuda"
    assert dt_cuda == torch.bfloat16

    dev_mps, dt_mps = resolve_device_and_dtype("mps")
    assert dev_mps == "mps"
    assert dt_mps == torch.bfloat16

    dev_auto, dt_auto = resolve_device_and_dtype("auto")
    if torch.cuda.is_available():
        assert dev_auto == "cuda"
        assert dt_auto == torch.bfloat16
    elif torch.backends.mps.is_available():
        assert dev_auto == "mps"
        assert dt_auto == torch.bfloat16
    else:
        assert dev_auto == "cpu"
        assert dt_auto == torch.float32


@pytest.mark.parametrize(
    ("input_name", "expected_id"),
    [
        ("0.6B", "Qwen/Qwen3-ASR-0.6B-hf"),
        ("0.6b", "Qwen/Qwen3-ASR-0.6B-hf"),
        ("Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-0.6B-hf"),
        ("qwen3-asr-0.6b", "Qwen/Qwen3-ASR-0.6B-hf"),
        ("1.7B", "Qwen/Qwen3-ASR-1.7B-hf"),
        ("1.7b", "Qwen/Qwen3-ASR-1.7B-hf"),
        ("Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-1.7B-hf"),
        ("qwen3-asr-1.7b", "Qwen/Qwen3-ASR-1.7B-hf"),
        ("my-org/custom-model", "my-org/custom-model"),
        ("  my-org/custom-model  ", "my-org/custom-model"),
        ("/models/local-asr", "/models/local-asr"),
    ],
)
def test_resolve_model_id(input_name, expected_id):
    """Verify model resolution for presets, custom HF repo IDs, and local paths."""
    assert resolve_model_id(input_name) == expected_id


def test_parse_args(monkeypatch):
    """Test CLI arguments parsing with default, preset, custom values, and env var."""
    defaults = parse_args([])
    assert defaults.host == "0.0.0.0"
    assert defaults.port == 8001
    assert defaults.device == "auto"
    assert defaults.model_id == "Qwen/Qwen3-ASR-1.7B-hf"

    custom = parse_args([
        "--host", "127.0.0.1",
        "--port", "8081",
        "--device", "cuda",
        "--model", "custom-org/custom-model",
    ])
    assert custom.host == "127.0.0.1"
    assert custom.port == 8081
    assert custom.device == "cuda"
    assert custom.model_id == "custom-org/custom-model"

    preset = parse_args(["--model", "Qwen3-ASR-1.7B"])
    assert preset.model_id == "Qwen/Qwen3-ASR-1.7B-hf"

    custom_model = parse_args(["--model", "my-org/custom"])
    assert custom_model.model_id == "my-org/custom"

    monkeypatch.setenv("PHONOSCORE_ASR_MODEL", "0.6b")
    env_args = parse_args([])
    assert env_args.model_id == "Qwen/Qwen3-ASR-0.6B-hf"


def test_cors_headers(client):
    """Verify CORS headers are present for cross-origin browser requests."""
    response = client.options(
        "/v1/audio/transcriptions",
        headers={
            "Origin": "http://192.168.1.50:8000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://192.168.1.50:8000"
    assert response.headers.get("access-control-allow-credentials") == "true"




def test_parse_args_endpoint_keeps_model_name():
    """With --endpoint, --model is the remote model name, not a local preset."""
    args = parse_args(["--endpoint", "https://api.example.com/v1/", "--model", "whisper-1"])
    assert args.endpoint == "https://api.example.com/v1"
    assert args.model_id == "whisper-1"


def test_transcribe_remote(monkeypatch):
    """The remote transcriber forwards audio, model, language and API key."""
    monkeypatch.setattr("asr_server.ENDPOINT", "https://api.example.com/v1")
    monkeypatch.setattr("asr_server.MODEL_ID", "whisper-1")
    monkeypatch.setenv("PHONOSCORE_ASR_API_KEY", "sk-test")
    response = MagicMock(ok=True)
    response.json.return_value = {"text": "Guten Tag"}
    post = MagicMock(return_value=response)
    monkeypatch.setattr("asr_server.requests.post", post)

    import asr_server
    assert asr_server.get_transcriber() is asr_server.transcribe_remote
    assert asr_server.transcribe_remote(b"wav", "de", filename="a.wav") == "Guten Tag"

    url = post.call_args.args[0]
    kwargs = post.call_args.kwargs
    assert url == "https://api.example.com/v1/audio/transcriptions"
    assert kwargs["headers"] == {"Authorization": "Bearer sk-test"}
    assert kwargs["data"] == {"model": "whisper-1", "language": "de"}
    assert kwargs["files"]["file"] == ("a.wav", b"wav")


def test_transcribe_remote_error(monkeypatch):
    """A failing remote API surfaces its status and body."""
    monkeypatch.setattr("asr_server.ENDPOINT", "https://api.example.com/v1")
    monkeypatch.setattr("asr_server.requests.post", MagicMock(return_value=MagicMock(ok=False, status_code=401, text="bad key")))
    import asr_server
    with pytest.raises(RuntimeError, match="401: bad key"):
        asr_server.transcribe_remote(b"wav", None)
