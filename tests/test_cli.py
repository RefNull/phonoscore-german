"""Tests for cli.py."""

import io
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

# Ensure repository root is in sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import pytest
import requests
import soundfile as sf

from cli import (
    array_to_wav_bytes,
    format_report,
    format_score_bar,
    load_audio_file,
    main,
    parse_args,
    query_asr,
    query_pronounce,
    record_audio,
)


def make_dummy_pcm16_array(num_samples: int = 1600, sample_rate: int = 16000) -> np.ndarray:
    """Generate a simple sine wave as int16 numpy array."""
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    return (sine * 32767).astype(np.int16).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Argument Parsing Tests
# ---------------------------------------------------------------------------

def test_parse_args_defaults():
    """Verify default CLI argument values."""
    args = parse_args([])
    assert args.text == "Ich habe morgen einen Termin beim Arzt."
    assert args.router_url == "http://localhost:8080"
    assert args.audio_file is None
    assert args.router is False
    assert args.asr_url == "http://localhost:8001"
    assert args.pronounce_url == "http://localhost:8002"
    assert args.samplerate == 16000


def test_parse_args_custom():
    """Verify custom CLI flags."""
    args = parse_args([
        "--text", "Wie heißen Sie?",
        "--router-url", "http://192.0.2.1:8080",
        "--audio-file", "samples/test.wav",
        "--router",
        "--asr-url", "http://localhost:8010",
        "--pronounce-url", "http://localhost:8020",
        "--samplerate", "48000",
    ])
    assert args.text == "Wie heißen Sie?"
    assert args.router_url == "http://192.0.2.1:8080"
    assert args.audio_file == "samples/test.wav"
    assert args.router is True
    assert args.asr_url == "http://localhost:8010"
    assert args.pronounce_url == "http://localhost:8020"
    assert args.samplerate == 48000


# ---------------------------------------------------------------------------
# Audio Processing & Capture Tests
# ---------------------------------------------------------------------------

def test_array_to_wav_bytes():
    """Verify audio numpy array conversion to valid 16-bit PCM WAV bytes."""
    data = make_dummy_pcm16_array(1600, 16000)
    wav_bytes = array_to_wav_bytes(data, samplerate=16000)

    assert isinstance(wav_bytes, bytes)
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes[:16]

    # Verify soundfile can decode back accurately
    buf = io.BytesIO(wav_bytes)
    read_data, read_rate = sf.read(buf, dtype="int16")
    assert read_rate == 16000
    assert len(read_data) == 1600
    assert np.allclose(read_data, data.squeeze(), atol=1)


def test_load_audio_file_success(tmp_path):
    """Verify loading pre-recorded audio file returns exact bytes."""
    sample_file = tmp_path / "sample.wav"
    sample_bytes = b"RIFF" + b"\x00" * 100
    sample_file.write_bytes(sample_bytes)

    loaded = load_audio_file(sample_file)
    assert loaded == sample_bytes


def test_load_audio_file_not_found():
    """Verify FileNotFoundError when audio file does not exist."""
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        load_audio_file("nonexistent_path_xyz.wav")


def test_load_audio_file_empty(tmp_path):
    """Verify ValueError when audio file is empty (0 bytes)."""
    empty_file = tmp_path / "empty.wav"
    empty_file.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        load_audio_file(empty_file)


def test_record_audio_mocked():
    """Verify microphone recording flow with mocked InputStream."""
    dummy_data = make_dummy_pcm16_array(1600, 16000)

    class MockInputStream:
        def __init__(self, samplerate, channels, dtype, callback):
            self.samplerate = samplerate
            self.channels = channels
            self.dtype = dtype
            self.callback = callback

        def __enter__(self):
            # Simulate PortAudio feeding dummy chunk
            self.callback(dummy_data, len(dummy_data), None, 0)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_input_calls = []

    def mock_input(prompt=""):
        mock_input_calls.append(prompt)
        return ""

    with patch("cli.sd", MagicMock(InputStream=MockInputStream)):
        wav_bytes = record_audio(samplerate=16000, input_fn=mock_input)

    assert len(mock_input_calls) == 2
    assert "start recording" in mock_input_calls[0]
    assert "stop" in mock_input_calls[1]
    assert wav_bytes.startswith(b"RIFF")


def test_record_audio_stream_failure():
    """Verify RuntimeError is raised when microphone stream fails."""
    with patch("cli.sd", MagicMock(InputStream=MagicMock(side_effect=RuntimeError("PortAudio device error")))):
        with pytest.raises(RuntimeError, match="Microphone recording failed"):
            record_audio(input_fn=lambda prompt="": None)


def test_record_audio_sounddevice_missing():
    """Verify informative RuntimeError when sounddevice is None."""
    with patch("cli.sd", None):
        with pytest.raises(RuntimeError, match="No microphone support"):
            record_audio(input_fn=lambda prompt="": None)


def test_main_pre_recorded_audio_without_sounddevice(tmp_path):
    """Verify --audio-file works seamlessly even if sounddevice is None (headless/SSH)."""
    sample_wav = tmp_path / "test.wav"
    sample_wav.write_bytes(b"RIFF" + b"\x00" * 100)

    with patch("cli.sd", None), \
         patch("cli.query_asr", return_value="Ich habe morgen einen Termin."), \
         patch("cli.query_pronounce", return_value={
             "score": 90.0,
             "transcription": "Ich habe morgen einen Termin.",
             "phoneme_error_rate": 0.05,
             "errors": []
         }):
        exit_code = main([
            "--text", "Ich habe morgen einen Termin.",
            "--audio-file", str(sample_wav),
        ])
        assert exit_code == 0



# ---------------------------------------------------------------------------
# ASR requests
# ---------------------------------------------------------------------------

def test_query_asr_routed_success():
    """Verify ASR request format through llama-swap router."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"text": "Ich habe morgen einen Termin beim Arzt."}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = query_asr(
            wav_bytes=b"dummy_wav",
            target_url="http://localhost:8080",
            direct=False,
            language="de",
        )

    assert result == "Ich habe morgen einen Termin beim Arzt."
    assert mock_post.call_count == 1
    call_url = mock_post.call_args[0][0]
    call_kwargs = mock_post.call_args[1]

    assert call_url == "http://localhost:8080/v1/audio/transcriptions"
    assert call_kwargs["data"] == {"language": "de", "model": "qwen3-asr"}
    assert "file" in call_kwargs["files"]
    assert call_kwargs["files"]["file"][1] == b"dummy_wav"


def test_query_asr_direct_success():
    """Verify ASR request format when querying ASR server directly."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"text": "Guten Tag"}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = query_asr(
            wav_bytes=b"dummy_wav",
            target_url="http://localhost:8001/",
            direct=True,
            language="de",
        )

    assert result == "Guten Tag"
    call_url = mock_post.call_args[0][0]
    call_kwargs = mock_post.call_args[1]

    assert call_url == "http://localhost:8001/v1/audio/transcriptions"
    assert call_kwargs["data"] == {"language": "de"}
    assert "model" not in call_kwargs["data"]


def test_query_asr_connection_error_routed():
    """Verify descriptive error when router connection is refused."""
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        with pytest.raises(ConnectionRefusedError) as exc_info:
            query_asr(b"dummy", "http://localhost:8080", direct=False)
        assert "llama-swap router" in str(exc_info.value)
        assert "llama-swap --config" in str(exc_info.value)


def test_query_asr_connection_error_direct():
    """Verify descriptive error when direct ASR connection is refused."""
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        with pytest.raises(ConnectionRefusedError) as exc_info:
            query_asr(b"dummy", "http://localhost:8001", direct=True)
        assert "ASR server" in str(exc_info.value)
        assert "python serve.py" in str(exc_info.value)


def test_query_asr_http_error():
    """Verify RuntimeError when ASR server returns non-200 status."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="status 500"):
            query_asr(b"dummy", "http://localhost:8001", direct=True)


# ---------------------------------------------------------------------------
# Scoring requests
# ---------------------------------------------------------------------------

def test_query_pronounce_routed_success():
    """Verify OpenPronounce request format through llama-swap router."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "score": 82.5,
        "transcription": "Ich habe morgen einen Termin beim Arzt.",
        "phoneme_error_rate": 0.12,
        "errors": [
            {
                "word": "Termin",
                "expected_ipa": "tɛʁˈmiːn",
                "actual_ipa": "tɛɹˈmiːn",
                "confidence": 0.85,
            }
        ],
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = query_pronounce(
            wav_bytes=b"dummy_wav",
            target_url="http://localhost:8080",
            direct=False,
            expected_text="Ich habe morgen einen Termin beim Arzt.",
            lang="de",
        )

    assert result["score"] == 82.5
    call_url = mock_post.call_args[0][0]
    call_kwargs = mock_post.call_args[1]

    assert call_url == "http://localhost:8080/upstream/openpronounce/assess"
    assert call_kwargs["data"] == {
        "expected_text": "Ich habe morgen einen Termin beim Arzt.",
        "lang": "de",
    }
    assert call_kwargs["files"]["file"][1] == b"dummy_wav"


def test_query_pronounce_direct_success():
    """Verify OpenPronounce request format when querying directly."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "score": 90.0,
        "transcription": "Guten Morgen",
        "phoneme_error_rate": 0.05,
        "errors": [],
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = query_pronounce(
            wav_bytes=b"dummy_wav",
            target_url="http://localhost:8002",
            direct=True,
            expected_text="Guten Morgen",
            lang="de",
        )

    assert result["score"] == 90.0
    call_url = mock_post.call_args[0][0]
    assert call_url == "http://localhost:8002/assess"


def test_query_pronounce_connection_error():
    """Verify descriptive error when OpenPronounce connection is refused."""
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        with pytest.raises(ConnectionRefusedError) as exc_info:
            query_pronounce(b"dummy", "http://localhost:8002", direct=True)
        assert "Pronunciation server" in str(exc_info.value)
        assert "python serve.py" in str(exc_info.value)


def test_query_pronounce_http_error():
    """Verify RuntimeError when OpenPronounce server returns non-200 status."""
    mock_resp = MagicMock()
    mock_resp.status_code = 422
    mock_resp.text = "Unprocessable Entity"

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="status 422"):
            query_pronounce(b"dummy", "http://localhost:8002", direct=True)


# ---------------------------------------------------------------------------
# Terminal Report Presentation Tests
# ---------------------------------------------------------------------------

def test_format_score_bar():
    """Test visual score bar generation for edge and intermediate scores."""
    bar_0 = format_score_bar(0.0, width=10)
    assert bar_0 == "[          ] 0.0 / 100"

    bar_100 = format_score_bar(100.0, width=10)
    assert bar_100 == "[==========] 100.0 / 100"

    bar_50 = format_score_bar(50.0, width=10)
    assert bar_50 == "[====>     ] 50.0 / 100"

    bar_82 = format_score_bar(82.5, width=10)
    assert bar_82 == "[=======>  ] 82.5 / 100"

    # Out of bounds clamping
    bar_neg = format_score_bar(-10.0, width=10)
    assert bar_neg == "[          ] 0.0 / 100"

    bar_over = format_score_bar(150.0, width=10)
    assert bar_over == "[==========] 100.0 / 100"


def test_format_report_with_errors():
    """Verify report formatting when phonetic errors exist."""
    assessment = {
        "score": 82.5,
        "transcription": "Ich habe morgen einen Termin beim Arzt.",
        "phoneme_error_rate": 0.12,
        "errors": [
            {
                "word": "Termin",
                "expected_ipa": "tɛʁˈmiːn",
                "actual_ipa": "tɛɹˈmiːn",
                "confidence": 0.85,
            }
        ],
    }

    report = format_report(
        target_text="Ich habe morgen einen Termin beim Arzt.",
        asr_text="Ich habe morgen einen Termin beim Arzt.",
        assessment=assessment,
    )

    # Check Score
    assert "82.5 / 100" in report
    # Check PER
    assert "12.0%" in report
    # Check Transcriptions
    assert "Expected:             Ich habe morgen einen Termin beim Arzt." in report
    assert "Heard (Qwen3-ASR):    Ich habe morgen einen Termin beim Arzt." in report
    assert "Heard (Wav2Vec2):     Ich habe morgen einen Termin beim Arzt." in report
    # Check Flagged Word details
    assert "Mispronounced words (1):" in report
    assert "Termin" in report
    assert "tɛʁˈmiːn" in report
    assert "tɛɹˈmiːn" in report
    assert "85.0%" in report


def test_format_report_no_errors():
    """Verify report formatting when pronunciation is perfect."""
    assessment = {
        "score": 100.0,
        "transcription": "Guten Tag",
        "phoneme_error_rate": 0.0,
        "errors": [],
    }

    report = format_report(
        target_text="Guten Tag",
        asr_text="Guten Tag",
        assessment=assessment,
    )

    assert "100.0 / 100" in report
    assert "0.0%" in report
    assert "No mispronounced words." in report


# ---------------------------------------------------------------------------
# End-to-End Evaluation Flow Tests
# ---------------------------------------------------------------------------

def test_main_pre_recorded_audio_evaluation_flow(tmp_path, capsys):
    """Verify end-to-end client execution with pre-recorded audio and mocked APIs."""
    audio_path = tmp_path / "audio.wav"
    dummy_wav = array_to_wav_bytes(make_dummy_pcm16_array())
    audio_path.write_bytes(dummy_wav)

    mock_asr_resp = MagicMock()
    mock_asr_resp.status_code = 200
    mock_asr_resp.json.return_value = {"text": "Ich habe morgen einen Termin beim Arzt."}

    mock_pronounce_resp = MagicMock()
    mock_pronounce_resp.status_code = 200
    mock_pronounce_resp.json.return_value = {
        "score": 85.0,
        "transcription": "Ich habe morgen einen Termin beim Arzt.",
        "phoneme_error_rate": 0.10,
        "errors": [
            {
                "word": "Termin",
                "expected_ipa": "tɛʁˈmiːn",
                "actual_ipa": "tɛɹˈmiːn",
                "confidence": 0.90,
            }
        ],
    }

    def mock_post_dispatch(url, *args, **kwargs):
        if "transcriptions" in url:
            return mock_asr_resp
        elif "assess" in url:
            return mock_pronounce_resp
        raise ValueError(f"Unexpected url: {url}")

    with patch("requests.post", side_effect=mock_post_dispatch) as mock_post:
        exit_code = main([
            "--audio-file", str(audio_path),
        ])

    assert exit_code == 0
    # Direct to the serve.py ports by default
    urls = [c.args[0] for c in mock_post.call_args_list]
    assert urls == ["http://localhost:8001/v1/audio/transcriptions", "http://localhost:8002/assess"]
    captured = capsys.readouterr()
    assert "PhonoScore German" in captured.out
    assert "85.0 / 100" in captured.out
    assert "Termin" in captured.out


def test_main_missing_audio_file(capsys):
    """Verify main returns exit code 1 when pre-recorded audio file is missing."""
    exit_code = main(["--audio-file", "missing_sample_file.wav"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Failed to load audio file" in captured.err


def test_main_connection_error(tmp_path, capsys):
    """Verify main handles backend connection refusal with exit code 1."""
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(array_to_wav_bytes(make_dummy_pcm16_array()))

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Refused")):
        exit_code = main([
            "--audio-file", str(audio_path),
        ])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Connection Error" in captured.err


def test_main_router_flag_uses_llama_swap(tmp_path):
    """--router sends both requests to the llama-swap URL."""
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(array_to_wav_bytes(make_dummy_pcm16_array()))

    with patch("cli.query_asr", return_value="Hallo") as mock_asr, \
         patch("cli.query_pronounce", return_value={"score": 90.0, "errors": []}) as mock_pronounce:
        exit_code = main(["--audio-file", str(audio_path), "--router", "--router-url", "http://192.0.2.1:8080"])

    assert exit_code == 0
    assert mock_asr.call_args.kwargs["target_url"] == "http://192.0.2.1:8080"
    assert mock_asr.call_args.kwargs["direct"] is False
    assert mock_pronounce.call_args.kwargs["target_url"] == "http://192.0.2.1:8080"
    assert mock_pronounce.call_args.kwargs["direct"] is False
