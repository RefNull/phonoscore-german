"""Tests for serve.py."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import MagicMock, patch

# Ensure repository root is in sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pytest

from serve import (
    bootstrap_environment,
    build_commands,
    check_dependencies,
    handle_logs,
    handle_status,
    handle_stop,
    is_pid_running,
    parse_args,
    probe_health,
    read_pid,
    resolve_python_executable,
    run_combined,
    spawn_detached,
    terminate_process,
)


def test_parse_args_defaults():
    """Verify default CLI arguments."""
    args = parse_args([])
    assert args.host == "0.0.0.0"
    assert args.asr_port == 8001
    assert args.pronounce_port == 8002
    assert args.model == "Qwen3-ASR-0.6B"
    assert args.device == "auto"


def test_parse_args_custom():
    """Verify custom CLI argument parsing."""
    args = parse_args(
        [
            "--host",
            "127.0.0.1",
            "--asr-port",
            "9001",
            "--pronounce-port",
            "9002",
            "--model",
            "Qwen3-ASR-1.7B",
            "--device",
            "cuda",
        ]
    )
    assert args.host == "127.0.0.1"
    assert args.asr_port == 9001
    assert args.pronounce_port == 9002
    assert args.model == "Qwen3-ASR-1.7B"
    assert args.device == "cuda"


def test_parse_args_port_collision():
    """Verify that using identical ports for ASR and Pronounce triggers an error."""
    with pytest.raises(SystemExit):
        parse_args(["--asr-port", "8080", "--pronounce-port", "8080"])


def test_parse_args_env_model(monkeypatch):
    """Verify that PHONOSCORE_ASR_MODEL env var sets the default model."""
    monkeypatch.setenv("PHONOSCORE_ASR_MODEL", "Qwen3-ASR-1.7B")
    args = parse_args([])
    assert args.model == "Qwen3-ASR-1.7B"


def test_build_commands():
    """Verify construction of subprocess execution commands with unbuffered stdout flag."""
    args = parse_args(["--host", "127.0.0.1", "--asr-port", "8010", "--pronounce-port", "8020", "--model", "custom-model", "--device", "cpu"])
    asr_cmd, pronounce_cmd = build_commands(args, python_exe="/opt/python")

    assert asr_cmd == [
        "/opt/python",
        "-u",
        "asr_server.py",
        "--host",
        "127.0.0.1",
        "--port",
        "8010",
        "--device",
        "cpu",
        "--model",
        "custom-model",
    ]
    assert pronounce_cmd == [
        "/opt/python",
        "-u",
        "pronounce_server.py",
        "--host",
        "127.0.0.1",
        "--port",
        "8020",
        "--device",
        "cpu",
    ]


def test_terminate_process_already_exited():
    """Verify terminate_process does nothing if process has already stopped."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 0

    terminate_process(mock_proc, "TestService")
    mock_proc.terminate.assert_not_called()
    mock_proc.kill.assert_not_called()


def test_terminate_process_graceful():
    """Verify terminate_process calls terminate() and waits."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.pid = 12345

    terminate_process(mock_proc, "TestService", timeout=1.0)
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=1.0)
    mock_proc.kill.assert_not_called()


def test_terminate_process_timeout_fallback_to_kill():
    """Verify terminate_process sends kill() if terminate() times out."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.pid = 12345
    mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="test", timeout=1.0), None]

    terminate_process(mock_proc, "TestService", timeout=1.0)
    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_called_once()


@patch("serve.probe_health", return_value=None)
@patch("serve.check_dependencies", return_value=True)
@patch("serve.subprocess.Popen")
@patch("serve.signal.signal")
def test_run_combined_process_exit_supervision(mock_signal, mock_popen, mock_check_deps, mock_probe, tmp_path):
    """Verify run_combined terminates the sibling process if one exits unexpectedly."""
    proc1 = MagicMock()
    proc1.pid = 101
    proc1.poll.return_value = None

    proc2 = MagicMock()
    proc2.pid = 102
    # Simulate proc2 exiting unexpectedly with error code 1
    proc2.poll.return_value = 1

    mock_popen.side_effect = [proc1, proc2]

    args = parse_args(["--log-dir", str(tmp_path)])
    exit_code = run_combined(args, poll_interval=0.01)

    assert exit_code == 1
    # Check that sibling proc1 was terminated
    proc1.terminate.assert_called_once()


@patch("serve.check_dependencies", return_value=True)
@patch("serve.subprocess.Popen")
@patch("serve.signal.signal")
def test_run_combined_logs_ready_once(mock_signal, mock_popen, mock_check_deps, tmp_path, caplog):
    """The web UI URL is logged once, after both health checks pass."""
    asr_proc = MagicMock(pid=201)
    pronounce_proc = MagicMock(pid=202)
    asr_proc.poll.return_value = None
    # Keep running for a few polls, then exit so run_combined returns
    polls = iter([None, None, None, None])
    pronounce_proc.poll.side_effect = lambda: next(polls, 3)
    mock_popen.side_effect = [asr_proc, pronounce_proc]
    health = [None, {"status": "ok"}, {"status": "ok"}, {"status": "ok"}, {"status": "ok"}]

    args = parse_args(["--log-dir", str(tmp_path)])
    with patch("serve.probe_health", side_effect=lambda url, timeout: health.pop(0) if health else {"status": "ok"}), \
         caplog.at_level("INFO", logger="serve"):
        assert run_combined(args, poll_interval=0.01) == 3

    ready = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Ready")]
    assert ready == ["Ready. Open http://127.0.0.1:8002 in your browser."]


def test_parse_args_python():
    """Verify --python argument parsing."""
    args = parse_args(["--python", "/custom/bin/python"])
    assert args.python == "/custom/bin/python"


def test_resolve_python_executable_explicit():
    """Explicit argument overrides auto-detection."""
    assert resolve_python_executable("/custom/bin/python") == "/custom/bin/python"


def test_resolve_python_executable_venv_active(monkeypatch):
    """When inside an active virtualenv, sys.executable is returned."""
    monkeypatch.setattr(sys, "prefix", "/some/venv")
    monkeypatch.setattr(sys, "base_prefix", "/base/python")
    assert resolve_python_executable() == sys.executable


def test_resolve_python_executable_auto_detect(tmp_path, monkeypatch):
    """When not in a venv, auto-detects .venv/bin/python if present in repo root."""
    fake_venv_python = tmp_path / ".venv" / "bin" / "python"
    fake_venv_python.parent.mkdir(parents=True)
    fake_venv_python.touch(mode=0o755)

    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.setattr("serve.REPO_DIR", tmp_path)

    resolved = resolve_python_executable()
    assert resolved == str(fake_venv_python)


def test_check_dependencies():
    """Verify check_dependencies succeeds with current venv python."""
    assert check_dependencies(sys.executable) is True
    assert check_dependencies("/nonexistent/python") is False


def test_run_combined_missing_dependencies():
    """Verify run_combined returns 1 if check_dependencies fails."""
    args = parse_args(["--python", "/nonexistent/python"])
    assert run_combined(args) == 1


def test_parse_args_lifecycle_actions():
    """Verify parsing of lifecycle, detached, and logging flags."""
    args_stop = parse_args(["--stop"])
    assert args_stop.stop is True

    args_status = parse_args(["--status"])
    assert args_status.status is True

    args_logs = parse_args(["--logs"])
    assert args_logs.logs is True

    args_logs_f = parse_args(["-f"])
    assert args_logs_f.logs is True

    args_detach = parse_args(["--detach"])
    assert args_detach.detach is True

    args_detach_d = parse_args(["-d"])
    assert args_detach_d.detach is True

    args_no_auto = parse_args(["--no-auto-setup"])
    assert args_no_auto.no_auto_setup is True

    args_custom_paths = parse_args(["--log-dir", "/tmp/test-logs", "--pid-file", "/tmp/test.pid"])
    assert args_custom_paths.log_dir == Path("/tmp/test-logs")
    assert args_custom_paths.pid_file == Path("/tmp/test.pid")

    # Action flags bypass port collision check
    args_same_port = parse_args(["--status", "--asr-port", "8080", "--pronounce-port", "8080"])
    assert args_same_port.status is True


def test_read_and_is_pid_running(tmp_path):
    """Verify read_pid and is_pid_running helpers."""
    pid_file = tmp_path / "test.pid"
    assert read_pid(pid_file) is None

    pid_file.write_text("invalid_pid\n")
    assert read_pid(pid_file) is None

    pid_file.write_text("12345\n")
    assert read_pid(pid_file) == 12345

    assert is_pid_running(0) is False
    assert is_pid_running(-1) is False
    assert is_pid_running(os.getpid()) is True

    with patch("serve.os.kill", side_effect=OSError):
        assert is_pid_running(999999) is False


def test_handle_stop_not_running(tmp_path, capsys):
    """Verify handle_stop when no process is running."""
    pid_file = tmp_path / "serve.pid"

    # Case 1: PID file does not exist
    ret = handle_stop(pid_file)
    assert ret == 0
    captured = capsys.readouterr()
    assert "PhonoScore is not running." in captured.out

    # Case 2: PID file exists but process is dead
    pid_file.write_text("999999\n")
    with patch("serve.is_pid_running", return_value=False):
        ret = handle_stop(pid_file)
        assert ret == 0
        assert not pid_file.exists()


def test_handle_stop_running(tmp_path, capsys):
    """Verify handle_stop terminates a running process and cleans up PID file."""
    pid_file = tmp_path / "serve.pid"
    pid_file.write_text("12345\n")

    # Process is running initially, then stops after SIGTERM
    running_states = [True, True, False]
    with patch("serve.is_pid_running", side_effect=lambda pid: running_states.pop(0) if running_states else False), \
         patch("serve.os.kill") as mock_kill:
        ret = handle_stop(pid_file)
        assert ret == 0
        mock_kill.assert_called_with(12345, signal.SIGTERM)
        assert not pid_file.exists()


def test_handle_stop_timeout_fallback_to_kill(tmp_path):
    """Verify handle_stop sends SIGKILL if SIGTERM times out."""
    pid_file = tmp_path / "serve.pid"
    pid_file.write_text("12345\n")

    # Simulate process remaining alive past deadline
    with patch("serve.is_pid_running", return_value=True), \
         patch("serve.os.kill") as mock_kill, \
         patch("serve.time.time", side_effect=[0.0, 1.0, 10.0, 10.1]):
        ret = handle_stop(pid_file)
        assert ret == 0
        # Expect SIGTERM first, then SIGKILL
        assert mock_kill.call_count >= 2
        calls = [c[0][1] for c in mock_kill.call_args_list]
        assert signal.SIGTERM in calls
        assert signal.SIGKILL in calls
        assert not pid_file.exists()


def test_probe_health():
    """Verify probe_health succeeds with valid JSON and handles network errors."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"status": "ok", "service": "qwen3-asr"}'
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_resp

    with patch("serve.urllib.request.urlopen", return_value=mock_cm):
        result = probe_health("http://127.0.0.1:8001/health")
        assert result == {"status": "ok", "service": "qwen3-asr"}

    with patch("serve.urllib.request.urlopen", side_effect=Exception("Connection refused")):
        assert probe_health("http://127.0.0.1:8001/health") is None


def test_handle_status(tmp_path, capsys):
    """Verify handle_status report formatting for running and stopped states."""
    pid_file = tmp_path / "serve.pid"
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Case 1: Running with active services
    pid_file.write_text("54321")
    with patch("serve.is_pid_running", return_value=True), \
         patch("serve.probe_health", side_effect=[
             {"status": "ok", "model": "Qwen3-ASR-0.6B", "device": "cpu"},
             {"status": "ok", "device": "cpu"},
         ]):
        ret = handle_status(pid_file, "0.0.0.0", 8001, 8002, log_dir)
        assert ret == 0
        captured = capsys.readouterr()
        assert "Supervisor:  RUNNING (PID 54321)" in captured.out
        assert "ASR:         UP" in captured.out
        assert "Pronounce:   UP" in captured.out
        assert "Web UI:      http://127.0.0.1:8002" in captured.out

    # Case 2: Not running
    pid_file.unlink(missing_ok=True)
    with patch("serve.probe_health", return_value=None):
        ret = handle_status(pid_file, "0.0.0.0", 8001, 8002, log_dir)
        assert ret == 1
        captured = capsys.readouterr()
        assert "Supervisor:  NOT RUNNING" in captured.out
        assert "ASR:         STOPPED" in captured.out
        assert "Pronounce:   STOPPED" in captured.out


def test_handle_logs(tmp_path, capsys):
    """Verify handle_logs reading and error handling."""
    missing_log = tmp_path / "missing.log"
    assert handle_logs(missing_log) == 1

    sample_log = tmp_path / "serve.log"
    sample_log.write_text("line 1\nline 2\nline 3\n")
    assert handle_logs(sample_log, lines_count=2) == 0
    captured = capsys.readouterr()
    assert "line 2\nline 3\n" in captured.out


def test_spawn_detached(tmp_path, capsys):
    """Verify spawn_detached background execution logic."""
    pid_file = tmp_path / "serve.pid"
    log_dir = tmp_path / "logs"

    args = parse_args(["--detach", "--pid-file", str(pid_file), "--log-dir", str(log_dir)])

    # Case 1: Already running
    pid_file.write_text("11111")
    with patch("serve.is_pid_running", return_value=True):
        ret = spawn_detached(args)
        assert ret == 1
        captured = capsys.readouterr()
        assert "already running" in captured.out

    # Case 2: Successful spawn
    pid_file.unlink(missing_ok=True)
    mock_proc = MagicMock()
    mock_proc.pid = 9988
    mock_proc.poll.return_value = None

    with patch("serve.is_pid_running", return_value=False), \
         patch("serve.subprocess.Popen", return_value=mock_proc):
        ret = spawn_detached(args)
        assert ret == 0
        assert pid_file.read_text().strip() == "9988"
        captured = capsys.readouterr()
        assert "PhonoScore started in the background (PID 9988)." in captured.out

    # Case 3: Process immediately exits with error code
    pid_file.unlink(missing_ok=True)
    mock_proc_fail = MagicMock()
    mock_proc_fail.poll.return_value = 2

    with patch("serve.is_pid_running", return_value=False), \
         patch("serve.subprocess.Popen", return_value=mock_proc_fail):
        ret = spawn_detached(args)
        assert ret == 2


def test_bootstrap_environment(tmp_path):
    """Verify bootstrap_environment virtual environment creation and pip install."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "requirements.txt").write_text("fastapi\n")

    venv_python = repo_dir / ".venv" / "bin" / "python"

    def fake_subprocess_run(cmd, *args, **kwargs):
        if "-m" in cmd and "venv" in cmd:
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.touch(mode=0o755)
        return MagicMock(returncode=0)

    with patch("serve.subprocess.run", side_effect=fake_subprocess_run), \
         patch("serve.check_dependencies", side_effect=[False, True]):
        result = bootstrap_environment(repo_dir)
        assert result == str(venv_python)


