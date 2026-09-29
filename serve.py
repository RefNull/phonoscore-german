#!/usr/bin/env python3
"""Run the PhonoScore ASR and pronunciation services together.

Creates .venv on first run, supervises both processes, and writes logs to ./logs.
Supports --detach, --status, --logs and --stop.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Optional
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [serve]: %(message)s",
)
logger = logging.getLogger("serve")


def get_default_pid_file() -> Path:
    return Path(__file__).resolve().parent / ".serve.pid"


def get_default_log_dir() -> Path:
    return Path(__file__).resolve().parent / "logs"


def read_pid(pid_file: Path) -> Optional[int]:
    if not pid_file.is_file():
        return None
    try:
        content = pid_file.read_text().strip()
        return int(content) if content else None
    except Exception:
        return None


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def check_dependencies(python_exe: str) -> bool:
    """Return True if python_exe can import fastapi and uvicorn."""
    cmd = [python_exe, "-c", "import fastapi; import uvicorn"]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return res.returncode == 0
    except Exception:
        return False


def bootstrap_environment(repo_dir: Path) -> Optional[str]:
    """Create .venv and install requirements.txt if needed; return its python or None."""
    venv_dir = repo_dir / ".venv"
    venv_python = (
        venv_dir / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else venv_dir / "bin" / "python"
    )

    if not venv_python.exists():
        logger.info("Creating virtual environment in %s", venv_dir)
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        except Exception as exc:
            logger.error("Could not create virtual environment: %s", exc)
            return None

    requirements = repo_dir / "requirements.txt"
    if requirements.exists() and not check_dependencies(str(venv_python)):
        logger.info("Installing requirements.txt into .venv (takes a few minutes)")
        try:
            subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=False)
            subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(requirements)], check=True)
        except Exception as exc:
            logger.error("pip install failed: %s", exc)
            return None

    if check_dependencies(str(venv_python)):
        return str(venv_python)
    return None


def resolve_python_executable(explicit_exe: Optional[str] = None, auto_setup: bool = True) -> str:
    """Choose the interpreter for the services.

    Order: --python, the active virtualenv, ./.venv or ./venv, a freshly bootstrapped .venv
    (if auto_setup), then sys.executable.
    """
    if explicit_exe:
        return explicit_exe

    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return sys.executable

    repo_dir = Path(__file__).resolve().parent
    candidates = [
        repo_dir / ".venv" / "bin" / "python",
        repo_dir / ".venv" / "Scripts" / "python.exe",
        repo_dir / "venv" / "bin" / "python",
        repo_dir / "venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    if auto_setup:
        bootstrapped = bootstrap_environment(repo_dir)
        if bootstrapped:
            return bootstrapped

    return sys.executable


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    default_model = os.environ.get("PHONOSCORE_ASR_MODEL", "Qwen3-ASR-0.6B")

    parser = argparse.ArgumentParser(
        description="Run the PhonoScore ASR and pronunciation services together.",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        default=False,
        help="stop the background deployment",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        default=False,
        help="show whether the services are up",
    )
    parser.add_argument(
        "--logs",
        "-f",
        dest="logs",
        action="store_true",
        default=False,
        help="follow the combined log",
    )
    parser.add_argument(
        "--detach",
        "-d",
        dest="detach",
        action="store_true",
        default=False,
        help="run in the background",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--asr-port",
        type=int,
        default=8001,
        help="ASR service port (default: 8001)",
    )
    parser.add_argument(
        "--pronounce-port",
        type=int,
        default=8002,
        help="pronunciation service and web UI port (default: 8002)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help=f"Qwen3-ASR-0.6B, Qwen3-ASR-1.7B, a Hugging Face repo or a local path (default: {default_model})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cuda, mps or cpu (default: auto)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="interpreter for the services (default: .venv, created if missing)",
    )
    parser.add_argument(
        "--no-auto-setup",
        action="store_true",
        default=False,
        help="do not create .venv or run pip install",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="log directory (default: ./logs)",
    )
    parser.add_argument(
        "--pid-file",
        type=str,
        default=None,
        help="PID file for --detach (default: ./.serve.pid)",
    )
    parser.add_argument(
        "--internal-worker",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,  # set by --detach on the background process
    )

    parsed = parser.parse_args(args)
    if not (parsed.stop or parsed.status or parsed.logs):
        if parsed.asr_port == parsed.pronounce_port:
            parser.error(f"--asr-port and --pronounce-port cannot be identical ({parsed.asr_port})")

    return parsed


def build_commands(
    args: argparse.Namespace,
    python_exe: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    """Return the (asr, pronounce) command lines."""
    exe = python_exe or sys.executable
    asr_cmd = [
        exe,
        "-u",
        "asr_server.py",
        "--host",
        str(args.host),
        "--port",
        str(args.asr_port),
        "--device",
        str(args.device),
        "--model",
        str(args.model),
    ]
    pronounce_cmd = [
        exe,
        "-u",
        "pronounce_server.py",
        "--host",
        str(args.host),
        "--port",
        str(args.pronounce_port),
        "--device",
        str(args.device),
    ]
    return asr_cmd, pronounce_cmd


def terminate_process(proc: subprocess.Popen, name: str, timeout: float = 5.0) -> None:
    """SIGTERM the process; SIGKILL it if it hasn't exited after timeout seconds."""
    if proc.poll() is not None:
        return

    logger.info("Stopping %s (PID %d)...", name, proc.pid)
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
        logger.info("%s terminated gracefully.", name)
    except subprocess.TimeoutExpired:
        logger.warning("%s did not exit within %.1fs; sending SIGKILL.", name, timeout)
        proc.kill()
        proc.wait()
    except Exception as exc:
        logger.error("Error stopping %s: %s", name, exc)


def handle_stop(pid_file: Path) -> int:
    pid = read_pid(pid_file)
    if not pid or not is_pid_running(pid):
        pid_file.unlink(missing_ok=True)
        print("PhonoScore is not running.")
        return 0

    print(f"Stopping PhonoScore (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        print(f"Process {pid} has already exited.")
        return 0

    deadline = time.time() + 6.0
    while time.time() < deadline:
        if not is_pid_running(pid):
            break
        time.sleep(0.2)

    if is_pid_running(pid):
        print(f"Process {pid} did not terminate within timeout; sending SIGKILL...")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    pid_file.unlink(missing_ok=True)
    print("Stopped.")
    return 0


def probe_health(url: str, timeout: float = 2.0) -> Optional[dict]:
    """GET a /health URL; return its JSON, or None if it isn't answering."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "serve-status"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    return None


def local_url(host: str, port: int) -> str:
    """URL for reaching a service bound to host from this machine."""
    return f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"


def handle_status(pid_file: Path, host: str, asr_port: int, pronounce_port: int, log_dir: Path) -> int:
    pid = read_pid(pid_file)
    running = pid is not None and is_pid_running(pid)

    asr_health = probe_health(f"{local_url(host, asr_port)}/health")
    pronounce_health = probe_health(f"{local_url(host, pronounce_port)}/health")

    print(f"Supervisor:  {f'RUNNING (PID {pid})' if running else 'NOT RUNNING'}")

    if asr_health:
        print(f"ASR:         UP on port {asr_port} (model: {asr_health.get('model', 'unknown')}, device: {asr_health.get('device', 'unknown')})")
    else:
        print(f"ASR:         {'STARTING' if running else 'STOPPED'} (port {asr_port})")

    if pronounce_health:
        print(f"Pronounce:   UP on port {pronounce_port} (device: {pronounce_health.get('device', 'unknown')})")
        print(f"Web UI:      {local_url(host, pronounce_port)}")
    else:
        print(f"Pronounce:   {'STARTING' if running else 'STOPPED'} (port {pronounce_port})")

    if log_dir.exists():
        print(f"Logs:        {log_dir}")
    return 0 if running else 1


def handle_logs(log_file: Path, follow: bool = False, lines_count: int = 50) -> int:
    """Print the last lines of log_file, then keep printing new lines if follow is set."""
    if not log_file.is_file():
        print(f"Log file not found: {log_file}")
        print("Start the deployment first with 'python serve.py'.")
        return 1

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()
        tail = all_lines[-lines_count:] if len(all_lines) > lines_count else all_lines
        for line in tail:
            sys.stdout.write(line)
        sys.stdout.flush()

        if follow:
            try:
                while True:
                    line = f.readline()
                    if line:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    else:
                        time.sleep(0.5)
            except KeyboardInterrupt:
                return 0
    return 0


def spawn_detached(args: argparse.Namespace) -> int:
    """Start run_combined in a new background session and record its PID."""
    pid_file = Path(args.pid_file).resolve() if args.pid_file else get_default_pid_file()
    existing_pid = read_pid(pid_file)
    if existing_pid and is_pid_running(existing_pid):
        print(f"PhonoScore is already running (PID {existing_pid}). Stop it with: python serve.py --stop")
        return 1

    log_dir = Path(args.log_dir).resolve() if args.log_dir else get_default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    serve_log = log_dir / "serve.log"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-worker",
        "--host", str(args.host),
        "--asr-port", str(args.asr_port),
        "--pronounce-port", str(args.pronounce_port),
        "--model", str(args.model),
        "--device", str(args.device),
        "--log-dir", str(log_dir),
        "--pid-file", str(pid_file),
    ]
    if args.python:
        cmd.extend(["--python", args.python])
    if args.no_auto_setup:
        cmd.append("--no-auto-setup")

    out_file = open(serve_log, "a", buffering=1, encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=out_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    time.sleep(0.6)
    ret = proc.poll()
    if ret is not None:
        print(f"Failed to start (exit code {ret}). See {serve_log}")
        return ret

    pid_file.write_text(str(proc.pid))
    print(f"PhonoScore started in the background (PID {proc.pid}).")
    print(f"  Web UI:  {local_url(args.host, args.pronounce_port)} (once --status shows UP)")
    print(f"  Logs:    {serve_log}")
    print("  Status:  python serve.py --status")
    print("  Stop:    python serve.py --stop")
    return 0


def run_combined(args: argparse.Namespace, poll_interval: float = 0.5) -> int:
    """Run both services until interrupted; if one exits, stop the other and return its code."""
    auto_setup = not getattr(args, "no_auto_setup", False)
    python_exe = resolve_python_executable(getattr(args, "python", None), auto_setup=auto_setup)

    if not check_dependencies(python_exe):
        logger.error(
            "%s cannot import fastapi/uvicorn. Run without --no-auto-setup, or install manually:\n"
            "  source .venv/bin/activate && pip install -r requirements.txt",
            python_exe,
        )
        return 1

    log_dir = Path(args.log_dir).resolve() if args.log_dir else get_default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    pid_file = Path(args.pid_file).resolve() if args.pid_file else get_default_pid_file()
    if getattr(args, "internal_worker", False):
        pid_file.write_text(str(os.getpid()))

    asr_log_path = log_dir / "asr.log"
    pronounce_log_path = log_dir / "pronounce.log"
    serve_log_path = log_dir / "serve.log"

    asr_log_f = open(asr_log_path, "a", buffering=1, encoding="utf-8")
    pronounce_log_f = open(pronounce_log_path, "a", buffering=1, encoding="utf-8")
    serve_log_f = open(serve_log_path, "a", buffering=1, encoding="utf-8")

    asr_cmd, pronounce_cmd = build_commands(args, python_exe=python_exe)

    logger.info("Starting PhonoScore (python: %s)", python_exe)
    logger.info("  ASR:        port %d (model: %s, device: %s)", args.asr_port, args.model, args.device)
    logger.info("  Pronounce:  port %d (device: %s)", args.pronounce_port, args.device)
    logger.info("  Logs:       %s", log_dir)

    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"

    asr_proc = subprocess.Popen(
        asr_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
    )
    pronounce_proc = subprocess.Popen(
        pronounce_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
    )

    def stream_forwarder(pipe, prefix: str, dedicated_log, unified_log):
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                dedicated_log.write(line)
                unified_line = f"{prefix} {line}"
                unified_log.write(unified_line)
                sys.stdout.write(unified_line)
                sys.stdout.flush()
        except Exception:
            pass
        finally:
            pipe.close()

    t_asr = threading.Thread(
        target=stream_forwarder,
        args=(asr_proc.stdout, "[asr]", asr_log_f, serve_log_f),
        daemon=True,
    )
    t_pronounce = threading.Thread(
        target=stream_forwarder,
        args=(pronounce_proc.stdout, "[pronounce]", pronounce_log_f, serve_log_f),
        daemon=True,
    )
    t_asr.start()
    t_pronounce.start()

    procs = [("ASR Service", asr_proc), ("Pronounce Service", pronounce_proc)]
    shutdown_initiated = False

    def handle_signal(signum, _frame):
        nonlocal shutdown_initiated
        if shutdown_initiated:
            return
        shutdown_initiated = True
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, stopping services...", sig_name)
        for name, proc in procs:
            terminate_process(proc, name)
        asr_log_f.close()
        pronounce_log_f.close()
        serve_log_f.close()
        if getattr(args, "internal_worker", False):
            pid_file.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    web_url = local_url(args.host, args.pronounce_port)
    health_urls = [f"{local_url(args.host, port)}/health" for port in (args.asr_port, args.pronounce_port)]
    ready = False

    try:
        while True:
            if not ready and all(probe_health(url, timeout=0.5) for url in health_urls):
                ready = True
                logger.info("Ready. Open %s in your browser.", web_url)
            for name, proc in procs:
                ret = proc.poll()
                if ret is not None:
                    logger.error("%s exited unexpectedly with code %d", name, ret)
                    for sibling_name, sibling_proc in procs:
                        if sibling_proc is not proc:
                            terminate_process(sibling_proc, sibling_name)
                    asr_log_f.close()
                    pronounce_log_f.close()
                    serve_log_f.close()
                    if getattr(args, "internal_worker", False):
                        pid_file.unlink(missing_ok=True)
                    return ret
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        handle_signal(signal.SIGINT, None)
        return 0


def main(args: Optional[list[str]] = None) -> int:
    cli_args = parse_args(args)
    pid_file = Path(cli_args.pid_file).resolve() if cli_args.pid_file else get_default_pid_file()
    log_dir = Path(cli_args.log_dir).resolve() if cli_args.log_dir else get_default_log_dir()

    if cli_args.stop:
        return handle_stop(pid_file)

    if cli_args.status:
        return handle_status(pid_file, cli_args.host, cli_args.asr_port, cli_args.pronounce_port, log_dir)

    if cli_args.logs:
        return handle_logs(log_dir / "serve.log", follow=True)

    if cli_args.detach and not cli_args.internal_worker:
        return spawn_detached(cli_args)

    return run_combined(cli_args)


if __name__ == "__main__":
    sys.exit(main())
