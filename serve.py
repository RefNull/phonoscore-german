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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [serve]: %(message)s")
logger = logging.getLogger("serve")

REPO_DIR = Path(__file__).resolve().parent


def read_pid(pid_file: Path) -> Optional[int]:
    try:
        return int(pid_file.read_text().strip())
    except (OSError, ValueError):
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
    try:
        return subprocess.run([python_exe, "-c", "import fastapi, uvicorn"], capture_output=True).returncode == 0
    except OSError:
        return False


def bootstrap_environment(repo_dir: Path) -> Optional[str]:
    """Create .venv and install requirements.txt if needed; return its python or None."""
    venv_python = repo_dir / ".venv" / "bin" / "python"
    try:
        if not venv_python.exists():
            logger.info("Creating virtual environment in %s", venv_python.parent.parent)
            subprocess.run([sys.executable, "-m", "venv", str(repo_dir / ".venv")], check=True)
        if not check_dependencies(str(venv_python)):
            logger.info("Installing requirements.txt into .venv (takes a few minutes)")
            subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(repo_dir / "requirements.txt")], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.error("Setting up .venv failed: %s", exc)
        return None
    return str(venv_python) if check_dependencies(str(venv_python)) else None


def resolve_python_executable(explicit_exe: Optional[str] = None, auto_setup: bool = True) -> str:
    """Choose the interpreter for the services.

    Order: --python, the active virtualenv, ./.venv, a freshly bootstrapped .venv (if auto_setup),
    then sys.executable.
    """
    if explicit_exe:
        return explicit_exe
    if sys.prefix != sys.base_prefix:
        return sys.executable
    venv_python = REPO_DIR / ".venv" / "bin" / "python"
    if venv_python.is_file() and os.access(venv_python, os.X_OK):
        return str(venv_python)
    if auto_setup:
        return bootstrap_environment(REPO_DIR) or sys.executable
    return sys.executable


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    default_model = os.environ.get("PHONOSCORE_ASR_MODEL", "Qwen3-ASR-0.6B")
    parser = argparse.ArgumentParser(description="Run the PhonoScore ASR and pronunciation services together.")
    parser.add_argument("--stop", action="store_true", help="stop the background deployment")
    parser.add_argument("--status", action="store_true", help="show whether the services are up")
    parser.add_argument("--logs", "-f", action="store_true", help="follow the combined log")
    parser.add_argument("--detach", "-d", action="store_true", help="run in the background")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    parser.add_argument("--asr-port", type=int, default=8001, help="ASR service port (default: 8001)")
    parser.add_argument("--pronounce-port", type=int, default=8002, help="pronunciation service and web UI port (default: 8002)")
    parser.add_argument(
        "--model",
        default=default_model,
        help=f"Qwen3-ASR-0.6B, Qwen3-ASR-1.7B, a Hugging Face repo or a local path (default: {default_model})",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu (default: auto)")
    parser.add_argument("--python", default=None, help="interpreter for the services (default: .venv, created if missing)")
    parser.add_argument("--no-auto-setup", action="store_true", help="do not create .venv or run pip install")
    parser.add_argument("--log-dir", type=Path, default=REPO_DIR / "logs", help="log directory (default: ./logs)")
    parser.add_argument("--pid-file", type=Path, default=REPO_DIR / ".serve.pid", help="PID file for --detach (default: ./.serve.pid)")
    parser.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)  # set by --detach

    parsed = parser.parse_args(args)
    if parsed.asr_port == parsed.pronounce_port and not (parsed.stop or parsed.status or parsed.logs):
        parser.error(f"--asr-port and --pronounce-port cannot be identical ({parsed.asr_port})")
    parsed.log_dir = parsed.log_dir.resolve()
    parsed.pid_file = parsed.pid_file.resolve()
    return parsed


def build_commands(args: argparse.Namespace, python_exe: Optional[str] = None) -> tuple[list[str], list[str]]:
    """Return the (asr, pronounce) command lines."""
    exe = python_exe or sys.executable
    asr_cmd = [exe, "-u", "asr_server.py", "--host", args.host, "--port", str(args.asr_port), "--device", args.device, "--model", args.model]
    pronounce_cmd = [exe, "-u", "pronounce_server.py", "--host", args.host, "--port", str(args.pronounce_port), "--device", args.device]
    return asr_cmd, pronounce_cmd


def terminate_process(proc: subprocess.Popen, name: str, timeout: float = 5.0) -> None:
    """SIGTERM the process; SIGKILL it if it hasn't exited after timeout seconds."""
    if proc.poll() is not None:
        return
    logger.info("Stopping %s (PID %d)...", name, proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("%s did not exit within %.1fs; sending SIGKILL.", name, timeout)
        proc.kill()
        proc.wait()


def handle_stop(pid_file: Path) -> int:
    pid = read_pid(pid_file)
    if not pid or not is_pid_running(pid):
        pid_file.unlink(missing_ok=True)
        print("PhonoScore is not running.")
        return 0

    print(f"Stopping PhonoScore (PID {pid})...")
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 6.0
    while is_pid_running(pid) and time.time() < deadline:
        time.sleep(0.2)
    if is_pid_running(pid):
        print(f"Process {pid} did not exit in time; sending SIGKILL.")
        os.kill(pid, signal.SIGKILL)

    pid_file.unlink(missing_ok=True)
    print("Stopped.")
    return 0


def probe_health(url: str, timeout: float = 2.0) -> Optional[dict]:
    """GET a /health URL; return its JSON, or None if it isn't answering."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def local_url(host: str, port: int) -> str:
    """URL for reaching a service bound to host from this machine."""
    return f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"


def handle_status(pid_file: Path, host: str, asr_port: int, pronounce_port: int, log_dir: Path) -> int:
    pid = read_pid(pid_file)
    running = pid is not None and is_pid_running(pid)
    asr = probe_health(f"{local_url(host, asr_port)}/health")
    pronounce = probe_health(f"{local_url(host, pronounce_port)}/health")
    down = "STARTING" if running else "STOPPED"

    print(f"Supervisor:  {f'RUNNING (PID {pid})' if running else 'NOT RUNNING'}")
    if asr:
        print(f"ASR:         UP on port {asr_port} (model: {asr.get('model', 'unknown')}, device: {asr.get('device', 'unknown')})")
    else:
        print(f"ASR:         {down} (port {asr_port})")
    if pronounce:
        print(f"Pronounce:   UP on port {pronounce_port} (device: {pronounce.get('device', 'unknown')})")
        print(f"Web UI:      {local_url(host, pronounce_port)}")
    else:
        print(f"Pronounce:   {down} (port {pronounce_port})")
    if log_dir.exists():
        print(f"Logs:        {log_dir}")
    return 0 if running else 1


def handle_logs(log_file: Path, follow: bool = False, lines_count: int = 50) -> int:
    """Print the last lines of log_file, then keep printing new lines if follow is set."""
    if not log_file.is_file():
        print(f"No log at {log_file}. Start PhonoScore first with 'python serve.py'.")
        return 1

    with open(log_file, encoding="utf-8", errors="replace") as f:
        sys.stdout.writelines(f.readlines()[-lines_count:])
        sys.stdout.flush()
        try:
            while follow:
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    return 0


def spawn_detached(args: argparse.Namespace) -> int:
    """Start run_combined in a new background session and record its PID."""
    existing_pid = read_pid(args.pid_file)
    if existing_pid and is_pid_running(existing_pid):
        print(f"PhonoScore is already running (PID {existing_pid}). Stop it with: python serve.py --stop")
        return 1

    args.log_dir.mkdir(parents=True, exist_ok=True)
    serve_log = args.log_dir / "serve.log"
    cmd = [
        sys.executable, str(REPO_DIR / "serve.py"), "--internal-worker",
        "--host", args.host,
        "--asr-port", str(args.asr_port),
        "--pronounce-port", str(args.pronounce_port),
        "--model", args.model,
        "--device", args.device,
        "--log-dir", str(args.log_dir),
        "--pid-file", str(args.pid_file),
    ]
    if args.python:
        cmd += ["--python", args.python]
    if args.no_auto_setup:
        cmd.append("--no-auto-setup")

    with open(serve_log, "a", encoding="utf-8") as out:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)

    time.sleep(0.6)
    ret = proc.poll()
    if ret is not None:
        print(f"Failed to start (exit code {ret}). See {serve_log}")
        return ret

    args.pid_file.write_text(str(proc.pid))
    print(f"PhonoScore started in the background (PID {proc.pid}).")
    print(f"  Web UI:  {local_url(args.host, args.pronounce_port)} (once --status shows UP)")
    print(f"  Logs:    {serve_log}")
    print("  Status:  python serve.py --status")
    print("  Stop:    python serve.py --stop")
    return 0


def run_combined(args: argparse.Namespace, poll_interval: float = 0.5) -> int:
    """Run both services until interrupted; if one exits, stop the other and return its code."""
    python_exe = resolve_python_executable(args.python, auto_setup=not args.no_auto_setup)
    if not check_dependencies(python_exe):
        logger.error(
            "%s cannot import fastapi/uvicorn. Run without --no-auto-setup, or install manually:\n"
            "  source .venv/bin/activate && pip install -r requirements.txt",
            python_exe,
        )
        return 1

    args.log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting PhonoScore (python: %s)", python_exe)
    logger.info("  ASR:        port %d (model: %s, device: %s)", args.asr_port, args.model, args.device)
    logger.info("  Pronounce:  port %d (device: %s)", args.pronounce_port, args.device)
    logger.info("  Logs:       %s", args.log_dir)

    # In --detach mode stdout already is serve.log, so only write it explicitly in the foreground
    serve_log = None if args.internal_worker else open(args.log_dir / "serve.log", "a", buffering=1, encoding="utf-8")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    services = []  # (name, process, log file)
    for name, cmd in zip(("asr", "pronounce"), build_commands(args, python_exe)):
        log = open(args.log_dir / f"{name}.log", "a", buffering=1, encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, cwd=REPO_DIR)
        threading.Thread(target=forward_output, args=(proc.stdout, name, log, serve_log), daemon=True).start()
        services.append((name, proc, log))

    def stop_on_signal(signum, _frame):
        logger.info("Received %s, stopping services...", signal.Signals(signum).name)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop_on_signal)
    signal.signal(signal.SIGTERM, stop_on_signal)

    health_urls = [f"{local_url(args.host, port)}/health" for port in (args.asr_port, args.pronounce_port)]
    ready = False
    exit_code = 0
    try:
        while True:
            if not ready and all(probe_health(url, timeout=0.5) for url in health_urls):
                ready = True
                logger.info("Ready. Open %s in your browser.", local_url(args.host, args.pronounce_port))
            exited = [(name, code) for name, proc, _ in services if (code := proc.poll()) is not None]
            if exited:
                name, exit_code = exited[0]
                logger.error("%s service exited unexpectedly with code %d", name, exit_code)
                break
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        pass
    finally:
        for name, proc, log in services:
            terminate_process(proc, f"{name} service")
            log.close()
        if serve_log:
            serve_log.close()
        if args.internal_worker:
            args.pid_file.unlink(missing_ok=True)
    return exit_code


def forward_output(pipe, name: str, service_log, serve_log) -> None:
    """Copy a service's output to its own log, and with a [name] prefix to stdout and serve.log."""
    for line in pipe:
        service_log.write(line)
        for out in filter(None, (sys.stdout, serve_log)):
            out.write(f"[{name}] {line}")
            out.flush()


def main(args: Optional[list[str]] = None) -> int:
    cli_args = parse_args(args)
    if cli_args.stop:
        return handle_stop(cli_args.pid_file)
    if cli_args.status:
        return handle_status(cli_args.pid_file, cli_args.host, cli_args.asr_port, cli_args.pronounce_port, cli_args.log_dir)
    if cli_args.logs:
        return handle_logs(cli_args.log_dir / "serve.log", follow=True)
    if cli_args.detach and not cli_args.internal_worker:
        return spawn_detached(cli_args)
    return run_combined(cli_args)


if __name__ == "__main__":
    sys.exit(main())
