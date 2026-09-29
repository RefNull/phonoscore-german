"""Pronunciation scoring service (OpenPronounce): POST /assess, and the web UI at GET /."""

import argparse
import asyncio
import logging
import os
from pathlib import Path
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import torch

try:
    import openpronounce
except ImportError:
    openpronounce = None

logger = logging.getLogger("pronounce_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def resolve_device(device_name: str = "auto") -> str:
    """Pick a device ("auto" tries cuda, then mps, then cpu) and pass it to OpenPronounce."""
    resolved = (
        ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        if device_name == "auto"
        else device_name.lower().strip()
    )
    os.environ["OPENPRONOUNCE_DEVICE"] = resolved
    return resolved


device = resolve_device()
CLIENT_HTML = Path(__file__).resolve().parent / "client.html"

app = FastAPI(title="PhonoScore", description="German pronunciation scoring (OpenPronounce)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def web_ui():
    return FileResponse(CLIENT_HTML, media_type="text/html")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "openpronounce", "device": device}


@app.post("/assess")
async def assess(
    file: UploadFile = File(...),
    expected_text: str = Form(...),
    lang: str = Form("de"),
):
    """Score the pronunciation in an audio file (.wav or .ogg) against the expected text."""
    if not expected_text or not expected_text.strip():
        raise HTTPException(status_code=422, detail="expected_text must not be empty or whitespace.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    if openpronounce is None:
        raise HTTPException(
            status_code=503,
            detail="Pronunciation assessor is not initialized or model is still loading.",
        )

    suffix = Path(file.filename).suffix.lower() if file.filename else ".wav"
    if suffix not in [".wav", ".ogg"]:
        suffix = ".wav"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(content)
        tmp.flush()
        tmp.close()

        raw_result = await asyncio.to_thread(
            lambda: openpronounce.compare_audio_with_text(
                openpronounce.load_audio(tmp.name),
                expected_text.strip(),
                lang=lang.strip() if lang else "de",
            )
        )

        # openpronounce 0.3.x shape. With the phone recognizer (default), expected/actual
        # are phone lists and confidence lives per phone; on the word-transcription path
        # they are IPA strings and there is no confidence.
        differences = raw_result.get("differences") or {}
        errors = [
            {
                "word": str(err.get("word", "")),
                "expected_ipa": "".join(err.get("expected") or ""),
                "actual_ipa": "".join(err.get("actual") or ""),
                "confidence": float(max((p.get("confidence", 0.0) for p in err.get("phones") or []), default=0.0)),
            }
            for err in differences.get("errors") or []
        ]

        return {
            "score": float(raw_result.get("score") or 0.0),
            "transcription": str(raw_result.get("transcribe") or ""),
            "phoneme_error_rate": float(differences.get("phoneme_error_rate") or 0.0),
            "errors": errors,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Pronunciation assessment error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pronunciation assessment failed: {str(exc)}")
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhonoScore pronunciation service (OpenPronounce)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8002, help="port (default: 8002)")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cuda, mps or cpu (default: auto)",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    import uvicorn

    cli_args = parse_args()
    if cli_args.device != "auto":
        device = resolve_device(cli_args.device)
    uvicorn.run(app, host=cli_args.host, port=cli_args.port)
