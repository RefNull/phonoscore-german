"""Qwen3-ASR speech-to-text service with an OpenAI-compatible /v1/audio/transcriptions endpoint.

Uses AutoProcessor + Qwen3ASRForConditionalGeneration directly. Neither AutoModelForSpeechSeq2Seq
nor the ASR pipeline works here: the pipeline cannot force a language for this model.
"""

import argparse
import asyncio
import logging
import os
import threading
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch

logger = logging.getLogger("asr_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

MODEL_PRESETS = {
    "0.6b": "Qwen/Qwen3-ASR-0.6B-hf",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B-hf",
    "1.7b": "Qwen/Qwen3-ASR-1.7B-hf",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B-hf",
}


def resolve_model_id(model_name: str) -> str:
    """Map a preset alias to its Hugging Face repo; pass anything else through."""
    stripped = model_name.strip()
    return MODEL_PRESETS.get(stripped.lower(), stripped)


MODEL_ID = resolve_model_id(os.environ.get("PHONOSCORE_ASR_MODEL", "Qwen3-ASR-0.6B"))
_transcriber = None
_load_lock = threading.Lock()


def resolve_device_and_dtype(device_name: str = "auto") -> tuple[str, torch.dtype]:
    """Pick device and dtype; "auto" tries cuda, then mps, then cpu."""
    dev = device_name.lower().strip()
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return dev, torch.bfloat16 if dev.startswith(("cuda", "mps")) else torch.float32


device, dtype = resolve_device_and_dtype()


def get_transcriber():
    """Load Qwen3-ASR on first use; return a callable (audio_bytes, language) -> text."""
    global _transcriber
    with _load_lock:
        if _transcriber is not None:
            return _transcriber
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
        from transformers.pipelines.audio_utils import ffmpeg_read

        logger.info("Loading Qwen3-ASR model '%s' on %s (%s)...", MODEL_ID, device, dtype)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
        sampling_rate = processor.feature_extractor.sampling_rate

        def run(audio_bytes: bytes, language: Optional[str]) -> str:
            audio = ffmpeg_read(audio_bytes, sampling_rate)
            inputs = processor.apply_transcription_request(audio, language=language).to(device, dtype)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=448)
            new_tokens = output[0, inputs["input_ids"].shape[1] :]
            return processor.decode(new_tokens, return_format="transcription_only")

        _transcriber = run
        return _transcriber


app = FastAPI(title="PhonoScore ASR", description="German speech transcription (Qwen3-ASR)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "qwen3-asr", "model": MODEL_ID, "device": device}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form("de"),
    model: Optional[str] = Form(None),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    language = language.strip() if language and language.strip() else None
    try:
        # Both run in a worker thread: the first call downloads and loads the model
        run = await asyncio.to_thread(get_transcriber)
        return {"text": await asyncio.to_thread(run, content, language)}
    except Exception as exc:
        logger.error("ASR transcription error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}")


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    default_model = os.environ.get("PHONOSCORE_ASR_MODEL", "Qwen3-ASR-0.6B")
    parser = argparse.ArgumentParser(description="PhonoScore ASR service (Qwen3-ASR)")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="port (default: 8001)")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu (default: auto)")
    parser.add_argument("--model", default=default_model, help=f"preset, Hugging Face repo or local path (default: {default_model})")
    parsed = parser.parse_args(args)
    parsed.model_id = resolve_model_id(parsed.model)
    return parsed


if __name__ == "__main__":
    import uvicorn

    cli_args = parse_args()
    device, dtype = resolve_device_and_dtype(cli_args.device)
    MODEL_ID = cli_args.model_id
    uvicorn.run(app, host=cli_args.host, port=cli_args.port)
