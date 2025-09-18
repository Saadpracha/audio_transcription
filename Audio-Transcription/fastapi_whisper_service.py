import os
import uuid
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import subprocess
import shutil

# Import summarizer module to call its function directly (does not execute main block)
try:
    import ai_summary  # type: ignore
except Exception:
    ai_summary = None  # type: ignore

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

# Optional parts
import boto3
import openai
import smtplib
from email.message import EmailMessage

# load .env
load_dotenv()

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcribe_service")

# Try to import user's whisper+diarization class (put your script in whisper_diarizer.py)
try:
    from whisper_diarizer import AudioTranscriberWithDiarization
except Exception:
    # For the simplified flow, we don't require this import. The full flow can use it later.
    AudioTranscriberWithDiarization = None  # type: ignore

# Config
STORAGE = os.getenv("STORAGE", "local")  # 'local' or 's3'
AUDIO_TMP_DIR = Path(os.getenv("AUDIO_TMP_DIR", "./tmp_audio")).resolve()
PERSIST_AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./audio")).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()
AUDIO_TMP_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

S3_BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Email configuration (optional)
EMAIL_SMTP_HOST = os.getenv("EMAIL_SMTP_HOST")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", 587))
EMAIL_SMTP_USER = os.getenv("EMAIL_SMTP_USER")
EMAIL_SMTP_PASS = os.getenv("EMAIL_SMTP_PASS")
EMAIL_FROM = os.getenv("EMAIL_FROM")

# Whisper config defaults
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
NUM_THREADS = int(os.getenv("NUM_THREADS", "2"))

PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "false").lower() in ("1", "true", "yes")

# In-memory job store (replace with DB/Redis for production)
jobs: Dict[str, Dict[str, Any]] = {}

# Initialize S3 client lazily
_s3_client = None

def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=AWS_REGION,
        )
    return _s3_client

# Instantiate a reusable transcriber object (kept for full flow; unused in simple mode)
_transcriber: Optional[AudioTranscriberWithDiarization] = None  # type: ignore

app = FastAPI(title="WhisperX + Diarization Transcription Service")


class WebhookPayload(BaseModel):
    audio: str
    call_id: Optional[str] = None
    language: Optional[str] = "en"
    email_to: Optional[str] = None
    no_diarization: Optional[bool] = False
    # allow additional fields; Pydantic will ignore unknowns unless configured otherwise


@app.on_event("startup")
def startup_event():
    logger.info("Service starting up...")
    # Full flow init is skipped in simple mode


def detect_extension_from_content_type(content_type: str) -> str:
    if not content_type:
        return "mp3"
    content_type = content_type.lower()
    if "mpeg" in content_type or "mp3" in content_type:
        return "mp3"
    if "wav" in content_type:
        return "wav"
    if "mpeg" in content_type:
        return "mp3"
    if "ogg" in content_type or "opus" in content_type:
        return "ogg"
    return "mp3"


def download_audio(url: str, dest_path: Path, auth: Optional[tuple] = None, headers: Optional[dict] = None, timeout: int = 60):
    # Ensure target directory exists
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as mk_ex:
        logger.exception("Failed to ensure tmp dir exists: %s", mk_ex)
    logger.info("Downloading audio from %s to %s", url, dest_path)
    with requests.get(url, stream=True, auth=auth, headers=headers, timeout=timeout) as r:
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        # If dest_path doesn't have extension, try to set
        if dest_path.suffix == "":
            ext = detect_extension_from_content_type(content_type)
            dest_path = dest_path.with_suffix("." + ext)
            # Ensure directory remains ensured after suffix change
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    logger.info("Downloaded audio to %s", dest_path)
    return dest_path


def derive_readable_audio_name(url: str, fallback: str) -> str:
    """Derive a readable filename from the URL path parts, fallback to provided name."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        if parts:
            # take last 2 parts if available (e.g., Recordings/RE...). Join with '_'.
            tail = parts[-2:]
            name = "_".join(tail)
            # ensure has extension
            if not os.path.splitext(name)[1]:
                name += ".wav"
            return name
    except Exception:
        pass
    return fallback


def upload_to_s3(local_path: Path, s3_key: str) -> str:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set in env")
    s3 = get_s3_client()
    logger.info("Uploading %s to s3://%s/%s", local_path, S3_BUCKET, s3_key)
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    # create a presigned url valid for 30 days
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=3600 * 24 * 30,
    )
    logger.info("Uploaded to S3 and generated presigned URL")
    return url


def summarize_transcript_with_openai(transcription: dict) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # Build a readable transcript (speaker + text + time)
    lines = []
    for seg in transcription.get("segments", []):
        start = seg.get("start")
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        lines.append(f"[{start:.2f}] {speaker}: {text}")

    # join; if extremely long, we could chunk - for simplicity we send everything (you may want to chunk in production)
    joined = "\n".join(lines)

    system_prompt = (
        "You are an assistant that creates concise call summaries. Given the transcript below, output:\n"
        "1) A short TL;DR (1-2 lines)\n2) Key points / decisions (bullet list)\n3) Action items with owners (if mentioned)\n4) Important quotes or timestamps\n        "
    )

    user_prompt = "Transcript:\n\n" + joined

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.info("Sending transcript to OpenAI for summarization (model=%s)", OPENAI_MODEL)
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=1500,
    )
    summary_text = resp["choices"][0]["message"]["content"].strip()
    return summary_text


def send_email_with_attachment(to_email: str, subject: str, body: str, attachment_path: Optional[Path] = None):
    if not EMAIL_SMTP_HOST:
        raise RuntimeError("EMAIL_SMTP_HOST not configured")
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM or EMAIL_SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    if attachment_path:
        with open(attachment_path, "rb") as f:
            data = f.read()
        # guess maintype by ext; we attach as octet-stream for simplicity
        msg.add_attachment(data, maintype="application", subtype="octet-stream", filename=attachment_path.name)

    logger.info("Sending email to %s via %s:%s", to_email, EMAIL_SMTP_HOST, EMAIL_SMTP_PORT)
    s = smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, timeout=30)
    s.starttls()
    if EMAIL_SMTP_USER and EMAIL_SMTP_PASS:
        s.login(EMAIL_SMTP_USER, EMAIL_SMTP_PASS)
    s.send_message(msg)
    s.quit()
    logger.info("Email sent")


def process_job(job_id: str, payload: dict):
    """Simplified pipeline: download audio, run CLI transcriber, done."""
    logger.info("Job %s started (simple mode)", job_id)
    jobs[job_id] = {"status": "started", "created_at": time.time(), "payload": payload}

    try:
        audio_url = payload.get("audio")
        if not audio_url:
            raise ValueError("payload must include 'audio' url")
        call_id = payload.get("call_id") or str(uuid.uuid4())

        jobs[job_id]["status"] = "downloading"
        # build a readable persisted audio name and a temp download target
        readable_name = derive_readable_audio_name(audio_url, f"{call_id}_{uuid.uuid4().hex}.wav")
        persisted_audio_path = PERSIST_AUDIO_DIR / readable_name
        temp_download_name = f"{call_id}_{uuid.uuid4().hex}"
        local_path = AUDIO_TMP_DIR / temp_download_name
        # support basic auth via audio_auth or headers
        audio_auth = None
        if isinstance(payload.get("audio_auth"), dict):
            aa = payload.get("audio_auth")
            audio_auth = (aa.get("username"), aa.get("password"))
        headers = payload.get("audio_headers")
        local_path = download_audio(audio_url, local_path, auth=audio_auth, headers=headers)

        # copy/move downloaded file to persisted audio folder with readable name
        try:
            persisted_audio_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_path, persisted_audio_path)
            jobs[job_id]["audio_path"] = str(persisted_audio_path)
        except Exception as cp_ex:
            logger.exception("Failed to persist audio copy: %s", cp_ex)
            jobs[job_id]["audio_path"] = str(local_path)

        # Run your CLI transcriber script
        jobs[job_id]["status"] = "transcribing"
        # per-run output directory derived from audio reference
        run_dir_name = Path(readable_name).stem
        run_output_dir = OUTPUT_DIR / run_dir_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        transcript_filename = f"{call_id}_diarized.json"
        transcript_path = run_output_dir / transcript_filename
        script_path = (Path(__file__).parent / "audio_transcribe_diarization.py").resolve()
        cmd = [sys.executable, str(script_path), str(local_path), "-o", str(transcript_path), "-l", payload.get("language", "en")]
        logger.info("Running transcriber: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(Path(__file__).parent))

        # Summarize using ai_summary.process_transcript if available
        try:
            if ai_summary is not None and hasattr(ai_summary, "process_transcript"):
                jobs[job_id]["status"] = "summarizing"
                summary_filename = f"{call_id}_summary.json"
                summary_path = run_output_dir / summary_filename
                ai_summary.process_transcript(str(transcript_path), str(summary_path), OPENAI_API_KEY)
                if summary_path.exists():
                    jobs[job_id]["summary_path"] = str(summary_path)
            else:
                logger.warning("ai_summary.process_transcript not available; skipping summarization")
            jobs[job_id]["output_dir"] = str(run_output_dir)
        except Exception as summarize_ex:
            logger.exception("Summarization step failed: %s", summarize_ex)

        jobs[job_id]["transcript_path"] = str(transcript_path)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["finished_at"] = time.time()
        jobs[job_id]["audio_local_path"] = str(local_path)

        logger.info("Job %s finished successfully (simple mode)", job_id)

    except Exception as e:
        logger.exception("Job %s failed: %s", job_id, e)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.post("/webhook")
def webhook_listener(payload: dict, background_tasks: BackgroundTasks):
    """Receives webhook (from Make or direct CRM). Payload must contain `audio` url. Optionally call_id, email_to.

    Example payload:
    {
      "audio": "https://api.twilio.com/.../Recordings/RExxx",
      "call_id": "12345",
      "email_to": "sales@example.com"
    }
    """
    logger.info("Received webhook payload: keys=%s", list(payload.keys()))
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "created_at": time.time()}
    # enqueue background job
    background_tasks.add_task(process_job, job_id, payload)
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    return job


@app.get("/health")
def health():
    return {"status": "ok", "models_preloaded": PRELOAD_MODELS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_whisper_service:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
