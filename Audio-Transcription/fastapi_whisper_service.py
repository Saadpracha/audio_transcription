import os
import uuid
import json
import time
import logging
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import subprocess
import shutil
from collections import deque

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


# Whisper config defaults
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
NUM_THREADS = int(os.getenv("NUM_THREADS", "2"))

PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "false").lower() in ("1", "true", "yes")

# In-memory job store (replace with DB/Redis for production)
jobs: Dict[str, Dict[str, Any]] = {}

# Job queue management
job_queue = deque()
queue_lock = threading.Lock()
is_processing = False

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


def add_job_to_queue(job_id: str, payload: dict):
    """Add a job to the processing queue."""
    with queue_lock:
        job_queue.append((job_id, payload))
        jobs[job_id]["queue_position"] = len(job_queue)
        logger.info("Job %s added to queue at position %d", job_id, len(job_queue))


def process_queue():
    """Process jobs from the queue one at a time."""
    global is_processing
    
    while True:
        with queue_lock:
            if not job_queue:
                is_processing = False
                break
            job_id, payload = job_queue.popleft()
            is_processing = True
        
        # Update queue positions for remaining jobs
        with queue_lock:
            for i, (queued_job_id, _) in enumerate(job_queue):
                if queued_job_id in jobs:
                    jobs[queued_job_id]["queue_position"] = i + 1
        
        logger.info("Processing job %s from queue", job_id)
        process_job(job_id, payload)
        
        # Small delay before processing next job
        time.sleep(1)


def start_queue_processor():
    """Start the queue processor in a background thread."""
    def queue_worker():
        while True:
            try:
                process_queue()
                time.sleep(5)  # Check for new jobs every 5 seconds
            except Exception as e:
                logger.exception("Error in queue processor: %s", e)
                time.sleep(10)  # Wait longer on error
    
    thread = threading.Thread(target=queue_worker, daemon=True)
    thread.start()
    logger.info("Queue processor started")

# Instantiate a reusable transcriber object (kept for full flow; unused in simple mode)
_transcriber: Optional[AudioTranscriberWithDiarization] = None  # type: ignore

app = FastAPI(title="WhisperX + Diarization Transcription Service")


class WebhookPayload(BaseModel):
    audio: str
    call_id: Optional[str] = None
    language: Optional[str] = "en"
    no_diarization: Optional[bool] = False
    # allow additional fields; Pydantic will ignore unknowns unless configured otherwise


@app.on_event("startup")
def startup_event():
    logger.info("Service starting up...")
    # Start the queue processor
    start_queue_processor()
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
    
    # Upload file with public read access
    s3.upload_file(
        str(local_path), 
        S3_BUCKET, 
        s3_key,
        ExtraArgs={
            'ACL': 'public-read',
            'ContentType': 'application/octet-stream'
        }
    )
    
    # Generate direct public URL (not presigned)
    region = AWS_REGION or 'us-east-1'
    if region == 'us-east-1':
        url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
    else:
        url = f"https://{S3_BUCKET}.s3.{region}.amazonaws.com/{s3_key}"
    
    logger.info("Uploaded to S3 with public access: %s", url)
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


def process_job(job_id: str, payload: dict):
    """Simplified pipeline: download audio, run CLI transcriber, upload to S3."""
    logger.info("Job %s started (simple mode)", job_id)
    jobs[job_id] = {"status": "started", "created_at": time.time(), "payload": payload}

    try:
        audio_url = payload.get("audio")
        if not audio_url:
            raise ValueError("payload must include 'audio' url")
        call_id = payload.get("call_id") or str(uuid.uuid4())

        jobs[job_id]["status"] = "downloading"
        # build a readable persisted audio name and a temp download target
        # Add timestamp to ensure unique file names
        timestamp = int(time.time())
        readable_name = derive_readable_audio_name(audio_url, f"{call_id}_{timestamp}_{uuid.uuid4().hex[:8]}.wav")
        temp_download_name = f"{call_id}_{timestamp}_{uuid.uuid4().hex[:8]}"
        local_path = AUDIO_TMP_DIR / temp_download_name
        # support basic auth via audio_auth or headers
        audio_auth = None
        if isinstance(payload.get("audio_auth"), dict):
            aa = payload.get("audio_auth")
            audio_auth = (aa.get("username"), aa.get("password"))
        headers = payload.get("audio_headers")
        local_path = download_audio(audio_url, local_path, auth=audio_auth, headers=headers)

        # Run your CLI transcriber script
        jobs[job_id]["status"] = "transcribing"
        # Create temporary local output directory for processing
        run_dir_name = Path(readable_name).stem
        run_output_dir = OUTPUT_DIR / run_dir_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        transcript_filename = f"{call_id}_{timestamp}_diarized.json"
        transcript_path = run_output_dir / transcript_filename
        script_path = (Path(__file__).parent / "audio_transcribe_diarization.py").resolve()
        cmd = [sys.executable, str(script_path), str(local_path), "-o", str(transcript_path), "-l", payload.get("language", "en")]
        logger.info("Running transcriber: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(Path(__file__).parent))

        # Summarize using ai_summary.process_transcript if available
        summary_path = None
        call_to_action_path = None
        try:
            if ai_summary is not None and hasattr(ai_summary, "process_transcript"):
                jobs[job_id]["status"] = "summarizing"
                summary_filename = f"{call_id}_{timestamp}_summary.json"
                summary_path = run_output_dir / summary_filename
                ai_summary.process_transcript(str(transcript_path), str(summary_path), OPENAI_API_KEY)
                
                # Check if call-to-action file was created
                call_to_action_filename = f"{call_id}_{timestamp}_call_to_action.json"
                call_to_action_path = run_output_dir / call_to_action_filename
                if not call_to_action_path.exists():
                    call_to_action_path = None
            else:
                logger.warning("ai_summary.process_transcript not available; skipping summarization")
        except Exception as summarize_ex:
            logger.exception("Summarization step failed: %s", summarize_ex)

        # Upload files to S3
        jobs[job_id]["status"] = "uploading_to_s3"
        s3_base_key = f"audio/{call_id}/"
        
        # Upload audio file
        audio_s3_key = f"{s3_base_key}{readable_name}"
        audio_s3_url = upload_to_s3(local_path, audio_s3_key)
        jobs[job_id]["audio_s3_url"] = audio_s3_url
        jobs[job_id]["audio_s3_key"] = audio_s3_key

        # Upload transcription file
        transcript_s3_key = f"{s3_base_key}{transcript_filename}"
        transcript_s3_url = upload_to_s3(transcript_path, transcript_s3_key)
        jobs[job_id]["transcript_s3_url"] = transcript_s3_url
        jobs[job_id]["transcript_s3_key"] = transcript_s3_key

        # Upload summary file if it exists
        if summary_path and summary_path.exists():
            summary_s3_key = f"{s3_base_key}{summary_path.name}"
            summary_s3_url = upload_to_s3(summary_path, summary_s3_key)
            jobs[job_id]["summary_s3_url"] = summary_s3_url
            jobs[job_id]["summary_s3_key"] = summary_s3_key

        # Upload call-to-action file if it exists
        if call_to_action_path and call_to_action_path.exists():
            call_to_action_s3_key = f"{s3_base_key}{call_to_action_path.name}"
            call_to_action_s3_url = upload_to_s3(call_to_action_path, call_to_action_s3_key)
            jobs[job_id]["call_to_action_s3_url"] = call_to_action_s3_url
            jobs[job_id]["call_to_action_s3_key"] = call_to_action_s3_key

        # Clean up local temporary files
        try:
            if local_path.exists():
                local_path.unlink()
            if run_output_dir.exists():
                shutil.rmtree(run_output_dir)
            logger.info("Cleaned up local temporary files")
        except Exception as cleanup_ex:
            logger.warning("Failed to clean up local files: %s", cleanup_ex)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["finished_at"] = time.time()
        jobs[job_id]["s3_base_key"] = s3_base_key

        logger.info("Job %s finished successfully (simple mode)", job_id)

    except Exception as e:
        logger.exception("Job %s failed: %s", job_id, e)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.post("/webhook")
def webhook_listener(payload: dict):
    """Receives webhook (from Make or direct CRM). Payload must contain `audio` url. Optionally call_id.

    Example payload:
    {
      "audio": "https://api.twilio.com/.../Recordings/RExxx",
      "call_id": "12345"
    }
    
    Returns:
    {
      "job_id": "uuid",
      "status_url": "/jobs/{job_id}",
      "call_id": "12345",
      "status": "queued",
      "queue_position": 1
    }
    """
    logger.info("Received webhook payload: keys=%s", list(payload.keys()))
    job_id = str(uuid.uuid4())
    call_id = payload.get("call_id") or str(uuid.uuid4())
    
    # Initialize job status
    jobs[job_id] = {
        "status": "queued", 
        "created_at": time.time(), 
        "call_id": call_id,
        "queue_position": 0
    }
    
    # Add to queue
    add_job_to_queue(job_id, payload)
    
    # Get current queue position
    current_position = jobs[job_id].get("queue_position", 0)
    
    return {
        "job_id": job_id, 
        "status_url": f"/jobs/{job_id}",
        "call_id": call_id,
        "status": "queued",
        "queue_position": current_position
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Get job status and results. When status is 'done', includes all S3 links.
    
    Response format when completed:
    {
      "job_id": "uuid",
      "call_id": "12345",
      "status": "done",
      "created_at": 1234567890,
      "finished_at": 1234567891,
      "s3_base_key": "audio/12345/",
      "files": {
        "audio": {
          "url": "https://s3.amazonaws.com/bucket/audio/12345/file.wav",
          "key": "audio/12345/file.wav",
          "filename": "12345_1234567890_abc12345.wav"
        },
        "transcription": {
          "url": "https://s3.amazonaws.com/bucket/audio/12345/transcript.json",
          "key": "audio/12345/12345_1234567890_diarized.json",
          "filename": "12345_1234567890_diarized.json"
        },
        "summary": {
          "url": "https://s3.amazonaws.com/bucket/audio/12345/summary.json",
          "key": "audio/12345/12345_1234567890_summary.json",
          "filename": "12345_1234567890_summary.json"
        },
        "call_to_action": {
          "url": "https://s3.amazonaws.com/bucket/audio/12345/call_to_action.json",
          "key": "audio/12345/12345_1234567890_call_to_action.json",
          "filename": "12345_1234567890_call_to_action.json"
        }
      }
    }
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    
    # If job is completed, structure the response with all S3 links
    if job.get("status") == "done":
        response = {
            "job_id": job_id,
            "call_id": job.get("call_id"),
            "status": job.get("status"),
            "created_at": job.get("created_at"),
            "finished_at": job.get("finished_at"),
            "s3_base_key": job.get("s3_base_key"),
            "files": {}
        }
        
        # Add audio file info
        if job.get("audio_s3_url"):
            response["files"]["audio"] = {
                "url": job.get("audio_s3_url"),
                "key": job.get("audio_s3_key"),
                "filename": job.get("audio_s3_key", "").split("/")[-1] if job.get("audio_s3_key") else None
            }
        
        # Add transcription file info
        if job.get("transcript_s3_url"):
            response["files"]["transcription"] = {
                "url": job.get("transcript_s3_url"),
                "key": job.get("transcript_s3_key"),
                "filename": job.get("transcript_s3_key", "").split("/")[-1] if job.get("transcript_s3_key") else None
            }
        
        # Add summary file info (if exists)
        if job.get("summary_s3_url"):
            response["files"]["summary"] = {
                "url": job.get("summary_s3_url"),
                "key": job.get("summary_s3_key"),
                "filename": job.get("summary_s3_key", "").split("/")[-1] if job.get("summary_s3_key") else None
            }
        
        # Add call-to-action file info (if exists)
        if job.get("call_to_action_s3_url"):
            response["files"]["call_to_action"] = {
                "url": job.get("call_to_action_s3_url"),
                "key": job.get("call_to_action_s3_key"),
                "filename": job.get("call_to_action_s3_key", "").split("/")[-1] if job.get("call_to_action_s3_key") else None
            }
        
        return response
    
    # For non-completed jobs, return the basic job info with queue status
    response = dict(job)
    
    # Add queue information
    with queue_lock:
        response["is_processing"] = is_processing
        response["queue_length"] = len(job_queue)
        if "queue_position" in job:
            response["queue_position"] = job["queue_position"]
    
    return response


@app.get("/queue/status")
def get_queue_status():
    """Get current queue status and statistics."""
    with queue_lock:
        return {
            "is_processing": is_processing,
            "queue_length": len(job_queue),
            "total_jobs": len(jobs),
            "completed_jobs": len([j for j in jobs.values() if j.get("status") == "done"]),
            "failed_jobs": len([j for j in jobs.values() if j.get("status") == "error"]),
            "queued_jobs": len([j for j in jobs.values() if j.get("status") == "queued"])
        }


@app.get("/health")
def health():
    return {"status": "ok", "models_preloaded": PRELOAD_MODELS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_whisper_service:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
