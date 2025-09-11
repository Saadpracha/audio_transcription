
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List

import whisperx
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


class AudioTranscriberWithDiarization:
    def __init__(self, model_size="small", device="cpu", compute_type="int8", num_threads=2):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.num_threads = num_threads
        self.model = None
        self.diarization_pipeline = None

        torch.set_num_threads(num_threads)
        load_dotenv()

    def load_models(self):
        # Load WhisperX model
        if self.model is None:
            print(f"[INFO] Loading Whisper model ({self.model_size}) on {self.device}...")
            self.model = whisperx.load_model(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            print("[INFO] Whisper model loaded.")

        # Load pyannote speaker diarization model v3.1
        if self.diarization_pipeline is None:
            print("[INFO] Loading speaker diarization model (v3.1)...")
            token = os.getenv("HUGGINGFACE_TOKEN")
            if not token:
                raise RuntimeError("❌ HUGGINGFACE_TOKEN not set in .env or environment.")
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=token
                )
                print("[INFO] Diarization model loaded.")
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load diarization model: {e}")

    def transcribe_audio(self, audio_path: str, language: str = "en") -> dict:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self.load_models()

        print(f"[INFO] Transcribing: {audio_path}")
        start_time = time.time()

        result = self.model.transcribe(audio_path, language=language, batch_size=16)

        duration = time.time() - start_time
        print(f"[INFO] Transcription completed in {duration:.2f} seconds")
        return result

    def perform_diarization(self, audio_path: str) -> List[Dict]:
        print("[INFO] Performing speaker diarization...")
        start_time = time.time()

        if self.diarization_pipeline is None:
            raise RuntimeError("❌ Diarization model not initialized.")

        with ProgressHook() as hook:
            diarization = self.diarization_pipeline(audio_path, hook=hook)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "duration": turn.end - turn.start
            })

        duration = time.time() - start_time
        print(f"[INFO] Diarization done in {duration:.2f} seconds — {len(segments)} segments.")
        return segments

    def align_transcription_with_speakers(self, transcription: dict, speaker_segments: List[Dict]) -> dict:
        if not speaker_segments:
            return transcription

        speaker_map = {}
        for seg in speaker_segments:
            for t in range(int(seg["start"] * 10), int(seg["end"] * 10) + 1):
                speaker_map[t / 10.0] = seg["speaker"]

        enhanced_segments = []
        for segment in transcription.get("segments", []):
            speakers_in_segment = [
                speaker_map.get(t / 10.0)
                for t in range(int(segment["start"] * 10), int(segment["end"] * 10) + 1)
                if speaker_map.get(t / 10.0)
            ]
            speaker = max(set(speakers_in_segment), key=speakers_in_segment.count) if speakers_in_segment else "Unknown"
            segment["speaker"] = speaker
            enhanced_segments.append(segment)

        transcription["segments"] = enhanced_segments
        return transcription

    def save_transcription(self, result: dict, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved full transcript to: {output_path}")

    def generate_speaker_summary(self, enhanced_transcription: dict) -> Dict[str, List[dict]]:
        speaker_summary = {}
        for segment in enhanced_transcription.get("segments", []):
            speaker = segment.get("speaker", "Unknown")
            if speaker not in speaker_summary:
                speaker_summary[speaker] = []
            speaker_summary[speaker].append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            })
        return speaker_summary

    def cleanup(self):
        del self.model
        del self.diarization_pipeline
        self.model = None
        self.diarization_pipeline = None
        import gc
        gc.collect()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Transcribe audio with speaker diarization (Whisper + Pyannote)")
    parser.add_argument("audio_file", help="Path to input audio file")
    parser.add_argument("-o", "--output", help="Output JSON file path", type=str)
    parser.add_argument("-m", "--model", default="small", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("-l", "--language", default="en", help="Language code (default: en)")
    parser.add_argument("-t", "--threads", default=2, type=int, help="Number of CPU threads")
    parser.add_argument("-d", "--device", choices=["cpu"], default="cpu", help="Device to run on")
    parser.add_argument("--no-diarization", action="store_true", help="Disable speaker diarization")
    parser.add_argument("--summary", action="store_true", help="Generate a per-speaker summary")
    return parser.parse_args()


def generate_output_path(audio_path: str, custom_output: Optional[str]) -> str:
    if custom_output:
        return custom_output
    base = Path(audio_path).stem
    return os.path.join("output", f"{base}_diarized.json")


def main():
    args = parse_arguments()
    output_path = generate_output_path(args.audio_file, args.output)

    print("=" * 60)
    print("🔊 Transcription + Diarization")
    print("=" * 60)
    print(f"📁 Input: {args.audio_file}")
    print(f"📄 Output: {output_path}")
    print(f"🤖 Model: {args.model}")
    print(f"🗣️ Diarization: {'Enabled' if not args.no_diarization else 'Disabled'}")
    print("=" * 60)

    transcriber = AudioTranscriberWithDiarization(
        model_size=args.model,
        device=args.device,
        num_threads=args.threads
    )

    # Transcribe
    transcription = transcriber.transcribe_audio(args.audio_file, args.language)

    # Diarize
    speaker_segments = []
    if not args.no_diarization:
        speaker_segments = transcriber.perform_diarization(args.audio_file)

    # Align speakers
    if speaker_segments:
        transcription = transcriber.align_transcription_with_speakers(transcription, speaker_segments)

    # Add metadata
    transcription["metadata"] = {
        "model": args.model,
        "language": args.language,
        "diarization_enabled": not args.no_diarization,
        "speakers_detected": len(set(seg["speaker"] for seg in speaker_segments)) if speaker_segments else 0
    }

    # Save full transcript
    transcriber.save_transcription(transcription, output_path)

    # Save summary if requested
    if args.summary and speaker_segments:
        summary = transcriber.generate_speaker_summary(transcription)
        summary_path = output_path.replace(".json", "_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved speaker summary to: {summary_path}")

    transcriber.cleanup()
    print("\n✅ All done!!")


if __name__ == "__main__":
    main()
