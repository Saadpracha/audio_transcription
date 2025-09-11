#!/usr/bin/env python3
"""
Optimized Audio Transcription Script
Supports command-line arguments for CPU threads and output file name.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List

import whisperx
import torch
import torchaudio
from pyannote.audio import Pipeline
from dotenv import load_dotenv

class AudioTranscriberWithDiarization:
    def __init__(self, model_size: str = "small", device: str = "cpu", 
                 compute_type: str = "int8", num_threads: int = 2):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.num_threads = num_threads
        self.model = None
        self.diarization_pipeline = None

        torch.set_num_threads(num_threads)
        load_dotenv()  # Load .env file

    def load_models(self) -> None:
        if self.model is None:
            print(f"Loading {self.model_size} Whisper model on {self.device}...")
            self.model = whisperx.load_model(
                self.model_size, 
                device=self.device, 
                compute_type=self.compute_type
            )
            print("Whisper model loaded successfully!")

        if self.diarization_pipeline is None:
            print("Loading speaker diarization model...")
            token = os.getenv("HUGGINGFACE_TOKEN")
            if not token:
                raise RuntimeError("❌ HUGGINGFACE_TOKEN not set in environment or .env file. Visit https://hf.co/settings/tokens to create one and accept the conditions at https://hf.co/pyannote/speaker-diarization.")
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=token
                )
                print("Diarization model loaded successfully!")
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load diarization model: {e}. Ensure you've accepted the model conditions at https://hf.co/pyannote/speaker-diarization and the token has access.")

    def transcribe_audio(self, audio_path: str, language: str = "en") -> dict:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self.load_models()

        print(f"Transcribing: {audio_path}")
        start_time = time.time()

        result = self.model.transcribe(
            audio_path, 
            language=language,
            batch_size=8,
        )

        duration = time.time() - start_time
        print(f"Transcription completed in {duration:.2f} seconds")
        return result

    def perform_diarization(self, audio_path: str) -> List[Dict]:
        print("Performing speaker diarization...")
        start_time = time.time()

        if self.diarization_pipeline is None:
            raise RuntimeError("❌ Diarization pipeline not initialized. Set HUGGINGFACE_TOKEN and ensure access to 'pyannote/speaker-diarization'.")
        try:
            diarization = self.diarization_pipeline(audio_path)
        except Exception as e:
            raise RuntimeError(f"❌ Diarization failed: {e}")

        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "duration": turn.end - turn.start
            })

        duration = time.time() - start_time
        print(f"Diarization completed in {duration:.2f} seconds")
        print(f"Found {len(speaker_segments)} speaker segments")

        return speaker_segments

    def align_transcription_with_speakers(self, transcription: dict, 
                                          speaker_segments: List[Dict]) -> dict:
        if not speaker_segments:
            return transcription

        speaker_map = {}
        for segment in speaker_segments:
            for t in range(int(segment["start"] * 10), int(segment["end"] * 10) + 1):
                speaker_map[t / 10.0] = segment["speaker"]

        enhanced_segments = []
        for segment in transcription.get("segments", []):
            speakers_in_range = [
                speaker_map[t / 10.0]
                for t in range(int(segment["start"] * 10), int(segment["end"] * 10) + 1)
                if t / 10.0 in speaker_map
            ]

            primary_speaker = max(set(speakers_in_range), key=speakers_in_range.count) if speakers_in_range else "Unknown"

            enhanced_segment = segment.copy()
            enhanced_segment["speaker"] = primary_speaker
            enhanced_segments.append(enhanced_segment)

        transcription["segments"] = enhanced_segments
        return transcription

    def save_transcription(self, result: dict, output_path: str) -> None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"✅ Transcription with diarization saved to: {output_path}")

    def generate_speaker_summary(self, enhanced_transcription: dict) -> Dict[str, List[dict]]:
        speaker_summary = {}
        for segment in enhanced_transcription.get("segments", []):
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "").strip()

            if speaker not in speaker_summary:
                speaker_summary[speaker] = []

            if text:
                speaker_summary[speaker].append({
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": text
                })

        return speaker_summary

    def cleanup(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

        if self.diarization_pipeline is not None:
            del self.diarization_pipeline
            self.diarization_pipeline = None

        import gc
        gc.collect()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audio Transcription with Speaker Diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python audio_transcribe_diarization.py input.mp3 -o output.json -t 4
  python audio_transcribe_diarization.py input.wav -o results/transcript.json -t 2 -m base
  python audio_transcribe_diarization.py meeting.wav --no-diarization"""
    )

    parser.add_argument("audio_file", help="Path to audio file to transcribe")
    parser.add_argument("-o", "--output", help="Output file path", type=str)
    parser.add_argument("-t", "--threads", help="Number of CPU threads", type=int, default=2)
    parser.add_argument("-m", "--model", choices=["tiny", "base", "small", "medium", "large"], default="small")
    parser.add_argument("-l", "--language", help="Language code", default="en")
    parser.add_argument("-d", "--device", choices=["cpu"], default="cpu")
    parser.add_argument("--no-diarization", action="store_true", help="Skip speaker diarization")
    parser.add_argument("--summary", action="store_true", help="Generate speaker summary in separate file")
    return parser.parse_args()


def generate_output_path(audio_path: str, custom_output: Optional[str] = None) -> str:
    if custom_output:
        return custom_output

    audio_file = Path(audio_path)
    output_dir = "output"
    output_file = f"{audio_file.stem}_diarized_transcript.json"
    return os.path.join(output_dir, output_file)


def main():
    try:
        args = parse_arguments()

        output_path = generate_output_path(args.audio_file, args.output)

        print("=" * 70)
        print("🎵 Audio Transcription with Speaker Diarization")
        print("=" * 70)
        print(f"📁 Input file: {args.audio_file}")
        print(f"📄 Output file: {output_path}")
        print(f"🧵 CPU threads: {args.threads}")
        print(f"🤖 Model: {args.model}")
        print(f"🌐 Language: {args.language}")
        print(f"💻 Device: {args.device}")
        print(f"👥 Diarization: {'Disabled' if args.no_diarization else 'Enabled'}")
        print("=" * 70)

        transcriber = AudioTranscriberWithDiarization(
            model_size=args.model,
            device=args.device,
            num_threads=args.threads
        )

        transcription_result = transcriber.transcribe_audio(args.audio_file, args.language)

        speaker_segments = []
        if not args.no_diarization:
            speaker_segments = transcriber.perform_diarization(args.audio_file)

        enhanced_result = transcription_result
        if speaker_segments:
            enhanced_result = transcriber.align_transcription_with_speakers(transcription_result, speaker_segments)

        enhanced_result["metadata"] = {
            "audio_file": args.audio_file,
            "model_size": args.model,
            "language": args.language,
            "device": args.device,
            "diarization_enabled": not args.no_diarization,
            "speaker_segments_count": len(speaker_segments),
            "processing_time": time.time()
        }

        transcriber.save_transcription(enhanced_result, output_path)

        if args.summary and speaker_segments:
            summary = transcriber.generate_speaker_summary(enhanced_result)
            summary_path = output_path.replace(".json", "_speaker_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"✅ Speaker summary saved to: {summary_path}")

        transcriber.cleanup()

        print("\n🎉 Transcription with diarization completed successfully!")
        if speaker_segments:
            unique_speakers = set(segment["speaker"] for segment in speaker_segments)
            print(f"👥 Found {len(unique_speakers)} unique speakers: {', '.join(unique_speakers)}")

    except KeyboardInterrupt:
        print("\n⚠️ Transcription interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
