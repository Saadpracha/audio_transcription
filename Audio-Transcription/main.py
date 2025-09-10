import whisperx
import json
import os
import gc
import torch
import time
import psutil

# Limit CPU threads to avoid overload
torch.set_num_threads(1)

# Load model once and keep it in memory
model = whisperx.load_model("small", device="cpu", compute_type="int8")

def transcribe_and_save(audio_path, output_path):
    start_time = time.time()

    # Record initial memory
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB

    # Run transcription
    result = model.transcribe(audio_path, language="en")

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    # Cleanup
    del result
    gc.collect()

    # Record end stats
    mem_after = process.memory_info().rss / (1024 * 1024)  # MB
    end_time = time.time()
    duration = end_time - start_time

    # Print stats
    print(f"✅ Transcription saved to {output_path}")
    print("\n--- Stats ---")
    print(f"⏱️ Time taken: {duration:.2f} sec")
    print(f"💾 Memory before: {mem_before:.2f} MB")
    print(f"💾 Memory after:  {mem_after:.2f} MB")

# Example run
transcribe_and_save("audio/22982-02.mp3", "output/22982-02.json")
