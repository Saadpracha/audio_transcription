import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get the API key securely from an environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OpenAI API key not found. Please set the environment variable.")
    exit()

# Function to load your transcript from a file
def load_transcript(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file at {file_path} does not exist.")
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Function to create the API prompt
def create_prompt(transcript):
    # Support multiple transcript shapes
    if isinstance(transcript, dict) and "segments" in transcript:
        # Expecting list of {'text': ...}
        full_transcript = "\n".join([seg.get('text', '') for seg in transcript['segments']])
    elif isinstance(transcript, dict) and "text" in transcript:
        full_transcript = transcript["text"]
    else:
        # Fallback: stringify the transcript object
        full_transcript = json.dumps(transcript, ensure_ascii=False)

    prompt = f"""
You are a meeting-notes assistant.
Return ONLY JSON matching the schema (summary, call_to_action_items).
Guidelines:
- Adapt detail to transcript length and complexity (brief for short calls; more detailed for long/complex ones).
- Capture outcomes, decisions, commitments, numbers, dates, and owners.
- Infer owners from speaker labels; leave 'owner' empty if unclear. Use "" for missing 'due'.

TRANSCRIPT:
<<< 
{full_transcript}
>>>
"""
    return prompt

# Function to call OpenAI's API and get the summary
def get_summary_from_openai(prompt, api_key):
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # you can also use "gpt-4o"
        messages=[
            {"role": "system", "content": "You are a helpful meeting-notes assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.7,
    )

    # Return the assistant content (defensive)
    try:
        return response.choices[0].message.content.strip()
    except Exception:
        return str(response)

# Helper: strip code fences like ```json ... ```
def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text

# Save parsed JSON and then summary + call_to_action to separate files
def save_outputs(parsed_json: dict, base_output_path: str):
    # Save full parsed JSON
    with open(base_output_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_json, f, indent=4, ensure_ascii=False)
    print(f"Full parsed JSON saved to {base_output_path}")

    root, _ = os.path.splitext(base_output_path)
    summary_path = f"{root}_summary.json"
    cta_path = f"{root}_call_to_action.json"

    # Extract summary and call_to_action keys (simple/flexible)
    summary_val = parsed_json.get("summary", "")
    # try common CTA keys
    cta_val = None
    for k in ("call_to_action", "call_to_action_items", "call_to_actions", "callToAction", "call_to_action_item"):
        if k in parsed_json:
            cta_val = parsed_json[k]
            break

    # Save summary (as JSON wrapper)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({"summary": summary_val}, f, indent=4, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")

    # Save CTA (normalize to empty list if None)
    with open(cta_path, 'w', encoding='utf-8') as f:
        json.dump({"call_to_action": cta_val if cta_val is not None else []}, f, indent=4, ensure_ascii=False)
    print(f"Call to action saved to {cta_path}")

# Save raw model response when JSON parse fails
def save_raw_response(raw_text: str, base_output_path: str):
    root, _ = os.path.splitext(base_output_path)
    raw_path = f"{root}.raw.txt"
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(raw_text)
    print(f"Raw response saved to {raw_path}")

# Main function to process the transcript and generate the summary
def process_transcript(input_path, output_path, api_key):
    transcript = load_transcript(input_path)
    if transcript is None:
        return

    prompt = create_prompt(transcript)
    response_text = get_summary_from_openai(prompt, api_key)
    if not response_text:
        print("No response from OpenAI.")
        return

    cleaned = strip_code_fences(response_text)
    try:
        parsed = json.loads(cleaned)
        save_outputs(parsed, output_path)
    except json.JSONDecodeError as e:
        print(f"Error decoding the response: {e}")
        print("Raw response:", response_text)
        save_raw_response(response_text, output_path)

# Example usage (update paths if needed)
if __name__ == "__main__":
    input_path = r"E:\work\leadgeneration\audio_transcription\Audio-Transcription\output\audio_1_diarization.json"
    output_path = r"E:\work\leadgeneration\audio_transcription\Audio-Transcription\output\summary2.json"
    process_transcript(input_path, output_path, api_key)
