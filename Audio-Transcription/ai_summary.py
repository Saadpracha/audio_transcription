import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import re

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
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to create the API prompt
def create_prompt(transcript):
    full_transcript = "\n".join([segment['text'] for segment in transcript['segments']])
    
    prompt = f"""
    You are a meeting-notes assistant.
    Return ONLY JSON matching the schema (summary, action_items).
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

    return response.choices[0].message.content.strip()

# Function to save the summary to a JSON file
def save_summary_to_json(response, output_path):
    try:
        # Strip code fences like ```json ... ```
        cleaned = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", response.strip())

        response_json = json.loads(cleaned)
        with open(output_path, 'w') as json_file:
            json.dump(response_json, json_file, indent=4)
        print(f"Summary saved to {output_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding the response: {e}")
        print("Raw response:", response)

# Main function to process the transcript and generate the summary
def process_transcript(input_path, output_path, api_key):
    # Step 1: Load the transcript
    transcript = load_transcript(input_path)
    if transcript is None:
        return  # Exit if file is not found
    
    # Step 2: Create the prompt with the full transcript
    prompt = create_prompt(transcript)
    
    # Step 3: Get the summary from OpenAI's API
    summary = get_summary_from_openai(prompt, api_key)
    
    # Step 4: Save the summary to a JSON file
    save_summary_to_json(summary, output_path)

# Example usage
input_path = r"E:\work\leadgeneration\audio_transcription\Audio-Transcription\output\audio_1_diarization.json"
output_path = r"E:\work\leadgeneration\audio_transcription\Audio-Transcription\output\summary2.json"

# Run the process
process_transcript(input_path, output_path, api_key)
