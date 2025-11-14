"""
Step 4: Parse Transcripts to CSV

Extracts structured data from combined transcripts using GPT-5.1, generating
weekly CSV files (YYYY-MM-DD.csv) with standardized fields.

Usage:
    python 4-parse-transcripts-to-csv.py

Or via Makefile:
    make parse

Prerequisites:
    - OpenAI API key set as OPENAI_API_KEY environment variable
    - Combined transcript files (YYYY-MM-DD.txt from step 3) in weekly folders
    - Prompt file: 4-parse-transcripts-to-csv-prompt.txt

Output Fields:
    - reporter_name: Name or pseudonym of the listener
    - reporter_location_city, reporter_location_country: Listener's location
    - reporter_location_latitude, reporter_location_longitude: Geographic coordinates
    - observation_text: Description of the reception
    - station_location_city, station_location_country: Identified station location
    - station_location_latitude, station_location_longitude: Station coordinates
    - full_text: Complete original report text

Important Behavior:
    - Skips folders where a CSV file already exists
    - You can safely re-run without overwriting existing parsed data
    - Uses parallel processing (4 workers) for efficiency

Output:
    Creates YYYY-MM-DD.csv in each week's folder containing parsed report data.
"""

import openai
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

client = openai.OpenAI()

# --- Load prompt ---
script_dir = Path(__file__).parent
prompt_path = script_dir / "4-parse-transcripts-to-csv-prompt.txt"
with open(prompt_path, "r", encoding="utf-8") as f:
    PARSE_PROMPT = f.read()

# --- Parse one transcript file into a CSV ---
def parse_transcript_to_csv(transcript_path: Path, output_csv: Path):
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    print(f"📄 {transcript_path.parent.name}: Parsing transcript...")

    response = client.chat.completions.create(
        model="gpt-5.1-2025-11-13",
        messages=[
            {"role": "system", "content": PARSE_PROMPT},
            {"role": "user", "content": transcript.strip()}
        ],
        max_completion_tokens=16384,
        temperature=0,
    )

    csv_text = response.choices[0].message.content.strip()

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        f.write(csv_text)

    print(f"✅ {transcript_path.parent.name}: Saved to {output_csv.name}")

# --- Process a single folder (for parallel execution) ---
def process_one_folder(folder_path: Path):
    try:
        input_file = folder_path / f"{folder_path.name}.txt"
        output_file = folder_path / f"{folder_path.name}.csv"

        if not input_file.exists():
            print(f"⚠️ {folder_path.name}: Skipping — no transcript found.")
            return

        if output_file.exists():
            print(f"⏭️ {folder_path.name}: Skipping — CSV already exists.")
            return

        parse_transcript_to_csv(input_file, output_file)
    except Exception as e:
        print(f"❌ {folder_path.name}: Error - {e}")

# --- Process all folders in parallel ---
def process_all_folders(base_dir: Path, max_workers=4):
    week_folders = [f for f in sorted(base_dir.iterdir()) if f.is_dir()]
    print(f"🧵 Starting parallel processing with {max_workers} workers...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one_folder, folder): folder for folder in week_folders}
        for future in as_completed(futures):
            folder = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"❌ {folder.name}: Exception during processing - {e}")

    elapsed = time.time() - start_time
    print(f"\n✅ All done in {elapsed:.1f} seconds.")

# --- Entry point ---
if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    base_dir = project_root / "archives"
    print(f"📍 Working in directory: {base_dir}")
    process_all_folders(base_dir, max_workers=4)
