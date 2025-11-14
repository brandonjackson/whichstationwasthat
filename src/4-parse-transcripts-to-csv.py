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

# --- Check if CSV result is blank or has no data ---
def is_csv_blank(csv_text: str) -> bool:
    """
    Check if CSV result is blank or contains no data rows.
    Returns True if CSV is empty, whitespace only, or has only header row(s).
    """
    csv_text = csv_text.strip()
    if not csv_text:
        return True
    
    lines = [line.strip() for line in csv_text.splitlines() if line.strip()]
    if len(lines) <= 1:  # Only header or empty
        return True
    
    # Check if there are any data rows (non-header lines with content)
    # Skip header line(s) and check if any remaining lines have content
    if len(lines) <= 2:  # Header + maybe one empty line
        # Check if second line is actually data or just empty/whitespace
        if len(lines) == 2:
            # If second line is very short or looks like just commas, it's probably blank
            data_line = lines[1]
            if len(data_line) < 10 or data_line.count(',') == data_line.count(',,') or not any(c.isalnum() for c in data_line):
                return True
    
    return False

# --- Parse one transcript file into a CSV ---
def parse_transcript_to_csv(transcript_path: Path, output_csv: Path, retry: bool = False):
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    transcript_lines = len(transcript.strip().splitlines())
    transcript_chars = len(transcript.strip())
    
    if retry:
        print(f"📄 {transcript_path.parent.name}: Retrying parse (transcript: {transcript_lines} lines, {transcript_chars:,} chars)...")
    else:
        print(f"📄 {transcript_path.parent.name}: Parsing transcript ({transcript_lines} lines, {transcript_chars:,} chars)...")

    # Use more forceful prompt on retry
    if retry:
        retry_prompt = PARSE_PROMPT + "\n\n⚠️ IMPORTANT: The previous attempt returned a blank or empty CSV. " \
                                      "You MUST extract and parse the data from the transcript. The transcript contains " \
                                      "DX listening reports that need to be converted to CSV format. Do not return an empty " \
                                      "result. Extract all reports and convert them to the required CSV format with proper " \
                                      "data rows. If the transcript has content, you must extract it."
        system_content = retry_prompt
    else:
        system_content = PARSE_PROMPT

    try:
        response = client.chat.completions.create(
            model="gpt-5.1-2025-11-13",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": transcript.strip()}
            ],
            max_completion_tokens=16384,
            temperature=0,
        )
        csv_text = response.choices[0].message.content.strip()
    except openai.APIError as e:
        print(f"❌ {transcript_path.parent.name}: OpenAI API Error - {e}")
        raise
    except openai.RateLimitError as e:
        print(f"❌ {transcript_path.parent.name}: Rate limit error - {e}")
        raise
    except Exception as e:
        print(f"❌ {transcript_path.parent.name}: Unexpected error during API call - {type(e).__name__}: {e}")
        raise
    
    # Check if the model returned an error message in the content
    if csv_text and (csv_text.lower().startswith('error') or 
                     'i cannot' in csv_text.lower() or 
                     'i am unable' in csv_text.lower() or
                     'sorry' in csv_text.lower() and 'cannot' in csv_text.lower()):
        print(f"⚠️  {transcript_path.parent.name}: Model returned what looks like an error message:")
        print(f"   {csv_text[:300]}...")
        if not retry:
            print(f"   Retrying with stricter prompt...")
            return parse_transcript_to_csv(transcript_path, output_csv, retry=True)
    csv_lines = len(csv_text.splitlines())
    csv_chars = len(csv_text)
    csv_non_empty_lines = [l for l in csv_text.splitlines() if l.strip()]
    csv_data_lines = len(csv_non_empty_lines)
    
    # Debug info
    print(f"   📊 Response: {csv_lines} total lines, {csv_data_lines} non-empty lines, {csv_chars:,} chars")
    if csv_non_empty_lines:
        print(f"   📋 First line: {csv_non_empty_lines[0][:100]}...")
        if len(csv_non_empty_lines) > 1:
            print(f"   📋 Second line: {csv_non_empty_lines[1][:100]}...")
    
    # Check if result is blank
    is_blank = is_csv_blank(csv_text)
    if is_blank:
        print(f"   ⚠️  Detected as blank: {csv_lines} lines, {csv_chars} chars")
    
    if is_blank:
        if not retry:
            print(f"⚠️  {transcript_path.parent.name}: CSV result is blank (has {csv_lines} lines, {csv_chars} chars) - retrying with stricter prompt...")
            # Show first few lines of transcript for debugging
            transcript_preview = '\n'.join(transcript.strip().splitlines()[:5])
            print(f"   📝 Transcript preview (first 5 lines):\n{transcript_preview}")
            return parse_transcript_to_csv(transcript_path, output_csv, retry=True)
        else:
            print(f"⚠️  {transcript_path.parent.name}: CSV still blank after retry ({csv_lines} lines, {csv_chars} chars), but saving anyway")
            # Show what we got
            if csv_text:
                print(f"   📄 CSV content preview (first 200 chars):\n{csv_text[:200]}")
    elif retry:
        print(f"✅ {transcript_path.parent.name}: Retry successful - CSV now has {csv_data_lines} data rows")

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
