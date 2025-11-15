"""
Step 4: Parse Transcripts to CSV

Extracts structured data from combined transcripts using GPT-4o, generating
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
import re
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

# --- Manual parsing fallback (no LLM) ---
def manual_parse_transcript_to_csv(transcript_path: Path) -> str:
    """
    Manual fallback parser that extracts basic structure from transcripts.
    This is a simpler parser that doesn't do geocoding or complex location extraction.
    """
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()
    
    print(f"🔧 {transcript_path.parent.name}: Using manual parser fallback...")
    
    lines = [line.strip() for line in transcript.strip().splitlines() if line.strip()]
    csv_rows = []
    
    # CSV header
    header = '"reporter_name","reporter_location_city","reporter_location_country","reporter_location_latitude","reporter_location_longitude","observation_text","station_location_city","station_location_country","station_location_latitude","station_location_longitude","full_text"'
    csv_rows.append(header)
    
    def escape_csv_field(text: str) -> str:
        """Escape a CSV field by wrapping in quotes and doubling internal quotes."""
        if not text:
            return '""'
        # Replace " with ""
        text = text.replace('"', '""')
        return f'"{text}"'
    
    for line in lines:
        if not line or ':' not in line:
            continue
        
        # Split reporter(s) from observations
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
        
        reporter_part = parts[0].strip()
        observation_part = parts[1].strip()
        
        # Extract reporter name(s) and location(s)
        # Format: "Name (Location)" or "Name1, Name2 (Location)" or just "Name"
        reporter_location = ""
        
        # Check if there's a location in parentheses at the end
        location_match = re.search(r'\(([^)]+)\)\s*$', reporter_part)
        if location_match:
            reporter_location = location_match.group(1)
            reporter_part = reporter_part[:location_match.start()].strip()
        
        # Split multiple reporters by comma
        reporter_names = [name.strip() for name in reporter_part.split(',') if name.strip()]
        
        if not reporter_names:
            continue
        
        # Split observations - look for semicolons, (a)/(b) markers, or numbered items
        observations = []
        
        # First, try splitting by semicolon
        if ';' in observation_part:
            obs_parts = [obs.strip() for obs in observation_part.split(';') if obs.strip()]
            observations.extend(obs_parts)
        else:
            # Check for (a), (b), (1), (2) patterns
            obs_pattern = r'\([a-z0-9]+\)[^)]*'
            matches = list(re.finditer(obs_pattern, observation_part))
            if matches:
                # Extract numbered/lettered observations
                start = 0
                for match in matches:
                    if match.start() > start:
                        # Text before this marker
                        prev_text = observation_part[start:match.start()].strip()
                        if prev_text:
                            observations.append(prev_text)
                    observations.append(match.group(0))
                    start = match.end()
                # Remaining text after last marker
                remaining = observation_part[start:].strip()
                if remaining:
                    observations.append(remaining)
            else:
                # Single observation
                observations.append(observation_part)
        
        # Create CSV rows: one per reporter-observation pair
        for reporter_name in reporter_names:
            for observation in observations:
                # Clean up observation text
                obs_text = observation.strip()
                
                # Try to extract station location from observation (basic pattern matching)
                station_city = ""
                station_country = ""
                
                # Look for common patterns like "City (Country)" or "City, Country" or just "City"
                # This is very basic - the LLM does this much better
                # Try to find city names (capitalized words that might be cities)
                city_patterns = [
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\(([^)]+)\)',  # City (Country)
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),',  # City,
                ]
                
                for pattern in city_patterns:
                    match = re.search(pattern, obs_text)
                    if match:
                        potential_city = match.group(1)
                        # Skip if it's a common word that's not a city
                        if potential_city.lower() not in ['cannot', 'probably', 'relay', 'relaying', 'yes', 'no', 'wavelength', 'wrong', 'details']:
                            station_city = potential_city
                            if len(match.groups()) > 1:
                                station_country = match.group(2)
                            break
                
                # Build CSV row
                csv_row = [
                    escape_csv_field(reporter_name),
                    escape_csv_field(reporter_location if reporter_location else ""),
                    escape_csv_field(""),  # reporter_location_country - would need geocoding
                    escape_csv_field(""),  # reporter_location_latitude
                    escape_csv_field(""),  # reporter_location_longitude
                    escape_csv_field(obs_text),
                    escape_csv_field(station_city),
                    escape_csv_field(station_country),
                    escape_csv_field(""),  # station_location_latitude
                    escape_csv_field(""),  # station_location_longitude
                    escape_csv_field(line)  # full_text
                ]
                
                csv_rows.append(','.join(csv_row))
    
    csv_text = '\n'.join(csv_rows)
    
    # Count rows
    data_rows = len(csv_rows) - 1  # Exclude header
    print(f"   📊 Manual parser extracted {data_rows} rows")
    
    return csv_text

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
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": transcript.strip()}
            ],
            max_completion_tokens=16384,
            temperature=0,
        )
        csv_text = response.choices[0].message.content.strip()
    except (openai.APIError, openai.RateLimitError) as e:
        print(f"❌ {transcript_path.parent.name}: OpenAI API Error - {e}")
        print(f"   Falling back to manual parser...")
        csv_text = manual_parse_transcript_to_csv(transcript_path)
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            f.write(csv_text)
        print(f"✅ {transcript_path.parent.name}: Saved manual parse to {output_csv.name}")
        return
    except Exception as e:
        print(f"❌ {transcript_path.parent.name}: Unexpected error during API call - {type(e).__name__}: {e}")
        print(f"   Falling back to manual parser...")
        csv_text = manual_parse_transcript_to_csv(transcript_path)
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            f.write(csv_text)
        print(f"✅ {transcript_path.parent.name}: Saved manual parse to {output_csv.name}")
        return
    
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
            # After retry failed, use manual fallback
            print(f"⚠️  {transcript_path.parent.name}: CSV still blank after retry ({csv_lines} lines, {csv_chars} chars) - using manual parser fallback...")
            csv_text = manual_parse_transcript_to_csv(transcript_path)
            with open(output_csv, "w", encoding="utf-8", newline="") as f:
                f.write(csv_text)
            print(f"✅ {transcript_path.parent.name}: Saved manual parse to {output_csv.name}")
            return
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
