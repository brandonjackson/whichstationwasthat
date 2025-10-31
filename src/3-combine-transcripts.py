"""
Step 3: Combine Transcript Snippets

Merges multiple OCR snippet files from a single week's column into a single
consolidated transcript file (YYYY-MM-DD.txt).

Usage:
    python 3-combine-transcripts.py                         # Process all folders
    python 3-combine-transcripts.py 1925-11-20             # Process single folder
    python 3-combine-transcripts.py --ignore-existing       # Process only folders without output
    python 3-combine-transcripts.py 1925-11-20 --ignore-existing  # Process folder if no output exists

Or via Makefile:
    make combine                                             # Process all folders

Prerequisites:
    - OpenAI API key set as OPENAI_API_KEY environment variable
    - OCR snippet .txt files (from step 2) in weekly folders
    - Prompt file: 3-combine-transcripts-prompt.txt

Behavior:
    - If no folder name provided: processes all weekly folders in archives/
    - If folder name provided: processes only that folder
    - With --ignore-existing: skips folders that already have YYYY-MM-DD.txt output file
    - Looks for snippet files (all .txt files except YYYY-MM-DD.txt)
    - Uses GPT-4o to intelligently combine snippets into coherent text
    - Regenerates the combined file from snippets each time it runs (unless --ignore-existing is used)
    - Uses parallel processing (4 workers) for efficiency when processing all folders

Output:
    Creates YYYY-MM-DD.txt in each week's folder containing the combined transcript.
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

client = openai.OpenAI()

# --- Load combination prompt ---
script_dir = Path(__file__).parent
prompt_path = script_dir / "3-combine-transcripts-prompt.txt"
with open(prompt_path, "r", encoding="utf-8") as f:
    COMBINE_PROMPT = f.read()

# --- Estimate token count (~4 characters per token) ---
def estimate_tokens(text: str) -> int:
    return len(text) // 4

# --- Combine all OCR .txt snippets in one folder ---
def combine_transcripts(folder_path: Path) -> str:
    snippet_files = sorted(
        [f for f in folder_path.glob("*.txt") if f.name != f"{folder_path.name}.txt"],
        key=lambda f: f.name
    )

    if not snippet_files:
        raise ValueError(f"No .txt files found in {folder_path}")

    print(f"🧩 {folder_path.name}: Found {len(snippet_files)} snippet(s)")

    # Combine snippets
    combined_snippets = ""
    for f in snippet_files:
        with open(f, "r", encoding="utf-8") as infile:
            content = infile.read().strip()
            combined_snippets += f"\n--- {f.name} ---\n{content}\n"

    full_prompt = f"{COMBINE_PROMPT.strip()}\n\n{combined_snippets.strip()}"
    char_count = len(full_prompt)
    token_est = estimate_tokens(full_prompt)

    print(f"📏 {folder_path.name}: Prompt size = {char_count:,} chars (~{token_est:,} tokens)")
    print(f"⏳ {folder_path.name}: Sending to GPT-4o...")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": COMBINE_PROMPT},
            {"role": "user", "content": combined_snippets.strip()}
        ],
        max_tokens=4096,
    )

    print(f"✅ {folder_path.name}: Response received.")
    return response.choices[0].message.content.strip()

# --- Process a single folder (for parallel execution) ---
def process_one_folder(folder_path: Path, ignore_existing=False):
    try:
        output_file = folder_path / f"{folder_path.name}.txt"
        
        # Skip if output file exists and ignore_existing flag is set
        if ignore_existing and output_file.exists():
            print(f"⏭️  {folder_path.name}: Output file already exists, skipping")
            return

        combined_text = combine_transcripts(folder_path)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(combined_text)

        print(f"💾 {folder_path.name}: Saved to {output_file.name}")
    except Exception as e:
        print(f"❌ {folder_path.name}: Error - {e}")

# --- Process all folders in parallel ---
def process_all_folders(base_dir: Path, max_workers=4, ignore_existing=False):
    week_folders = [f for f in sorted(base_dir.iterdir()) if f.is_dir()]
    
    # Filter out folders with existing output if flag is set
    if ignore_existing:
        filtered_folders = []
        for folder in week_folders:
            output_file = folder / f"{folder.name}.txt"
            if not output_file.exists():
                filtered_folders.append(folder)
            else:
                print(f"⏭️  {folder.name}: Output file already exists, skipping")
        
        week_folders = filtered_folders
        
        if not week_folders:
            print("✅ All folders already have output files. Nothing to process.")
            return
    
    print(f"🧵 Starting parallel processing with {max_workers} workers...")
    if ignore_existing:
        print(f"📋 Processing {len(week_folders)} folder(s) without existing output")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one_folder, folder, ignore_existing): folder for folder in week_folders}
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
    
    # Parse command-line arguments
    args = sys.argv[1:]
    ignore_existing = "--ignore-existing" in args
    
    # Remove flag from args to get folder name (if provided)
    if ignore_existing:
        args = [arg for arg in args if arg != "--ignore-existing"]
    
    # Check if a specific folder was provided as command-line argument
    if len(args) > 0:
        folder_name = args[0]
        target_folder = base_dir / folder_name
        
        if not target_folder.exists():
            print(f"❌ Error: Folder '{folder_name}' not found in {base_dir}")
            sys.exit(1)
        
        if not target_folder.is_dir():
            print(f"❌ Error: '{folder_name}' is not a directory")
            sys.exit(1)
        
        print(f"📍 Processing single folder: {target_folder}")
        if ignore_existing:
            print(f"🔍 --ignore-existing flag: will skip if output file exists")
        process_one_folder(target_folder, ignore_existing=ignore_existing)
    else:
        print(f"📍 Working in directory: {base_dir}")
        if ignore_existing:
            print(f"🔍 --ignore-existing flag: will skip folders with existing output")
        process_all_folders(base_dir, max_workers=4, ignore_existing=ignore_existing)