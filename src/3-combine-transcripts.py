"""
Step 3: Combine Transcript Snippets

Merges multiple OCR snippet files from a single week's column into a single
consolidated transcript file (YYYY-MM-DD.txt).

Usage:
    python 3-combine-transcripts.py                         # Process folders with changed inputs (default)
    python 3-combine-transcripts.py 1925-11-20             # Process single folder if inputs changed
    python 3-combine-transcripts.py --force                 # Force reprocess all folders
    python 3-combine-transcripts.py --ignore-existing       # Process only folders without output (legacy)
    python 3-combine-transcripts.py 1925-11-20 --force     # Force reprocess single folder

Or via Makefile:
    make combine                                             # Process folders with changed inputs

Prerequisites:
    - OpenAI API key set as OPENAI_API_KEY environment variable
    - OCR snippet .txt files (from step 2) in weekly folders
    - Prompt file: 3-combine-transcripts-prompt.txt

Behavior:
    - Default: Only processes folders where input snippet files are newer than the output file
    - If no folder name provided: processes all weekly folders in archives/
    - If folder name provided: processes only that folder
    - With --force: reprocesses all folders regardless of modification times
    - With --ignore-existing: skips folders that already have YYYY-MM-DD.txt output file (legacy behavior)
    - Looks for snippet files (all .txt files except YYYY-MM-DD.txt)
    - Uses GPT-5.1 to intelligently combine snippets into coherent text
    - Uses parallel processing (2 workers) for efficiency when processing all folders

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

# --- Check if input files have changed since output was created ---
def inputs_have_changed(folder_path: Path, output_file: Path) -> bool:
    """
    Returns True if any input snippet file is newer than the output file.
    Returns True if output file doesn't exist.
    Returns False if output is newer than all inputs.
    """
    if not output_file.exists():
        return True
    
    output_mtime = output_file.stat().st_mtime
    
    # Get all snippet files (exclude the output file itself)
    snippet_files = [
        f for f in folder_path.glob("*.txt") 
        if f.name != output_file.name
    ]
    
    if not snippet_files:
        # No input files, but output exists - don't reprocess
        return False
    
    # Check if any input file is newer than the output
    for snippet_file in snippet_files:
        if snippet_file.stat().st_mtime > output_mtime:
            return True
    
    return False

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
    print(f"⏳ {folder_path.name}: Sending to GPT-5.1...")

    response = client.chat.completions.create(
        model="gpt-5.1-2025-11-13",
        messages=[
            {"role": "system", "content": COMBINE_PROMPT},
            {"role": "user", "content": combined_snippets.strip()}
        ],
        max_completion_tokens=16384,
    )

    print(f"✅ {folder_path.name}: Response received.")
    return response.choices[0].message.content.strip()

# --- Process a single folder (for parallel execution) ---
def process_one_folder(folder_path: Path, ignore_existing=False, force=False):
    try:
        output_file = folder_path / f"{folder_path.name}.txt"
        
        # Legacy behavior: skip if output exists and ignore_existing flag is set
        if ignore_existing and output_file.exists():
            print(f"⏭️  {folder_path.name}: Output file already exists, skipping")
            return
        
        # Default behavior: check if inputs have changed (unless --force is used)
        if not force and not ignore_existing:
            if not inputs_have_changed(folder_path, output_file):
                print(f"⏭️  {folder_path.name}: No changes detected, skipping")
                return

        combined_text = combine_transcripts(folder_path)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(combined_text)

        print(f"💾 {folder_path.name}: Saved to {output_file.name}")
    except Exception as e:
        print(f"❌ {folder_path.name}: Error - {e}")

# --- Process all folders in parallel ---
def process_all_folders(base_dir: Path, max_workers=2, ignore_existing=False, force=False):
    week_folders = [f for f in sorted(base_dir.iterdir()) if f.is_dir()]
    
    # Filter out folders with existing output if legacy ignore_existing flag is set
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
    
    # Filter by change detection if not forcing and not using legacy ignore_existing
    if not force and not ignore_existing:
        filtered_folders = []
        for folder in week_folders:
            output_file = folder / f"{folder.name}.txt"
            if inputs_have_changed(folder, output_file):
                filtered_folders.append(folder)
        
        week_folders = filtered_folders
        
        if not week_folders:
            print("✅ All folders are up to date. Nothing to process.")
            return
    
    print(f"🧵 Starting parallel processing with {max_workers} workers...")
    if ignore_existing:
        print(f"📋 Processing {len(week_folders)} folder(s) without existing output")
    elif force:
        print(f"🔄 Force mode: Processing {len(week_folders)} folder(s)")
    else:
        print(f"📋 Processing {len(week_folders)} folder(s) with changed inputs")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one_folder, folder, ignore_existing, force): folder for folder in week_folders}
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
    force = "--force" in args
    
    # Remove flags from args to get folder name (if provided)
    args = [arg for arg in args if arg not in ["--ignore-existing", "--force"]]
    
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
        if force:
            print(f"🔄 --force flag: will reprocess regardless of modification times")
        elif ignore_existing:
            print(f"🔍 --ignore-existing flag: will skip if output file exists")
        else:
            print(f"🔍 Default: will process only if input files have changed")
        process_one_folder(target_folder, ignore_existing=ignore_existing, force=force)
    else:
        print(f"📍 Working in directory: {base_dir}")
        if force:
            print(f"🔄 --force flag: will reprocess all folders")
        elif ignore_existing:
            print(f"🔍 --ignore-existing flag: will skip folders with existing output")
        else:
            print(f"🔍 Default: will process only folders with changed inputs")
        process_all_folders(base_dir, max_workers=2, ignore_existing=ignore_existing, force=force)