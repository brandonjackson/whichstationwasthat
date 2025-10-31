"""
Step 3: Combine Transcript Snippets

Merges multiple OCR snippet files from a single week's column into a single
consolidated transcript file (YYYY-MM-DD.txt).

Usage:
    python 3-combine-transcripts.py

Or via Makefile:
    make combine

Prerequisites:
    - OpenAI API key set as OPENAI_API_KEY environment variable
    - OCR snippet .txt files (from step 2) in weekly folders
    - Prompt file: 3-combine-transcripts-prompt.txt

Behavior:
    - Processes all weekly folders in archives/
    - Looks for snippet files (all .txt files except YYYY-MM-DD.txt)
    - Uses GPT-4o to intelligently combine snippets into coherent text
    - Regenerates the combined file from snippets each time it runs
    - Uses parallel processing (4 workers) for efficiency

Output:
    Creates YYYY-MM-DD.txt in each week's folder containing the combined transcript.
"""

import os
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
def process_one_folder(folder_path: Path):
    try:
        output_file = folder_path / f"{folder_path.name}.txt"

        combined_text = combine_transcripts(folder_path)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(combined_text)

        print(f"💾 {folder_path.name}: Saved to {output_file.name}")
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