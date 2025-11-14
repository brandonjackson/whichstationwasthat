"""
Step 3: Combine Transcript Snippets

Merges multiple OCR snippet files from a single week's column into a single
consolidated transcript file using iterative manual merging.

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
    - Uses iterative manual merging: merges files one by one, finding overlap points using edit distance
    - Uses parallel processing (10 workers) for efficiency when processing all folders
    - Final cleanup with LLM for format consistency

Output:
    Creates combine-pre-llm.txt, combine-report.txt, and YYYY-MM-DD.txt (final cleaned transcript).
"""

import os
import sys
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

client = openai.OpenAI()

# --- Load cleanup prompt ---
script_dir = Path(__file__).parent
prompt_path = script_dir / "3-combine-transcripts-prompt.txt"
with open(prompt_path, "r", encoding="utf-8") as f:
    CLEANUP_PROMPT = f.read()

# --- Normalize string for comparison (remove whitespace and punctuation) ---
def normalize_for_comparison(text: str) -> str:
    """Remove whitespace and punctuation, convert to lowercase."""
    # Remove all whitespace and punctuation, keep only alphanumeric
    normalized = re.sub(r'[^\w]', '', text.lower())
    return normalized

# --- Calculate edit distance between two strings ---
def edit_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

# --- Find matching line in file B for the last line of file A ---
def find_merge_point(file_a_lines: list[str], file_b_lines: list[str]) -> tuple[bool, int, float, str, str, list]:
    """
    Find the merge point between file A and file B.
    Uses multiple strategies to handle OCR variations.
    
    Returns:
        (match_found, match_index, match_strength, file_a_line, file_b_line, closest_matches)
        - match_found: True if a match was found
        - match_index: index in file_b_lines where match was found (-1 if no match)
        - match_strength: similarity score (0-1, higher is better) or -1 if no match
        - file_a_line: the full last line from file A
        - file_b_line: the matching line from file B
        - closest_matches: list of dicts with closest matches for debugging
    """
    if not file_a_lines:
        return (False, -1, -1.0, "", "", [])
    
    last_line_a = file_a_lines[-1].strip()
    
    # Extract reporter name and content after colon
    reporter_a = ""
    content_after_colon_a = ""
    if ":" in last_line_a:
        parts = last_line_a.split(":", 1)
        reporter_a = parts[0].strip()
        content_after_colon_a = parts[1].strip()
    
    normalized_reporter_a = normalize_for_comparison(reporter_a)
    normalized_content_start_a = normalize_for_comparison(content_after_colon_a[:5] if len(content_after_colon_a) >= 5 else content_after_colon_a)
    
    # Strategy 1: Match on first 8 characters (original approach)
    final_line_start_str = last_line_a[:8] if len(last_line_a) >= 8 else last_line_a
    normalized_target_8 = normalize_for_comparison(final_line_start_str)
    
    # Strategy 2: Match on first 12 characters (more context)
    final_line_start_str_12 = last_line_a[:12] if len(last_line_a) >= 12 else last_line_a
    normalized_target_12 = normalize_for_comparison(final_line_start_str_12)
    
    best_match_idx = -1
    best_match_strength = -1.0
    best_match_line = ""
    best_strategy = ""
    closest_matches = []  # Track closest matches for reporting
    
    # Search through file B for a matching line
    for idx, line_b in enumerate(file_b_lines):
        line_b_stripped = line_b.strip()
        if len(line_b_stripped) < 4:  # Minimum length check
            continue
        
        # Extract reporter name and content for line B
        reporter_b = ""
        content_after_colon_b = ""
        if ":" in line_b_stripped:
            parts = line_b_stripped.split(":", 1)
            reporter_b = parts[0].strip()
            content_after_colon_b = parts[1].strip()
        
        normalized_reporter_b = normalize_for_comparison(reporter_b)
        normalized_content_start_b = normalize_for_comparison(content_after_colon_b[:5] if len(content_after_colon_b) >= 5 else content_after_colon_b)
        
        # Strategy 1: First 8 characters
        if normalized_target_8:
            line_b_start_8 = line_b_stripped[:8] if len(line_b_stripped) >= 8 else line_b_stripped
            normalized_candidate_8 = normalize_for_comparison(line_b_start_8)
            if normalized_candidate_8:
                distance = edit_distance(normalized_target_8, normalized_candidate_8)
                max_len = max(len(normalized_target_8), len(normalized_candidate_8))
                if max_len > 0:
                    similarity = 1.0 - (distance / max_len)
                    if similarity > best_match_strength:
                        best_match_idx = idx
                        best_match_strength = similarity
                        best_match_line = line_b_stripped
                        best_strategy = "first_8_chars"
                    # Track for closest matches
                    closest_matches.append({
                        'index': idx,
                        'line': line_b_stripped,
                        'strategy': 'first_8_chars',
                        'similarity': similarity,
                        'reporter_similarity': -1.0,
                        'content_similarity': -1.0
                    })
        
        # Strategy 2: Reporter name + first 5 chars of content (handles OCR variations)
        if normalized_reporter_a and normalized_reporter_b and normalized_content_start_a and normalized_content_start_b:
            # Check reporter name similarity
            reporter_distance = edit_distance(normalized_reporter_a, normalized_reporter_b)
            reporter_max_len = max(len(normalized_reporter_a), len(normalized_reporter_b))
            reporter_similarity = 0.0
            if reporter_max_len > 0:
                reporter_similarity = 1.0 - (reporter_distance / reporter_max_len)
            
            # Check content start similarity (first 5 chars)
            content_distance = edit_distance(normalized_content_start_a, normalized_content_start_b)
            content_max_len = max(len(normalized_content_start_a), len(normalized_content_start_b))
            content_similarity = 0.0
            if content_max_len > 0:
                content_similarity = 1.0 - (content_distance / content_max_len)
            
            # Combined score: both reporter and content must be reasonably similar
            # Reporter similarity weighted more (60%), content (40%)
            combined_similarity = (reporter_similarity * 0.6) + (content_similarity * 0.4)
            
            # Require both to be at least somewhat similar
            if reporter_similarity >= 0.6 and content_similarity >= 0.5:
                if combined_similarity > best_match_strength:
                    best_match_idx = idx
                    best_match_strength = combined_similarity
                    best_match_line = line_b_stripped
                    best_strategy = "reporter_and_content"
            
            # Track for closest matches
            closest_matches.append({
                'index': idx,
                'line': line_b_stripped,
                'strategy': 'reporter_and_content',
                'similarity': combined_similarity,
                'reporter_similarity': reporter_similarity,
                'content_similarity': content_similarity
            })
        
        # Strategy 3: First 12 characters
        if normalized_target_12:
            line_b_start_12 = line_b_stripped[:12] if len(line_b_stripped) >= 12 else line_b_stripped
            normalized_candidate_12 = normalize_for_comparison(line_b_start_12)
            if normalized_candidate_12:
                distance = edit_distance(normalized_target_12, normalized_candidate_12)
                max_len = max(len(normalized_target_12), len(normalized_candidate_12))
                if max_len > 0:
                    similarity = 1.0 - (distance / max_len)
                    if similarity > best_match_strength:
                        best_match_idx = idx
                        best_match_strength = similarity
                        best_match_line = line_b_stripped
                        best_strategy = "first_12_chars"
                    # Track for closest matches
                    closest_matches.append({
                        'index': idx,
                        'line': line_b_stripped,
                        'strategy': 'first_12_chars',
                        'similarity': similarity,
                        'reporter_similarity': -1.0,
                        'content_similarity': -1.0
                    })
    
    # Sort closest matches by similarity and keep top 3
    closest_matches.sort(key=lambda x: x['similarity'], reverse=True)
    closest_matches = closest_matches[:3]
    
    # Check if best match meets threshold
    match_found = False
    if best_strategy == "reporter_and_content":
        # For reporter+content strategy, need both thresholds met
        match_found = best_match_strength >= 0.65  # Combined threshold
    elif best_match_strength >= 0.7:
        match_found = True
    
    if match_found:
        return (True, best_match_idx, best_match_strength, last_line_a, best_match_line, closest_matches)
    else:
        return (False, -1, best_match_strength if best_match_strength > 0 else -1.0, last_line_a, "", closest_matches)

# --- Merge two files (A and B) ---
def merge_files(file_a_lines: list[str], file_b_lines: list[str], file_a_name: str, file_b_name: str) -> tuple[list[str], dict]:
    """
    Merge file B into file A.
    
    Returns:
        (merged_lines, merge_info)
        - merged_lines: the merged result
        - merge_info: dictionary with merge details for reporting
    """
    merge_info = {
        'file_a': file_a_name,
        'file_b': file_b_name,
        'match_found': False,
        'match_index': -1,
        'match_strength': -1.0,
        'file_a_line': '',
        'file_b_line': '',
        'closest_matches': []
    }
    
    # Find merge point
    match_found, match_idx, match_strength, file_a_line, file_b_line, closest_matches = find_merge_point(file_a_lines, file_b_lines)
    
    merge_info['match_found'] = match_found
    merge_info['match_index'] = match_idx
    merge_info['match_strength'] = match_strength
    merge_info['file_a_line'] = file_a_line
    merge_info['file_b_line'] = file_b_line
    merge_info['closest_matches'] = closest_matches
    
    if match_found:
        # Merge: file A up to (but not including) last line, then file B from matching line
        merged = file_a_lines[:-1] + file_b_lines[match_idx:]
    else:
        # No match: simple concatenation
        merged = file_a_lines + file_b_lines
    
    return merged, merge_info

# --- Remove empty lines from text ---
def remove_empty_lines(text: str) -> str:
    """Remove all empty lines from text."""
    lines = text.splitlines(keepends=True)
    non_empty_lines = [line for line in lines if line.strip()]
    return "".join(non_empty_lines)

# --- Clean up merged text with LLM (minimal changes) ---
def cleanup_with_llm(merged_text: str, folder_name: str) -> str:
    """
    Send merged text to LLM for minimal cleanup - just handle fragments and ensure format.
    """
    print(f"🧹 {folder_name}: Sending to LLM for final cleanup and format consistency...")
    
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": CLEANUP_PROMPT},
            {"role": "user", "content": merged_text}
        ],
        max_completion_tokens=16384,
    )
    
    print(f"✅ {folder_name}: LLM cleanup complete.")
    return response.choices[0].message.content.strip()

# --- Combine all OCR .txt snippets in one folder using iterative merging ---
def combine_transcripts(folder_path: Path) -> tuple[str, str, str]:
    """
    Combine snippets using iterative manual merging, then clean up with LLM.
    
    Returns:
        (merged_text, cleaned_text, combine_report)
    """
    snippet_files = sorted(
        [f for f in folder_path.glob("*.txt") 
         if f.name != f"{folder_path.name}.txt" 
         and f.name != "combine-pre-llm.txt" 
         and f.name != "combine-report.txt"],
        key=lambda f: f.name
    )

    if not snippet_files:
        raise ValueError(f"No .txt files found in {folder_path}")

    print(f"🧩 {folder_path.name}: Found {len(snippet_files)} snippet(s)")

    # Read all files
    file_contents = {}
    for f in snippet_files:
        with open(f, "r", encoding="utf-8") as infile:
            lines = infile.readlines()
            file_contents[f.name] = lines
        print(f"  📄 {f.name}: {len(lines)} lines")

    # Iteratively merge files
    if len(snippet_files) == 0:
        raise ValueError("No files to merge")
    
    # Start with first file
    current_merged = file_contents[snippet_files[0].name].copy()
    merge_reports = []
    
    # Merge remaining files one by one
    for i in range(1, len(snippet_files)):
        file_a_name = snippet_files[i-1].name if i == 1 else f"merged_{i-1}"
        file_b_name = snippet_files[i].name
        
        print(f"  🔗 Merging {file_a_name} + {file_b_name}...")
        
        merged, merge_info = merge_files(current_merged, file_contents[snippet_files[i].name], file_a_name, file_b_name)
        current_merged = merged
        merge_reports.append(merge_info)
        
        if merge_info['match_found']:
            print(f"    ✅ Match found at index {merge_info['match_index']} (strength: {merge_info['match_strength']:.3f})")
        else:
            print(f"    ⚠️  No match found, concatenating")

    # Generate combine report text
    report_lines = []
    report_lines.append(f"Combine Report for {folder_path.name}\n")
    report_lines.append("=" * 60 + "\n\n")
    
    for i, report in enumerate(merge_reports):
        report_lines.append(f"Merge {i+1}: {report['file_a']} + {report['file_b']}\n")
        report_lines.append("-" * 60 + "\n")
        report_lines.append(f"Match found: {report['match_found']}\n")
        
        if report['match_found']:
            report_lines.append(f"Match index in file B: {report['match_index']}\n")
            report_lines.append(f"Match strength: {report['match_strength']:.3f}\n")
            report_lines.append(f"\nFile A last line: {report['file_a_line']}\n")
            report_lines.append(f"File B matching line: {report['file_b_line']}\n")
        else:
            report_lines.append("No matching line found - files concatenated.\n")
            report_lines.append(f"File A last line: {report['file_a_line']}\n")
            if report['match_strength'] > 0:
                report_lines.append(f"Best similarity found: {report['match_strength']:.3f}\n")
            
            # Show closest matches
            if report.get('closest_matches'):
                report_lines.append(f"\nClosest matches (top 3):\n")
                for j, match in enumerate(report['closest_matches'], 1):
                    report_lines.append(f"  {j}. Index {match['index']}, similarity: {match['similarity']:.3f} ({match['strategy']})\n")
                    if match['reporter_similarity'] >= 0:
                        report_lines.append(f"     Reporter similarity: {match['reporter_similarity']:.3f}\n")
                    if match['content_similarity'] >= 0:
                        report_lines.append(f"     Content similarity: {match['content_similarity']:.3f}\n")
                    report_lines.append(f"     Line: {match['line']}\n")
        
        report_lines.append("\n")
    
    combine_report_text = "".join(report_lines)
    merged_text = "".join(current_merged)
    
    # Remove empty lines from merged text
    merged_text = remove_empty_lines(merged_text)
    
    # Clean up with LLM (minimal changes - just fragments and format)
    cleaned_text = cleanup_with_llm(merged_text, folder_path.name)
    
    return merged_text, cleaned_text, combine_report_text

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
    
    # Get all snippet files (exclude output files)
    snippet_files = [
        f for f in folder_path.glob("*.txt") 
        if f.name != output_file.name
        and f.name != "combine-pre-llm.txt"
        and f.name != "combine-report.txt"
    ]
    
    if not snippet_files:
        # No input files, but output exists - don't reprocess
        return False
    
    # Check if any input file is newer than the output
    for snippet_file in snippet_files:
        if snippet_file.stat().st_mtime > output_mtime:
            return True
    
    return False

# --- Process a single folder (for parallel execution) ---
def process_one_folder(folder_path: Path, ignore_existing=False, force=False):
    try:
        output_file_pre_llm = folder_path / "combine-pre-llm.txt"
        output_file_report = folder_path / "combine-report.txt"
        output_file_final = folder_path / f"{folder_path.name}.txt"
        
        # Check if we should skip (using combine-pre-llm.txt as the output indicator)
        if ignore_existing and output_file_pre_llm.exists():
            print(f"⏭️  {folder_path.name}: Output file already exists, skipping")
            return
        
        # Default behavior: check if inputs have changed (unless --force is used)
        if not force and not ignore_existing:
            if not inputs_have_changed(folder_path, output_file_pre_llm):
                print(f"⏭️  {folder_path.name}: No changes detected, skipping")
                return

        merged_text, cleaned_text, combine_report = combine_transcripts(folder_path)
        
        # Write outputs
        with open(output_file_pre_llm, "w", encoding="utf-8") as f:
            f.write(merged_text)
        
        with open(output_file_report, "w", encoding="utf-8") as f:
            f.write(combine_report)
        
        # Save the LLM-cleaned version as the final output
        with open(output_file_final, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        print(f"💾 {folder_path.name}: Saved combine-pre-llm.txt, combine-report.txt, and {output_file_final.name}")
    except Exception as e:
        print(f"❌ {folder_path.name}: Error - {e}")
        import traceback
        traceback.print_exc()

# --- Process all folders in parallel ---
def process_all_folders(base_dir: Path, max_workers=10, ignore_existing=False, force=False):
    week_folders = [f for f in sorted(base_dir.iterdir()) if f.is_dir()]
    
    # Filter out folders with existing output if legacy ignore_existing flag is set
    if ignore_existing:
        filtered_folders = []
        for folder in week_folders:
            output_file = folder / "combine-pre-llm.txt"
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
            output_file = folder / "combine-pre-llm.txt"
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
        process_all_folders(base_dir, max_workers=10, ignore_existing=ignore_existing, force=force)
