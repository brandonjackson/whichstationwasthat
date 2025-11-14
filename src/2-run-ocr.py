"""
Step 2: Run OCR on Scanned Images

Processes scanned images (JPEG/PNG) using GPT-5.1 vision API to extract text transcriptions.
Each image generates a corresponding .txt file containing the OCR output.

Usage:
    python 2-run-ocr.py

Or via Makefile:
    make ocr

Prerequisites:
    - OpenAI API key set as OPENAI_API_KEY environment variable
    - Scanned images placed in appropriate weekly folders under archives/
    - Prompt file: 2-run-ocr-prompt.txt

Important Behavior:
    - Skips images that already have corresponding .txt files
    - Existing .txt files are NEVER overwritten
    - You can safely re-run this script without losing manual corrections
    - Process new images while preserving previously processed ones

Note:
    OCR errors are common with historical documents. Review and manually correct
    the generated .txt files as needed. Your corrections will be preserved even
    if you re-run the OCR step.
"""

import os
import base64
import mimetypes
from pathlib import Path
import openai

# --- OpenAI client setup ---
client = openai.OpenAI()  # Will use OPENAI_API_KEY from env

# --- Load prompt from external file ---
script_dir = Path(__file__).parent
prompt_path = script_dir / "2-run-ocr-prompt.txt"
with open(prompt_path, "r", encoding="utf-8") as f:
    PROMPT = f.read()

# --- Convert image to base64 data URL ---
def image_to_base64_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    with open(image_path, "rb") as f:
        image_data = f.read()
        base64_data = base64.b64encode(image_data).decode("utf-8")
        return f"data:{mime_type};base64,{base64_data}"

# --- Check if transcript is packed into too few lines ---
def is_packed_into_few_lines(transcript: str, min_lines: int = 5) -> bool:
    """
    Check if transcript has substantial content but is packed into too few lines.
    Returns True if it has >200 characters but <= min_lines lines.
    """
    lines = len(transcript.strip().splitlines())
    char_count = len(transcript.strip())
    return char_count > 200 and lines <= min_lines

# --- Run OCR using GPT-5.1 ---
def transcribe_with_openai(image_path: str, retry: bool = False) -> str:
    image_url = image_to_base64_data_url(image_path)
    
    # Use more forceful prompt on retry
    if retry:
        retry_prompt = PROMPT + "\n\n⚠️ IMPORTANT: The previous attempt packed everything into too few lines. " \
                                "You MUST format the transcript with proper line breaks. Each report should be on its own line. " \
                                "Do NOT pack multiple reports onto a single line. Use line breaks liberally - it's better to have " \
                                "too many lines than too few. Break the text into multiple lines as it appears in the column."
        system_content = retry_prompt
    else:
        system_content = PROMPT
    
    response = client.chat.completions.create(
        model="gpt-5.1-2025-11-13",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}}
            ]}
        ],
        max_completion_tokens=16384,
    )
    return response.choices[0].message.content

# --- Main processing function ---
def process_folders(folder_root: Path):
    week_folders = sorted([f for f in folder_root.iterdir() if f.is_dir()])
    
    for week_folder in week_folders:
        print(f"\n📁 Processing folder: {week_folder.name}")
        image_files = sorted(
            [f for f in week_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]],
            key=lambda f: f.name,
            reverse=True
        )

        for image_path in image_files:
            output_path = image_path.with_suffix(".txt")

            if output_path.exists():
                print(f"✅ Already exists: {output_path.name}")
                continue

            print(f" → OCR: {image_path.name}")
            try:
                transcript = transcribe_with_openai(str(image_path))
                
                # Check if transcript is packed into too few lines
                if is_packed_into_few_lines(transcript):
                    lines = len(transcript.strip().splitlines())
                    chars = len(transcript.strip())
                    print(f"⚠️  Warning: Transcript has {chars} chars but only {lines} lines - retrying with stricter prompt...")
                    transcript = transcribe_with_openai(str(image_path), retry=True)
                    
                    # Check again after retry
                    if is_packed_into_few_lines(transcript):
                        lines_after = len(transcript.strip().splitlines())
                        print(f"⚠️  Still packed ({lines_after} lines) but saving anyway")
                    else:
                        print(f"✅ Retry successful: now has {len(transcript.strip().splitlines())} lines")
                
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(transcript.strip())
                print(f"📝 Saved: {output_path.name}")
            except Exception as e:
                print(f"❌ Failed to process {image_path.name}: {e}")

# --- Run the script ---
if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    folder_root = project_root / "archives"
    process_folders(folder_root)
