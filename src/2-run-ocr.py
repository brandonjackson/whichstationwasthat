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

# --- Run OCR using GPT-4o ---
def transcribe_with_openai(image_path: str) -> str:
    image_url = image_to_base64_data_url(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}}
            ]}
        ],
        max_tokens=2048,
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
