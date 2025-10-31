import pandas as pd
from pathlib import Path

def merge_weekly_csvs(base_dir: Path, output_csv: str = "data.csv"):
    all_rows = []

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue

        csv_path = folder / f"{folder.name}.csv"
        if not csv_path.exists():
            print(f"⚠️ Skipping {folder.name}: No CSV found.")
            continue

        try:
            df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
            df.insert(0, "issue_date", folder.name)  # Add issue date as first column
            all_rows.append(df)
            print(f"✅ Loaded: {csv_path.name} ({len(df)} rows)")
        except Exception as e:
            print(f"❌ Failed to load {csv_path.name}: {e}")

    if all_rows:
        combined_df = pd.concat(all_rows, ignore_index=True)
        # Output to project root, not archives folder
        project_root = base_dir.parent
        output_path = project_root / output_csv
        combined_df.to_csv(output_path, index=False)
        print(f"\n✅ Combined CSV saved to: {output_path}")
        print(f"📊 Total rows: {len(combined_df)}")
    else:
        print("🚫 No data found to merge.")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    archives_dir = project_root / "archives"
    merge_weekly_csvs(archives_dir)
