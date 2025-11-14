"""
Step 1: Create Archive Folders

Generates weekly archive folders organized by publication date in YYYY-MM-DD format.

Usage:
    python 1-create-folders.py

Or via Makefile:
    make folders

Configuration:
    Edit the start_date_str and num_weeks parameters in the script to customize
    the date range for folder creation.

Behavior:
    - Creates folders in archives/ directory
    - Uses mkdir(exist_ok=True), so existing folders are not overwritten
    - Folders are named as YYYY-MM-DD (e.g., 1925-11-20)
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

def create_weekly_folders(start_date_str, num_weeks):
    # Parse the start date
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    
    # Get the project root (parent of src/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    archives_dir = project_root / "archives"
    
    # Create archives directory if it doesn't exist
    archives_dir.mkdir(exist_ok=True)
    
    # Loop over the number of weeks and create folders
    for i in range(num_weeks):
        # Calculate the folder date
        folder_date = start_date + timedelta(weeks=i)
        folder_name = folder_date.strftime("%Y-%m-%d")
        
        # Create the folder in the archives directory
        folder_path = archives_dir / folder_name
        folder_path.mkdir(exist_ok=True)
        print(f"Created folder: {folder_path}")

# Example usage:
create_weekly_folders(start_date_str="1925-11-20", num_weeks=722)