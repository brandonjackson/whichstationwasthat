# Which Station Was That?

A digital history project digitizing and publishing amateur DX (long-distance) radio listening reports from the 1920s and 1930s, extracted from the BBC's "Which Station Was That?" column in *World Radio* magazine. This project makes accessible a valuable archive of early radio culture, capturing how amateur listeners across Europe documented and shared their experiences of long-distance radio reception during the pioneering years of broadcast radio.

## 📚 About This Project

Long-distance radio listening, known as "DXing" (DX = "distance"), emerged as a popular hobby in the 1920s as radio technology expanded across Europe. Amateur listeners documented their reception of distant stations, often providing detailed descriptions of signal strength, program content, call signs, and atmospheric conditions. These reports offer unique insights into early radio culture, amateur experimentation, and the development of international broadcasting networks.

The "Which Station Was That?" column, which ran from 1925 to 1939 in the BBC's *World Radio* magazine (and its predecessor, *Radio Supplement*), published letters from amateur listeners who had received unidentified distant stations and requested help identifying them. Each column typically featured dozens of reports, creating a rich, weekly snapshot of European radio reception during this formative period.

This project transforms scanned magazine pages into structured, machine-readable data, enabling researchers to analyze patterns in reception reports, geographic coverage, station identification, and the evolving vocabulary of early radio enthusiasts.

## 🗂️ Archives Overview

**Primary Sources:**
- *Radio Supplement* (1922-1925)
- *World Radio* magazine (1925-1939)

**Repository:** University of Cambridge Library

**Column Duration:** 1925-1939

The source materials are preserved in the University of Cambridge Library's collections. Scanned images of the "Which Station Was That?" column pages have been digitized and organized chronologically in weekly folders (YYYY-MM-DD format) within the `archives/` directory.

All scans are released under a **CC0 (Public Domain Dedication)** license, making them freely available for research, publication, and reuse.

## 🔄 Pipeline Overview

The digitization pipeline consists of five sequential stages:

1. **Folder Creation** (`make folders`)  
   Generates weekly archive folders organized by publication date.

2. **Optical Character Recognition** (`make ocr`)  
   Processes scanned images (JPEG/PNG) using GPT-4o vision to extract text transcriptions. Each image generates a corresponding `.txt` file containing the OCR output.

3. **Transcript Combination** (`make combine`)  
   Merges multiple OCR snippet files from a single week's column into a single consolidated transcript file (`YYYY-MM-DD.txt`).

4. **CSV Parsing** (`make parse`)  
   Extracts structured data from combined transcripts using GPT-4o, generating weekly CSV files (`YYYY-MM-DD.csv`) with fields including:
   - Reporter name and location
   - Station location and identification
   - Observation text and full report text
   - Geographic coordinates

5. **Data Merging** (`make merge`)  
   Combines all weekly CSV files into a single `data.csv` file with an `issue_date` column for temporal analysis.

## 🚀 Installation & Usage

### Prerequisites

- Python 3.8+
- Make (usually pre-installed on macOS/Linux)
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)

### Installation

Install required Python packages:

```bash
make install
```

This installs `openai` and `pandas`.

### Running the Pipeline

The project uses a `Makefile` with convenient commands. Run the complete pipeline:

```bash
make all
```

This executes all five stages sequentially. **Safe to re-run:** The pipeline won't overwrite existing `.txt` files from the OCR step, and it won't clean or delete any files.

You can also run each stage individually:

```bash
make folders    # Step 1: Create archive folders
make ocr        # Step 2: Run OCR on images (skips existing .txt files)
make combine    # Step 3: Combine transcripts
make parse      # Step 4: Parse to CSV (skips existing CSV files)
make merge      # Step 5: Merge weekly CSVs
make help       # Show all available commands
make clean      # Remove generated files (keeps images)
```

**Note:** For detailed documentation on each step, including behavior, prerequisites, and output formats, see the docstrings in the individual Python scripts in the `src/` directory.

## 📊 Results Overview

The final output (`data.csv`) contains structured records of DX listening reports with the following fields:

- `issue_date`: Publication date (YYYY-MM-DD)
- `reporter_name`: Name or pseudonym of the listener
- `reporter_location_city`, `reporter_location_country`: Listener's location
- `reporter_location_latitude`, `reporter_location_longitude`: Geographic coordinates
- `observation_text`: Description of the reception
- `station_location_city`, `station_location_country`: Identified station location
- `station_location_latitude`, `station_location_longitude`: Station coordinates
- `full_text`: Complete original report text

The dataset enables quantitative and qualitative analysis of early radio reception patterns, geographic distribution of listeners and stations, seasonal variations, and the development of radio terminology and community practices.

## 📁 Project Structure

```
whichstationwasthat/
├── archives/          # Weekly folders containing scans and processed data
│   └── YYYY-MM-DD/    # Individual issue folders
│       ├── *.jpeg     # Scanned images
│       ├── *.txt      # OCR transcripts (snippets and combined)
│       └── *.csv      # Parsed weekly data
├── src/               # Pipeline scripts
│   ├── 1-create-folders.py
│   ├── 2-run-ocr.py
│   ├── 3-combine-transcripts.py
│   ├── 4-parse-transcripts-to-csv.py
│   └── 5-merge-weekly-csvs.py
├── Makefile           # Convenient command shortcuts
├── data.csv           # Final merged dataset
└── README.md          # This file
```

## 📄 License

**Scans and Images:** Released under **CC0 1.0 Universal (Public Domain Dedication)**, allowing unrestricted use, modification, and distribution.

**Code:** See LICENSE file for details.

## 🤝 Contributing

This is a digital history project in progress. Contributions, corrections, and research applications are welcome. Please see the project repository for guidelines on reporting OCR errors, suggesting improvements to parsing logic, or sharing research findings.

## 🔗 Acknowledgments

- Source materials: *World Radio* magazine and *Radio Supplement*, University of Cambridge Library
- Project inspired by the historical value of amateur radio culture and early broadcasting networks
