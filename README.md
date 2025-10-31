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

1. **Folder Creation** (`1-create-folders.py`)  
   Generates weekly archive folders organized by publication date.

2. **Optical Character Recognition** (`2-run-ocr.py`)  
   Processes scanned images (JPEG/PNG) using GPT-4o vision to extract text transcriptions. Each image generates a corresponding `.txt` file containing the OCR output.

3. **Transcript Combination** (`3-combine-transcripts.py`)  
   Merges multiple OCR snippet files from a single week's column into a single consolidated transcript file (`YYYY-MM-DD.txt`).

4. **CSV Parsing** (`4-parse-transcripts-to-csv.py`)  
   Extracts structured data from combined transcripts using GPT-4o, generating weekly CSV files (`YYYY-MM-DD.csv`) with fields including:
   - Reporter name and location
   - Station location and identification
   - Observation text and full report text
   - Geographic coordinates

5. **Data Merging** (`5-merge-weekly-csvs.py`)  
   Combines all weekly CSV files into a single `data.csv` file with an `issue_date` column for temporal analysis.

## 🚀 Installation & Usage

### Prerequisites

- Python 3.8+
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- Required Python packages:
  ```bash
  pip install openai pandas
  ```

### Running the Pipeline

#### Step 1: Create Archive Folders

```bash
cd src
python 1-create-folders.py
```

This creates weekly folders in `archives/` based on the start date and number of weeks specified in the script.

#### Step 2: Run OCR on Scanned Images

Place your scanned images (JPEG/PNG) in the appropriate weekly folders under `archives/`. Then run:

```bash
python 2-run-ocr.py
```

This processes all images in the archive folders and generates `.txt` files with OCR transcriptions. The script skips images that already have corresponding `.txt` files.

**Note:** OCR errors are common with historical documents. Review and manually correct the generated `.txt` files as needed before proceeding to the next step.

#### Step 3: Combine Transcripts

```bash
python 3-combine-transcripts.py
```

This merges all OCR snippet files within each week's folder into a single consolidated transcript file (`YYYY-MM-DD.txt`).

#### Step 4: Parse Transcripts to CSV

```bash
python 4-parse-transcripts-to-csv.py
```

This extracts structured data from the combined transcripts and generates weekly CSV files. Each CSV contains parsed reports with standardized fields.

#### Step 5: Merge Weekly CSVs

```bash
python 5-merge-weekly-csvs.py
```

This combines all weekly CSV files into a single `data.csv` file in the project root, ready for analysis.

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
