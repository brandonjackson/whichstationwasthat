.PHONY: help install folders ocr combine parse merge analyze all clean

# Default target
help:
	@echo "Which Station Was That? - Pipeline Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  make install    - Install Python dependencies"
	@echo "  make folders    - Create weekly archive folders (Step 1)"
	@echo "  make ocr        - Run OCR on scanned images (Step 2)"
	@echo "  make combine    - Combine transcript snippets (Step 3)"
	@echo "  make parse      - Parse transcripts to CSV (Step 4)"
	@echo "  make merge      - Merge weekly CSVs into data.csv (Step 5)"
	@echo "  make analyze    - Generate data analysis visualizations (Step 6)"
	@echo "  make all        - Run complete pipeline (steps 1-6)"
	@echo "  make clean      - Remove generated .txt and .csv files (keeps images)"
	@echo ""

# Install Python dependencies
install:
	python3 -m pip install openai pandas matplotlib

# Step 1: Create weekly archive folders
folders:
	@echo "📁 Creating weekly archive folders..."
	cd src && python3 1-create-folders.py

# Step 2: Run OCR on scanned images
ocr:
	@echo "🔍 Running OCR on scanned images..."
	cd src && python3 2-run-ocr.py

# Step 3: Combine transcript snippets
combine:
	@echo "🧩 Combining transcript snippets..."
	cd src && python3 3-combine-transcripts.py

# Step 4: Parse transcripts to CSV
parse:
	@echo "📄 Parsing transcripts to CSV..."
	cd src && python3 4-parse-transcripts-to-csv.py

# Step 5: Merge weekly CSVs
merge:
	@echo "📊 Merging weekly CSVs..."
	cd src && python3 5-merge-weekly-csvs.py

# Step 6: Data Analysis
analyze:
	@echo "📈 Generating data analysis visualizations..."
	cd src && python3 6-analyze-data.py

# Run complete pipeline (steps 1-6)
all: folders ocr combine parse merge analyze
	@echo ""
	@echo "✅ Pipeline complete! Final dataset: data.csv, figures in results/"

# Clean generated files (keeps images)
clean:
	@echo "🧹 Cleaning generated files..."
	@find archives -type f \( -name "*.txt" -o -name "*.csv" \) -not -path "*/\.*" -delete
	@rm -f data.csv
	@echo "✅ Clean complete (images preserved)"

