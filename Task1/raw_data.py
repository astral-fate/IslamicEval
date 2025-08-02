# SCRIPT 1: format_raw_texts.py
# This script reads raw Quran and Hadith data and formats it into a clean
# JSON structure for token classification tasks.

import json
import os

# --- 1. Configuration ---
QURAN_JSON_PATH = "quran.json"
SIX_HADITH_BOOKS_JSON_PATH = "six_hadith_books.json"
OUTPUT_RAW_DATASET_PATH = "raw_dataset.json" # Output file for this script

# --- 2. Helper Function ---
def create_dummy_files():
    """Creates dummy data files if they don't exist, for demonstration."""
    if not os.path.exists(QURAN_JSON_PATH):
        print(f"Creating dummy '{QURAN_JSON_PATH}'...")
        quran_data = [{"ayah_text": "بِسْمِ اللَّهِ الرَّحْمَـٰنِ الرَّحِيمِ"}]
        with open(QURAN_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(quran_data, f, ensure_ascii=False, indent=4)

    if not os.path.exists(SIX_HADITH_BOOKS_JSON_PATH):
        print(f"Creating dummy '{SIX_HADITH_BOOKS_JSON_PATH}'...")
        six_hadith_data = [{"hadithTxt": "من سلك طريقا يلتمس فيه علما سهل الله له به طريقا إلى الجنة"}]
        with open(SIX_HADITH_BOOKS_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(six_hadith_data, f, ensure_ascii=False, indent=4)

# --- 3. Main Data Formatting Function ---
def format_raw_data(quran_path, hadith_path, output_path):
    """
    Loads raw religious texts and saves them in a structured format.
    """
    print("Starting raw data formatting process...")
    try:
        with open(quran_path, 'r', encoding='utf-8') as f:
            quran_data = json.load(f)
        with open(hadith_path, 'r', encoding='utf-8') as f:
            six_books_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure source files are in the same directory.")
        return

    # Extract the text from the source files [cite: 1]
    ayah_texts = [item['ayah_text'] for item in quran_data if 'ayah_text' in item and item['ayah_text']]
    hadith_texts = [item['hadithTxt'] for item in six_books_data if 'hadithTxt' in item and item['hadithTxt']]
    print(f"Loaded {len(ayah_texts)} Ayahs and {len(hadith_texts)} Hadiths.")

    processed_data = []

    # Format Ayah texts
    for ayah in ayah_texts:
        processed_data.append({
            "full_text": ayah,
            "span_text": ayah,
            "char_start": 0,
            "char_end": len(ayah),
            "label_type": "Ayah"
        })

    # Format Hadith texts
    for hadith in hadith_texts:
        processed_data.append({
            "full_text": hadith,
            "span_text": hadith,
            "char_start": 0,
            "char_end": len(hadith),
            "label_type": "Hadith"
        })

    # Save the final formatted dataset
    print(f"Saving {len(processed_data)} total examples to '{output_path}'...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print("✅ Raw dataset saved successfully.")

# --- 4. Main Execution ---
if __name__ == "__main__":
    create_dummy_files() # Create dummy files if needed
    format_raw_data(QURAN_JSON_PATH, SIX_HADITH_BOOKS_JSON_PATH, OUTPUT_RAW_DATASET_PATH)
    print("\n--- Raw Data Formatting Script Finished ---")
