```python
F1 Score: 0.34766220639596546
'recall', 'true', average, warn_for)
{'F1 Score': 0.34766220639596546}
```


```python
# -*- coding: utf-8 -*-
"""
This script performs token classification using a Database Lookup method.
It identifies spans of Ayahs and Hadiths by creating a knowledge base from
raw Quran and Hadith data and finding exact matches in the test set.

It requires the following data files:
1. quran.json & nine_books_data.csv: For building the knowledge base.
2. test_SubtaskA.xml: The official test file for the final prediction phase.
"""

import json
import pandas as pd
import re
import os
import zipfile

# --- 1. Configuration ---

# --- Input Data Paths ---
# Knowledge base sources
QURAN_JSON_PATH = "quran.json"
NINE_BOOKS_CSV_PATH = "nine_books_data.csv"

# --- Test File Path ---
TEST_XML_PATH = "test_SubtaskA.xml"

# --- Output Paths ---
SUBMISSION_TSV_PATH = "submission.tsv"
SUBMISSION_ZIP_PATH = "submission.zip"


# --- 2. Data Loading and Setup ---

def create_dummy_files():
    """
    Creates dummy data files for demonstration if they don't exist.
    Replace these with your actual data files.
    """
    if not os.path.exists(QURAN_JSON_PATH):
        print(f"Creating dummy '{QURAN_JSON_PATH}'...")
        quran_data = [
            {"surah_id": 1, "surah_name": "الفاتحة", "ayah_id": 1, "ayah_text": "بِسْمِ اللَّهِ الرَّحْمَـٰنِ الرَّحِيمِ"},
            {"surah_id": 1, "surah_name": "الفاتحة", "ayah_id": 2, "ayah_text": "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"}
        ]
        with open(QURAN_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(quran_data, f, ensure_ascii=False, indent=4)

    if not os.path.exists(NINE_BOOKS_CSV_PATH):
        print(f"Creating dummy '{NINE_BOOKS_CSV_PATH}'...")
        nine_books_data = {
            'hadithID': [5],
            'hadithTxt': ["إنما الأعمال بالنيات ، وإنما لكل امرئ ما نوى"]
        }
        pd.DataFrame(nine_books_data).to_csv(NINE_BOOKS_CSV_PATH, index=False)

    if not os.path.exists(TEST_XML_PATH):
        print(f"Creating dummy '{TEST_XML_PATH}'...")
        xml_content = """
<Question>
	<ID>A-Q001</ID>
	<Model>Model-6</Model>
	<Text>هل يمكن أن يكون الابتلاء بفتح الرزق والخيرات عموما؟</Text>
	<Response>
نعم، يمكن أن يكون الابتلاء بفتح الرزق والخيرات عمومًا. يُذكر في القرآن الكريم: {الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ}.
    </Response>
</Question>
<Question>
	<ID>A-Q002</ID>
	<Model>Model-3</Model>
	<Text>ما هو الدليل على وجوب النية؟</Text>
	<Response>الدليل هو قول النبي: إنما الأعمال بالنيات ، وإنما لكل امرئ ما نوى وهو حديث صحيح.</Response>
</Question>
<Question>
	<ID>A-Q003</ID>
	<Model>Model-1</Model>
	<Text>ما هو تعريف الإيمان؟</Text>
	<Response>لا يوجد اقتباس مباشر في هذا الرد.</Response>
</Question>
        """
        with open(TEST_XML_PATH, 'w', encoding='utf-8') as f:
            f.write(xml_content)


def build_knowledge_base(quran_path, hadith_path):
    """Loads all Ayahs and Hadiths into sets for fast lookup."""
    print("Building knowledge base from source files...")
    try:
        with open(quran_path, 'r', encoding='utf-8') as f:
            quran_data = json.load(f)
        hadith_df = pd.read_csv(hadith_path)
    except FileNotFoundError as e:
        print(f"Error: Missing required source data file: {e}.")
        return None, None

    # Using sets provides very fast 'in' checks (average O(1) time complexity)
    # .strip() removes leading/trailing whitespace that could cause a mismatch
    ayah_set = {item['ayah_text'].strip() for item in quran_data if 'ayah_text' in item and item['ayah_text']}
    hadith_set = {hadith.strip() for hadith in hadith_df['hadithTxt'].dropna().tolist() if hadith}

    print(f"Knowledge base built with {len(ayah_set)} unique Ayahs and {len(hadith_set)} unique Hadiths.")
    return ayah_set, hadith_set


def load_test_data_from_xml(xml_path):
    """Loads test data from the provided XML file."""
    print(f"Loading and parsing test data from {xml_path}...")
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = re.compile(r"<Question>.*?<ID>(.*?)</ID>.*?<Response>(.*?)</Response>.*?</Question>", re.DOTALL)
        matches = pattern.findall(content)

        if not matches:
            print("Warning: Could not find any valid <Question> blocks in the test file.")
            return []

        test_data = [{'Question_ID': match[0].strip(), 'Text': match[1].strip()} for match in matches]
        print(f"Successfully loaded and parsed {len(test_data)} test examples.")
        return test_data

    except FileNotFoundError:
        print(f"Error: Test file not found at {xml_path}. Please ensure the file exists.")
        return []


# --- 3. Prediction using Database Lookup ---

def predict_with_lookup(test_data, ayah_set, hadith_set):
    """Finds spans by looking up substrings in the knowledge base."""
    print("Predicting spans using database lookup method...")
    all_predictions = []

    for item in test_data:
        question_id = item["Question_ID"]
        text = item["Text"]
        spans_found = []

        # Search for Ayah matches
        for ayah in ayah_set:
            if ayah in text:
                # Find all occurrences of the same Ayah in the text
                start_pos = 0
                while True:
                    start_index = text.find(ayah, start_pos)
                    if start_index == -1:
                        break
                    end_index = start_index + len(ayah)
                    spans_found.append({"Question_ID": question_id, "Span_Start": start_index, "Span_End": end_index, "Span_Type": "Ayah"})
                    start_pos = start_index + 1 # Continue search from the next character

        # Search for Hadith matches
        for hadith in hadith_set:
            if hadith in text:
                start_pos = 0
                while True:
                    start_index = text.find(hadith, start_pos)
                    if start_index == -1:
                        break
                    end_index = start_index + len(hadith)
                    spans_found.append({"Question_ID": question_id, "Span_Start": start_index, "Span_End": end_index, "Span_Type": "Hadith"})
                    start_pos = start_index + 1

        if spans_found:
            all_predictions.extend(spans_found)
        else:
            # If no spans were found for this Question_ID, add the 'No_Spans' entry
            all_predictions.append({"Question_ID": question_id, "Span_Start": 0, "Span_End": 0, "Span_Type": "No_Spans"})

    return all_predictions


# --- 4. Submission File Generation ---

def generate_submission_file(predictions, output_path, zip_path):
    """Generates the final TSV submission file and zips it."""
    if not predictions:
        print("No predictions to save. Skipping submission file generation.")
        return

    print(f"Generating submission file at {output_path}...")
    submission_df = pd.DataFrame(predictions)
    # Ensure the correct column order
    submission_df = submission_df[["Question_ID", "Span_Start", "Span_End", "Span_Type"]]

    submission_df.to_csv(output_path, sep='\t', index=False, header=False)
    print("Submission file generated.")

    print(f"Creating zip file at {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(output_path, os.path.basename(output_path))
    print("Zip file created successfully.")


# --- Main Execution Block ---

if __name__ == "__main__":
    create_dummy_files()

    # Build the knowledge base from source files
    ayah_set, hadith_set = build_knowledge_base(QURAN_JSON_PATH, NINE_BOOKS_CSV_PATH)

    if ayah_set is None:
        print("\n--- Script Aborted: Could not build knowledge base. ---")
        exit()

    # Load the test data from the XML file
    test_data = load_test_data_from_xml(TEST_XML_PATH)

    if test_data:
        # Run the prediction logic
        predictions = predict_with_lookup(test_data, ayah_set, hadith_set)

        # Generate the final submission files
        generate_submission_file(predictions, SUBMISSION_TSV_PATH, SUBMISSION_ZIP_PATH)

        print("\n--- Script Finished ---")
    else:
        print("\n--- Script Aborted: Could not load test data. ---")
```

    Building knowledge base from source files...
    Knowledge base built with 6058 unique Ayahs and 68560 unique Hadiths.
    Loading and parsing test data from test_SubtaskA.xml...
    Successfully loaded and parsed 104 test examples.
    Predicting spans using database lookup method...
    Generating submission file at submission.tsv...
    Submission file generated.
    Creating zip file at submission.zip...
    Zip file created successfully.
    
    --- Script Finished ---
    
