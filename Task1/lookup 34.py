```
F1 Score: 0.34766220639596546
'recall', 'true', average, warn_for)
{'F1 Score': 0.34766220639596546}
```
# -*- coding: utf-8 -*-
"""
This script performs token classification using an Enhanced Database Lookup method.

It expands the search dictionary by:
1.  Removing Arabic diacritics (Tashkeel) from the source texts.
2.  Splitting long texts into smaller, overlapping segments.

This increases the chances of matching partial or differently vocalized citations
in the test data. The script then evaluates the performance of this method on a
development set using a character-level F1 score.

It requires the following data files:
- Knowledge Base: quran.json, six_hadith_books.json
- Development Set: dev_SubtaskA.xml, dev_SubtaskA.tsv
"""

import json
import pandas as pd
import re
import os
import zipfile
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import numpy as np

# --- 1. Configuration ---

# --- Knowledge Base Paths ---
# Ensure these paths are correct for your environment
QURAN_JSON_PATH = "/content/drive/MyDrive/FinalIslamic/data/quran.json"
HADITH_JSON_PATH = "/content/drive/MyDrive/FinalIslamic/data/six_hadith_books.json"

# --- Development Set Paths ---
DEV_XML_PATH = "/content/drive/MyDrive/FinalIslamic/data/dev_SubtaskA.xml"
DEV_TSV_PATH = "/content/drive/MyDrive/FinalIslamic/data/dev_SubtaskA.tsv"

# --- Output Paths ---
SUBMISSION_TSV_PATH = "submission.tsv"
SUBMISSION_ZIP_PATH = "submission.zip"
DEV_PREDICTIONS_PATH = 'enhanced_lookup_dev_predictions.tsv'


# --- 2. Text Processing and Knowledge Base Enhancement ---

def normalize_arabic(text):
    """Removes Arabic diacritics (Tashkeel) and Tatweel from the text."""
    if not isinstance(text, str):
        return ""
    # This regex targets the Unicode range for Arabic diacritics and the Tatweel character.
    text = re.sub(r'[\u064B-\u0652\u0640]', '', text)
    return text

def split_text_into_segments(text, min_words=5, max_words=15, step=3):
    """
    Splits a long text into smaller, overlapping segments.
    This helps find partial matches.
    """
    words = text.split()
    segments = set()

    # Add the full text as a segment itself
    if words:
        segments.add(text)

    # Generate overlapping segments of different lengths
    for length in range(min_words, max_words + 1):
        for i in range(0, len(words) - length + 1, step):
            segment = ' '.join(words[i:i+length])
            segments.add(segment)
            
    return list(segments)


def build_enhanced_knowledge_base(quran_path, hadith_path):
    """
    Loads Ayahs and Hadiths, then expands the knowledge base with normalized
    and segmented versions of the texts.
    """
    print("Building ENHANCED knowledge base from source files...")
    
    # --- Load Quran Data ---
    try:
        with open(quran_path, 'r', encoding='utf-8') as f:
            quran_data = json.load(f)
        # Use a set for fast, unique lookups
        ayah_set = {item['ayah_text'].strip() for item in quran_data if 'ayah_text' in item and item['ayah_text']}
    except FileNotFoundError:
        print(f"Error: Quran file not found at {quran_path}.")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {quran_path}.")
        return None, None

    # --- Load Hadith Data ---
    try:
        with open(hadith_path, 'r', encoding='utf-8') as f:
            hadith_data = json.load(f)
        # Extract the 'Matn' field, ensuring it's a non-empty string
        hadith_set = {item['Matn'].strip() for item in hadith_data if 'Matn' in item and isinstance(item['Matn'], str) and item['Matn'].strip()}
    except FileNotFoundError:
        print(f"Error: Hadith file not found at {hadith_path}.")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {hadith_path}.")
        return None, None

    print(f"Loaded {len(ayah_set)} unique Ayahs and {len(hadith_set)} unique Hadiths.")

    # --- Enhance Knowledge Base ---
    print("Enhancing knowledge base with normalization and segmentation...")
    enhanced_ayah_set = set()
    for ayah in tqdm(ayah_set, desc="Processing Ayahs"):
        # 1. Add original text
        enhanced_ayah_set.add(ayah)
        # 2. Add normalized (no Tashkeel) text
        normalized_ayah = normalize_arabic(ayah)
        enhanced_ayah_set.add(normalized_ayah)
        # 3. Add segments of both original and normalized text
        for segment in split_text_into_segments(ayah):
            enhanced_ayah_set.add(segment)
        for segment in split_text_into_segments(normalized_ayah):
            enhanced_ayah_set.add(segment)

    enhanced_hadith_set = set()
    for hadith in tqdm(hadith_set, desc="Processing Hadiths"):
        # 1. Add original text
        enhanced_hadith_set.add(hadith)
        # 2. Add normalized text
        normalized_hadith = normalize_arabic(hadith)
        enhanced_hadith_set.add(normalized_hadith)
        # 3. Add segments of both
        for segment in split_text_into_segments(hadith):
            enhanced_hadith_set.add(segment)
        for segment in split_text_into_segments(normalized_hadith):
            enhanced_hadith_set.add(segment)

    # Sort by length (desc) to prioritize matching longer spans first
    sorted_ayahs = sorted(list(enhanced_ayah_set), key=len, reverse=True)
    sorted_hadiths = sorted(list(enhanced_hadith_set), key=len, reverse=True)

    print(f"Knowledge base enhanced. Ayah variations: {len(sorted_ayahs)}, Hadith variations: {len(sorted_hadiths)}.")
    return sorted_ayahs, sorted_hadiths


# --- 3. Data Loading for Prediction/Evaluation ---

def load_data_from_xml(xml_path):
    """Loads question IDs and response texts from the provided XML file."""
    print(f"Loading and parsing data from {xml_path}...")
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Using a more robust regex to handle various XML formatting
        pattern = re.compile(r"<Question>.*?<ID>(.*?)</ID>.*?<Response>(.*?)</Response>.*?</Question>", re.DOTALL)
        matches = pattern.findall(content)

        if not matches:
            print(f"Warning: Could not find any valid <Question> blocks in {xml_path}.")
            return {}

        # Return a dictionary for easy lookup by Question_ID
        data_dict = {match[0].strip(): match[1].strip() for match in matches}
        print(f"Successfully loaded and parsed {len(data_dict)} examples.")
        return data_dict

    except FileNotFoundError:
        print(f"Error: Test file not found at {xml_path}. Please ensure the file exists.")
        return {}


# --- 4. Prediction using Enhanced Database Lookup ---

def predict_with_enhanced_lookup(test_data_dict, ayah_list, hadith_list):
    """
    Finds spans by looking up substrings in the enhanced knowledge base.
    Prioritizes longer matches to avoid fragmented predictions.
    """
    print("Predicting spans using ENHANCED database lookup method...")
    all_predictions = []
    
    # Character array to mark positions that have already been classified
    # This prevents overlapping matches (e.g., a short segment within a full match)
    
    for question_id, text in tqdm(test_data_dict.items(), desc="Predicting"):
        spans_found = []
        # Create a tracker for each character in the text to mark if it's been used in a span
        classified_chars = [False] * len(text)

        # Search for Ayah matches (longer ones first)
        for ayah in ayah_list:
            if ayah in text:
                start_pos = 0
                while True:
                    start_index = text.find(ayah, start_pos)
                    if start_index == -1:
                        break
                    end_index = start_index + len(ayah)
                    # Check if this span overlaps with an already found span
                    if not any(classified_chars[start_index:end_index]):
                        spans_found.append({"Question_ID": question_id, "Span_Start": start_index, "Span_End": end_index, "Span_Type": "Ayah"})
                        # Mark these characters as classified
                        for i in range(start_index, end_index):
                            classified_chars[i] = True
                    start_pos = start_index + 1

        # Reset classified characters for Hadith search to allow different types
        # Or keep them if you want to prevent any overlap at all. Let's keep them.
        
        # Search for Hadith matches (longer ones first)
        for hadith in hadith_list:
            if hadith in text:
                start_pos = 0
                while True:
                    start_index = text.find(hadith, start_pos)
                    if start_index == -1:
                        break
                    end_index = start_index + len(hadith)
                    if not any(classified_chars[start_index:end_index]):
                        spans_found.append({"Question_ID": question_id, "Span_Start": start_index, "Span_End": end_index, "Span_Type": "Hadith"})
                        for i in range(start_index, end_index):
                            classified_chars[i] = True
                    start_pos = start_index + 1
        
        if spans_found:
            all_predictions.extend(spans_found)
        else:
            # If no spans were found, add the 'No_Spans' entry
            all_predictions.append({"Question_ID": question_id, "Span_Start": 0, "Span_End": 0, "Span_Type": "No_Spans"})

    return pd.DataFrame(all_predictions)


# --- 5. Evaluation Logic ---

def evaluate_predictions(predictions_df, reference_df, qid_response_mapping):
    """
    Evaluates predictions using the official character-level scoring logic.
    """
    print("\n🎯 Starting Evaluation using official scoring logic...")
    print("=" * 60)

    # Constants for character-level arrays
    Normal_Text_Tag, Ayah_Tag, Hadith_Tag = 0, 1, 2
    all_y_true, all_y_pred = [], []
    total_f1, count_valid_question = 0, 0

    for question_id in qid_response_mapping.keys():
        # Ensure the question is in both prediction and reference sets
        if question_id not in predictions_df['Question_ID'].values or question_id not in reference_df['Question_ID'].values:
            continue

        count_valid_question += 1
        response_text = qid_response_mapping[question_id]
        
        # --- Ground Truth Array ---
        truth_char_array = [Normal_Text_Tag] * len(response_text)
        question_ground_truth = reference_df[reference_df['Question_ID'] == question_id]
        
        is_no_annotation = (len(question_ground_truth) > 0 and question_ground_truth['Label'].iloc[0] == 'NoAnnotation')

        if not is_no_annotation:
            for _, row in question_ground_truth.iterrows():
                start, end = int(row['Span_Start']), int(row['Span_End'])
                label_type = row['Label']
                tag = Ayah_Tag if label_type == 'Ayah' else Hadith_Tag
                if 0 <= start < end <= len(response_text):
                    truth_char_array[start:end] = [tag] * (end - start)

        # --- Prediction Array ---
        pred_char_array = [Normal_Text_Tag] * len(response_text)
        question_predictions = predictions_df[predictions_df['Question_ID'] == question_id]
        
        is_no_spans_pred = (len(question_predictions) > 0 and question_predictions['Span_Type'].iloc[0] == 'No_Spans')

        if not is_no_spans_pred:
            for _, row in question_predictions.iterrows():
                start, end = int(row['Span_Start']), int(row['Span_End'])
                span_type = row['Span_Type']
                tag = Ayah_Tag if span_type == 'Ayah' else Hadith_Tag
                if 0 <= start < end <= len(response_text):
                    pred_char_array[start:end] = [tag] * (end - start)

        # --- Calculate F1 for this question and aggregate ---
        f1 = f1_score(truth_char_array, pred_char_array, average='macro', zero_division=0)
        total_f1 += f1
        all_y_true.extend(truth_char_array)
        all_y_pred.extend(pred_char_array)

    # Final F1 score is the average of per-question F1 scores
    final_f1_score = total_f1 / count_valid_question if count_valid_question > 0 else 0.0
    
    print_evaluation_report(all_y_true, all_y_pred, final_f1_score)
    return final_f1_score

def print_evaluation_report(y_true, y_pred, final_f1):
    """Generates and prints a detailed classification report."""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE EVALUATION STATISTICS")
    print("="*60)

    label_map = {0: 'Neither', 1: 'Ayah', 2: 'Hadith'}
    y_true_labels = [label_map[label] for label in y_true]
    y_pred_labels = [label_map[label] for label in y_pred]
    
    print("\n📈 CHARACTER-LEVEL CLASSIFICATION REPORT")
    print("-" * 60)
    labels = ['Neither', 'Ayah', 'Hadith']
    print(classification_report(y_true_labels, y_pred_labels, labels=labels, zero_division=0, digits=4))

    print("\n" + "="*60)
    print("🎯 FINAL SUMMARY")
    print("="*60)
    print(f"**Final Macro-Averaged F1 Score (Official Metric): {final_f1:.6f}**")
    print("="*60)


# --- 6. Main Execution Block ---

if __name__ == "__main__":
    # --- STEP 1: Build the Knowledge Base ---
    ayah_list, hadith_list = build_enhanced_knowledge_base(QURAN_JSON_PATH, HADITH_JSON_PATH)

    if ayah_list is None or hadith_list is None:
        print("\n--- Script Aborted: Could not build the knowledge base. ---")
        exit()

    # --- STEP 2: Load Development Data for Evaluation ---
    dev_texts_dict = load_data_from_xml(DEV_XML_PATH)
    
    try:
        ground_truth_df = pd.read_csv(DEV_TSV_PATH, sep='\t')
        print(f"Successfully loaded {len(ground_truth_df)} ground truth annotations.")
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {DEV_TSV_PATH}.")
        dev_texts_dict = {} # Prevent running predictions if ground truth is missing

    if not dev_texts_dict:
        print("\n--- Script Aborted: Could not load development data. ---")
        exit()
        
    # --- STEP 3: Run Prediction on the Development Set ---
    predictions_df = predict_with_enhanced_lookup(dev_texts_dict, ayah_list, hadith_list)

    # Save dev predictions for inspection
    predictions_df.to_csv(DEV_PREDICTIONS_PATH, sep='\t', index=False, header=True)
    print(f"\n📁 Development set predictions saved to: {DEV_PREDICTIONS_PATH}")
    
    # --- STEP 4: Evaluate the Predictions ---
    final_f1 = evaluate_predictions(predictions_df, ground_truth_df, dev_texts_dict)

    print(f"\n🎉 EVALUATION COMPLETED!")
    print(f"Final Macro F1-Score on the development set: {final_f1:.6f}")
