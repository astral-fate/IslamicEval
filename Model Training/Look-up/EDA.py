# -*- coding: utf-8 -*-
"""
enhanced_lookup_with_analysis.py

This script performs token classification using an Enhanced Database Lookup method
and then conducts an advanced error analysis on its predictions.

Pipeline:
1.  Builds an enhanced knowledge base by normalizing and segmenting source texts.
2.  Loads the development set text and ground truth annotations.
3.  Runs the lookup prediction logic on the development set.
4.  Performs a multi-part error analysis on the results:
    - Character-Level Classification Report & Confusion Matrix.
    - Span-Level Error Logging (False Positives/Negatives).
    - Performance vs. Span Length Analysis with statistics and boxplots.
"""

import json
import pandas as pd
import re
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Configuration ---

# --- Knowledge Base Paths ---
# Ensure these paths are correct for your environment
QURAN_JSON_PATH = "/content/drive/MyDrive/FinalIslamic/data/quran.json"
HADITH_JSON_PATH = "/content/drive/MyDrive/FinalIslamic/data/six_hadith_books.json"

# --- Development Set Paths ---
DEV_XML_PATH = "/content/drive/MyDrive/FinalIslamic/data/dev_SubtaskA.xml"
DEV_TSV_PATH = "/content/drive/MyDrive/FinalIslamic/data/dev_SubtaskA.tsv"

# --- Output Paths for Analysis ---
OUTPUT_CSV_PATH = "/content/enhanced_lookup_error_analysis.csv" # Saved to Colab's local storage

# --- 2. Text Processing and Knowledge Base Enhancement ---

def normalize_arabic(text):
    """Removes Arabic diacritics (Tashkeel) and Tatweel from the text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\u064B-\u0652\u0640]', '', text)
    return text

def split_text_into_segments(text, min_words=5, max_words=15, step=3):
    """Splits a long text into smaller, overlapping segments."""
    words = text.split()
    segments = set()
    if words:
        segments.add(text)
    for length in range(min_words, max_words + 1):
        for i in range(0, len(words) - length + 1, step):
            segment = ' '.join(words[i:i+length])
            segments.add(segment)
    return list(segments)

def build_enhanced_knowledge_base(quran_path, hadith_path):
    """Loads and expands the knowledge base with normalized and segmented texts."""
    print("Building ENHANCED knowledge base from source files...")
    try:
        with open(quran_path, 'r', encoding='utf-8') as f:
            quran_data = json.load(f)
        ayah_set = {item['ayah_text'].strip() for item in quran_data if 'ayah_text' in item and item['ayah_text']}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading Quran data: {e}")
        return None, None

    try:
        with open(hadith_path, 'r', encoding='utf-8') as f:
            hadith_data = json.load(f)
        hadith_set = {item['Matn'].strip() for item in hadith_data if 'Matn' in item and isinstance(item['Matn'], str) and item['Matn'].strip()}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading Hadith data: {e}")
        return None, None

    print(f"Loaded {len(ayah_set)} unique Ayahs and {len(hadith_set)} unique Hadiths.")
    print("Enhancing knowledge base with normalization and segmentation...")
    enhanced_ayah_set = set()
    for ayah in tqdm(ayah_set, desc="Processing Ayahs"):
        normalized_ayah = normalize_arabic(ayah)
        for text_version in {ayah, normalized_ayah}:
            enhanced_ayah_set.add(text_version)
            for segment in split_text_into_segments(text_version):
                enhanced_ayah_set.add(segment)

    enhanced_hadith_set = set()
    for hadith in tqdm(hadith_set, desc="Processing Hadiths"):
        normalized_hadith = normalize_arabic(hadith)
        for text_version in {hadith, normalized_hadith}:
            enhanced_hadith_set.add(text_version)
            for segment in split_text_into_segments(text_version):
                enhanced_hadith_set.add(segment)

    sorted_ayahs = sorted(list(enhanced_ayah_set), key=len, reverse=True)
    sorted_hadiths = sorted(list(enhanced_hadith_set), key=len, reverse=True)
    print(f"Knowledge base enhanced. Ayah variations: {len(sorted_ayahs)}, Hadith variations: {len(sorted_hadiths)}.")
    return sorted_ayahs, sorted_hadiths

# --- 3. Data Loading and Prediction ---

def load_data_from_xml(xml_path):
    """Loads question IDs and response texts from the provided XML file."""
    print(f"Loading and parsing data from {xml_path}...")
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        pattern = re.compile(r"<Question>.*?<ID>(.*?)</ID>.*?<Response>(.*?)</Response>.*?</Question>", re.DOTALL)
        matches = pattern.findall(content)
        data_dict = {match[0].strip(): match[1].strip() for match in matches}
        print(f"Successfully loaded and parsed {len(data_dict)} examples.")
        return data_dict
    except FileNotFoundError:
        print(f"Error: XML file not found at {xml_path}.")
        return {}

def predict_with_enhanced_lookup(test_data_dict, ayah_list, hadith_list):
    """Finds spans by looking up substrings in the enhanced knowledge base."""
    print("Predicting spans using ENHANCED database lookup method...")
    all_predictions = []
    for question_id, text in tqdm(test_data_dict.items(), desc="Predicting"):
        spans_found = []
        classified_chars = [False] * len(text)

        # Search for Ayah matches (longer ones first)
        for ayah in ayah_list:
            if ayah in text:
                start_pos = 0
                while True:
                    start_index = text.find(ayah, start_pos)
                    if start_index == -1: break
                    end_index = start_index + len(ayah)
                    if not any(classified_chars[start_index:end_index]):
                        spans_found.append({"Question_ID": question_id, "Span_Start": start_index, "Span_End": end_index, "Span_Type": "Ayah"})
                        classified_chars[start_index:end_index] = [True] * len(ayah)
                    start_pos = start_index + 1

        # Search for Hadith matches
        for hadith in hadith_list:
            if hadith in text:
                start_pos = 0
                while True:
                    start_index = text.find(hadith, start_pos)
                    if start_index == -1: break
                    end_index = start_index + len(hadith)
                    if not any(classified_chars[start_index:end_index]):
                        spans_found.append({"Question_ID": question_id, "Span_Start": start_index, "Span_End": end_index, "Span_Type": "Hadith"})
                        classified_chars[start_index:end_index] = [True] * len(hadith)
                    start_pos = start_index + 1
        
        if spans_found:
            all_predictions.extend(spans_found)
        else:
            all_predictions.append({"Question_ID": question_id, "Span_Start": 0, "Span_End": 0, "Span_Type": "No_Spans"})

    return pd.DataFrame(all_predictions)

# --- 4. Advanced Analysis Functions ---

def character_level_analysis(ground_truth_df, predictions_df, dev_texts):
    """Performs character-level analysis and plots a confusion matrix."""
    print("\n" + "="*80)
    print("📊 ANALYSIS 1: CHARACTER-LEVEL PERFORMANCE")
    print("="*80)

    label_map = {'Neither': 0, 'Ayah': 1, 'Hadith': 2}
    tag_map = {0: 'Neither', 1: 'Ayah', 2: 'Hadith'}
    all_y_true, all_y_pred = [], []

    for qid, text in dev_texts.items():
        true_chars = [label_map['Neither']] * len(text)
        true_spans = ground_truth_df[ground_truth_df['Question_ID'] == qid]
        if not (len(true_spans) > 0 and true_spans['Label'].iloc[0] == 'NoAnnotation'):
             for _, row in true_spans.iterrows():
                start, end, label = int(row['Span_Start']), int(row['Span_End']), row['Label']
                if end <= len(text):
                    true_chars[start:end] = [label_map[label]] * (end - start)

        pred_chars = [label_map['Neither']] * len(text)
        pred_spans = predictions_df[predictions_df['Question_ID'] == qid]
        if not (len(pred_spans) > 0 and pred_spans['Span_Type'].iloc[0] == 'No_Spans'):
            for _, row in pred_spans.iterrows():
                start, end, label = int(row['Span_Start']), int(row['Span_End']), row['Span_Type']
                if end <= len(text):
                    pred_chars[start:end] = [label_map[label]] * (end - start)

        all_y_true.extend(true_chars)
        all_y_pred.extend(pred_chars)

    y_true_labels = [tag_map[tag] for tag in all_y_true]
    y_pred_labels = [tag_map[tag] for tag in all_y_pred]
    labels = ['Neither', 'Ayah', 'Hadith']

    print("\n📈 Character-Level Classification Report:\n")
    print(classification_report(y_true_labels, y_pred_labels, labels=labels, digits=4))

    print("\n📈 Character-Level Confusion Matrix:")
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Character-Level Confusion Matrix (Enhanced Lookup)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def span_level_error_analysis(ground_truth_df, predictions_df, dev_texts):
    """Identifies and logs False Positive and False Negative spans."""
    print("\n" + "="*80)
    print("📝 ANALYSIS 2: SPAN-LEVEL ERROR LOGGING (FP/FN)")
    print("="*80)

    error_log = []
    span_stats = {'TP': [], 'FP': [], 'FN': []}

    for qid in tqdm(dev_texts.keys(), desc="Analyzing Spans"):
        text = dev_texts[qid]
        true_spans = ground_truth_df[ground_truth_df['Question_ID'] == qid]
        pred_spans = predictions_df[predictions_df['Question_ID'] == qid]

        true_spans_set = set()
        if not (len(true_spans) > 0 and true_spans['Label'].iloc[0] == 'NoAnnotation'):
            for _, r in true_spans.iterrows():
                true_spans_set.add((int(r['Span_Start']), int(r['Span_End']), r['Label']))

        pred_spans_set = set()
        if not (len(pred_spans) > 0 and pred_spans['Span_Type'].iloc[0] == 'No_Spans'):
            for _, r in pred_spans.iterrows():
                pred_spans_set.add((int(r['Span_Start']), int(r['Span_End']), r['Span_Type']))

        matched_true_spans = set()
        matched_pred_spans = set()

        for t_start, t_end, t_label in true_spans_set:
            for p_start, p_end, p_label in pred_spans_set:
                if max(t_start, p_start) < min(t_end, p_end) and t_label == p_label:
                    span_info = {'text': text[t_start:t_end], 'label': t_label, 'length': t_end - t_start}
                    if span_info not in span_stats['TP']:
                        span_stats['TP'].append(span_info)
                    matched_true_spans.add((t_start, t_end, t_label))
                    matched_pred_spans.add((p_start, p_end, p_label))

        for t_start, t_end, t_label in true_spans_set - matched_true_spans:
            span_text = text[t_start:t_end]
            error_log.append({'Question_ID': qid, 'Error_Type': 'False Negative', 'Label': t_label, 'Span_Text': span_text})
            span_stats['FN'].append({'text': span_text, 'label': t_label, 'length': len(span_text)})

        for p_start, p_end, p_label in pred_spans_set - matched_pred_spans:
            span_text = text[p_start:p_end]
            error_log.append({'Question_ID': qid, 'Error_Type': 'False Positive', 'Label': p_label, 'Span_Text': span_text})
            span_stats['FP'].append({'text': span_text, 'label': p_label, 'length': len(span_text)})

    error_df = pd.DataFrame(error_log)
    error_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ Detailed error log saved to: {OUTPUT_CSV_PATH}")
    print(f"Found {len(span_stats['TP'])} TPs, {len(span_stats['FP'])} FPs, and {len(span_stats['FN'])} FNs.")
    return span_stats

def analyze_performance_by_length(span_stats):
    """Visualizes model performance based on the length of the spans."""
    print("\n" + "="*80)
    print("📏 ANALYSIS 3: PERFORMANCE BY SPAN LENGTH")
    print("="*80)

    stats_data = {
        'True Positives': [s['length'] for s in span_stats['TP']],
        'False Positives': [s['length'] for s in span_stats['FP']],
        'False Negatives': [s['length'] for s in span_stats['FN']]
    }

    print("\n📈 Descriptive Statistics for Span Lengths:")
    for key, values in stats_data.items():
        if values:
            print(f"  - {key}:")
            print(f"    - Count: {len(values)}")
            print(f"    - Mean: {np.mean(values):.2f}, Median: {np.median(values):.2f}")
            print(f"    - Min: {np.min(values)}, Max: {np.max(values)}")

    plot_df = pd.DataFrame([
        {'Category': key, 'Length': length}
        for key, values in stats_data.items()
        for length in values
    ])

    print("\n📈 Boxplot of Span Lengths by Category:")
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='Category', y='Length', data=plot_df)
    plt.title('Distribution of Span Lengths for TP, FP, and FN (Enhanced Lookup)')
    plt.ylabel('Span Length (characters)')
    plt.xlabel('Category')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# --- 5. Main Execution Block ---

if __name__ == "__main__":
    print("🚀 Starting Enhanced Lookup Method with Advanced Error Analysis")
    print("-" * 60)

    ayah_list, hadith_list = build_enhanced_knowledge_base(QURAN_JSON_PATH, HADITH_JSON_PATH)
    if ayah_list is None:
        print("\n--- Script Aborted: Could not build the knowledge base. ---")
    else:
        dev_texts_dict = load_data_from_xml(DEV_XML_PATH)
        try:
            ground_truth_df = pd.read_csv(DEV_TSV_PATH, sep='\t')
            print(f"Successfully loaded {len(ground_truth_df)} ground truth annotations.")
            
            predictions_df = predict_with_enhanced_lookup(dev_texts_dict, ayah_list, hadith_list)
            
            # --- Run Full Analysis ---
            character_level_analysis(ground_truth_df, predictions_df, dev_texts_dict)
            span_stats = span_level_error_analysis(ground_truth_df, predictions_df, dev_texts_dict)
            analyze_performance_by_length(span_stats)
            
            print("\n🎉 Analysis complete!")

        except FileNotFoundError:
            print(f"Error: Ground truth file not found at {DEV_TSV_PATH}.")
            print("\n--- Script Aborted: Could not load development data. ---")
