# -*- coding: utf-8 -*-
"""
enhanced_error_analysis.py

Purpose:
- Load a fine-tuned AraBERT model for token classification.
- Run predictions on the development set.
- Generate an advanced error analysis including:
  1. A character-level confusion matrix.
  2. A detailed log of False Positive and False Negative spans.
  3. A statistical and visual analysis of model performance based on span length.
"""
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. Configuration ---
# Ensure these paths are correct for your environment
FINETUNED_MODEL_PATH = "/content/drive/MyDrive/FinalIslamic/arabert_finetuned_basline_30s"
DEV_XML_PATH = "/content/drive/MyDrive/FinalIslamic/data/dev_SubtaskA.xml"
DEV_TSV_PATH = "/content/drive/MyDrive/FinalIslamic/data/dev_SubtaskA.tsv"
OUTPUT_CSV_PATH = "/content/dev_set_error_analysis.csv" # Saved to Colab's local storage

# --- 2. Data Loading and Prediction Functions ---

def load_dev_data(xml_path, tsv_path):
    """Loads development texts and ground truth annotations."""
    print(f"📖 Loading development data...")
    # Load texts from XML
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        pattern = re.compile(r"<Question>.*?<ID>(.*?)</ID>.*?<Response>(.*?)</Response>.*?</Question>", re.DOTALL)
        matches = pattern.findall(content)
        dev_texts = {m[0].strip(): m[1].strip() for m in matches}
        print(f"✅ Loaded {len(dev_texts)} development question texts.")
    except FileNotFoundError:
        print(f"❌ FATAL: XML file not found at {xml_path}")
        return None, None

    # Load ground truth from TSV
    try:
        ground_truth_df = pd.read_csv(tsv_path, sep='\t')
        print(f"✅ Loaded {len(ground_truth_df)} ground truth annotations.")
    except FileNotFoundError:
        print(f"❌ FATAL: TSV file not found at {tsv_path}")
        return None, None

    return dev_texts, ground_truth_df

def predict_on_dev_set(model, tokenizer, dev_texts_dict):
    """Generates span predictions for the entire development set."""
    print("🤖 Predicting spans using the fine-tuned model...")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    label_list = list(model.config.id2label.values())
    all_predictions = []

    for qid, text in tqdm(dev_texts_dict.items(), desc="Predicting on Dev Set"):
        if not text or not text.strip():
            continue

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits

        preds = torch.argmax(logits, dim=2)[0].cpu().numpy()
        spans = []
        current_span = None

        for i, pred_id in enumerate(preds):
            label = label_list[pred_id]
            word_id = inputs.word_ids(batch_index=0)[i]
            if word_id is None: continue

            if label.startswith('B-'):
                if current_span: spans.append(current_span)
                cs = inputs.token_to_chars(i)
                current_span = {'type': label[2:], 'start': cs.start, 'end': cs.end}
            elif label.startswith('I-') and current_span and current_span['type'] == label[2:]:
                cs = inputs.token_to_chars(i)
                if cs: current_span['end'] = cs.end
            else:
                if current_span:
                    spans.append(current_span)
                current_span = None

        if current_span: spans.append(current_span)

        if spans:
            for span in spans:
                all_predictions.append({
                    "Question_ID": qid,
                    "Span_Start": span['start'],
                    "Span_End": span['end'],
                    "Span_Type": span['type']
                })

    return pd.DataFrame(all_predictions)


# --- 3. Advanced Analysis Functions ---

def character_level_analysis(ground_truth_df, predictions_df, dev_texts):
    """Performs character-level analysis, including a classification report and confusion matrix."""
    print("\n" + "="*80)
    print("📊 ANALYSIS 1: CHARACTER-LEVEL PERFORMANCE")
    print("="*80)

    label_map = {'Neither': 0, 'Ayah': 1, 'Hadith': 2}
    tag_map = {0: 'Neither', 1: 'Ayah', 2: 'Hadith'}
    all_y_true, all_y_pred = [], []

    for qid, text in dev_texts.items():
        # Create ground truth character array
        true_chars = [label_map['Neither']] * len(text)
        true_spans = ground_truth_df[ground_truth_df['Question_ID'] == qid]
        if not (len(true_spans) > 0 and true_spans['Label'].iloc[0] == 'NoAnnotation'):
             for _, row in true_spans.iterrows():
                start, end, label = int(row['Span_Start']), int(row['Span_End']), row['Label']
                if end <= len(text):
                    true_chars[start:end] = [label_map[label]] * (end - start)

        # Create prediction character array
        pred_chars = [label_map['Neither']] * len(text)
        pred_spans = predictions_df[predictions_df['Question_ID'] == qid]
        for _, row in pred_spans.iterrows():
            start, end, label = int(row['Span_Start']), int(row['Span_End']), row['Span_Type']
            if end <= len(text):
                pred_chars[start:end] = [label_map[label]] * (end - start)

        all_y_true.extend(true_chars)
        all_y_pred.extend(pred_chars)

    y_true_labels = [tag_map[tag] for tag in all_y_true]
    y_pred_labels = [tag_map[tag] for tag in all_y_pred]
    labels = ['Neither', 'Ayah', 'Hadith']

    # Print Classification Report
    print("\n📈 Character-Level Classification Report:\n")
    print(classification_report(y_true_labels, y_pred_labels, labels=labels, digits=4))

    # Plot Confusion Matrix
    print("\n📈 Character-Level Confusion Matrix:")
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Character-Level Confusion Matrix')
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
        for _, r in true_spans.iterrows():
             if r['Label'] != 'NoAnnotation':
                true_spans_set.add((int(r['Span_Start']), int(r['Span_End']), r['Label']))

        pred_spans_set = set()
        for _, r in pred_spans.iterrows():
            pred_spans_set.add((int(r['Span_Start']), int(r['Span_End']), r['Span_Type']))

        # Simple overlap check for matching
        matched_true_spans = set()
        matched_pred_spans = set()

        for t_start, t_end, t_label in true_spans_set:
            for p_start, p_end, p_label in pred_spans_set:
                # Check for overlap and matching labels
                if max(t_start, p_start) < min(t_end, p_end) and t_label == p_label:
                    span_info = {
                        'text': text[t_start:t_end],
                        'label': t_label,
                        'length': t_end - t_start
                    }
                    if span_info not in span_stats['TP']:
                        span_stats['TP'].append(span_info)
                    matched_true_spans.add((t_start, t_end, t_label))
                    matched_pred_spans.add((p_start, p_end, p_label))

        # Identify False Negatives (unmatched true spans)
        for t_start, t_end, t_label in true_spans_set - matched_true_spans:
            span_text = text[t_start:t_end]
            error_log.append({'Question_ID': qid, 'Error_Type': 'False Negative', 'Label': t_label, 'Span_Text': span_text})
            span_stats['FN'].append({'text': span_text, 'label': t_label, 'length': len(span_text)})

        # Identify False Positives (unmatched predicted spans)
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

    tp_lengths = [s['length'] for s in span_stats['TP']]
    fp_lengths = [s['length'] for s in span_stats['FP']]
    fn_lengths = [s['length'] for s in span_stats['FN']]

    print("\n📈 Descriptive Statistics for Span Lengths:")
    stats_data = {
        'True Positives': tp_lengths,
        'False Positives': fp_lengths,
        'False Negatives': fn_lengths
    }
    for key, values in stats_data.items():
        if values:
            print(f"  - {key}:")
            print(f"    - Count: {len(values)}")
            print(f"    - Mean: {np.mean(values):.2f}, Median: {np.median(values):.2f}")
            print(f"    - Min: {np.min(values)}, Max: {np.max(values)}")

    # Create DataFrame for plotting
    plot_data = []
    for length in tp_lengths:
        plot_data.append({'Category': 'True Positive', 'Length': length})
    for length in fp_lengths:
        plot_data.append({'Category': 'False Positive', 'Length': length})
    for length in fn_lengths:
        plot_data.append({'Category': 'False Negative', 'Length': length})
    plot_df = pd.DataFrame(plot_data)

    # Plot using Boxplot
    print("\n📈 Boxplot of Span Lengths by Category:")
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='Category', y='Length', data=plot_df)
    plt.title('Distribution of Span Lengths for TP, FP, and FN')
    plt.ylabel('Span Length (characters)')
    plt.xlabel('Category')
    plt.yscale('log') # Use log scale for better visualization if lengths vary widely
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


# --- 4. Main Execution ---

def main():
    """Main function to orchestrate the loading, prediction, and analysis."""
    print("🚀 Starting Advanced Error Analysis Pipeline")
    print("-" * 40)

    # Load Model and Tokenizer
    if not os.path.exists(FINETUNED_MODEL_PATH):
        print(f"❌ FATAL: Model directory not found at {FINETUNED_MODEL_PATH}. Exiting.")
        return
    print(f"🚀 Loading model from: {FINETUNED_MODEL_PATH}")
    model = AutoModelForTokenClassification.from_pretrained(FINETUNED_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
    print("✅ Model and tokenizer loaded successfully.")

    # Load Data
    dev_texts, ground_truth_df = load_dev_data(DEV_XML_PATH, DEV_TSV_PATH)
    if dev_texts is None:
        return

    # Generate Predictions
    predictions_df = predict_on_dev_set(model, tokenizer, dev_texts)

    # Run Analyses
    character_level_analysis(ground_truth_df, predictions_df, dev_texts)
    span_stats = span_level_error_analysis(ground_truth_df, predictions_df, dev_texts)
    analyze_performance_by_length(span_stats)

    print("\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()
