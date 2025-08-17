#
# Script to evaluate a fine-tuned model on the development set
# and generate a detailed performance report.
#

import json
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)
from sklearn.metrics import f1_score, classification_report
import re
import os

# --- 1. Configuration ---
# ⚠️ UPDATE THESE PATHS to point to your files.

# Path to your saved model directory (containing config.json, pytorch_model.bin, etc.)
MODEL_PATH = "/content/drive/MyDrive/FinalIslamic/arabert_finetuned_model_v_temp4"

# Path to the development set ground truth annotations
DEV_TSV_PATH = "/content/drive/MyDrive/FinalIslamic/data/dev_SubtaskA.tsv"

# Path to the development set XML file containing the full text for each question
DEV_XML_PATH = "/content/drive/MyDrive/FinalIslamic/data/dev_SubtaskA.xml"


# --- 2. Data Loading ---

def load_dev_data_from_xml(xml_path):
    """Loads development data texts from the XML file."""
    print(f"Loading and parsing development text data from {xml_path}...")
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        pattern = re.compile(r"<Question>.*?<ID>(.*?)</ID>.*?<Response>(.*?)</Response>.*?</Question>", re.DOTALL)
        matches = pattern.findall(content)
        if not matches:
            print("Warning: No matches found in the XML file. Check the file path and structure.")
            return {}
        # Create a dictionary mapping Question_ID to its full text
        return {m[0].strip(): m[1].strip() for m in matches}
    except FileNotFoundError:
        print(f"Error: The file at '{xml_path}' was not found.")
        return {}
    except Exception as e:
        print(f"Error parsing dev XML file: {e}")
        return {}


# --- 3. Prediction ---

def predict_on_dev_data(model, tokenizer, dev_texts_dict, label_list):
    """Predicts spans on the loaded development data."""
    print("Predicting spans on the development set...")
    model.eval()
    device = model.device
    all_predictions = []

    for qid, text in dev_texts_dict.items():
        if not text:
            all_predictions.append({"Question_ID": qid, "Span_Start": 0, "Span_End": 0, "Span_Type": "No_Spans"})
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
            if word_id is None:
                continue

            if label.startswith('B-'):
                if current_span:
                    spans.append(current_span)
                cs = inputs.token_to_chars(i)
                current_span = {'type': label[2:], 'start': cs.start, 'end': cs.end}
            elif label.startswith('I-') and current_span and current_span['type'] == label[2:]:
                cs = inputs.token_to_chars(i)
                current_span['end'] = cs.end
            elif current_span:
                spans.append(current_span)
                current_span = None

        if current_span:
            spans.append(current_span)

        if spans:
            for span in spans:
                all_predictions.append({
                    "Question_ID": qid,
                    "Span_Start": span['start'],
                    "Span_End": span['end'],
                    "Span_Type": span['type']
                })
        else:
            all_predictions.append({
                "Question_ID": qid,
                "Span_Start": 0,
                "Span_End": 0,
                "Span_Type": "No_Spans"
            })

    return pd.DataFrame(all_predictions)


# --- 4. Evaluation and EDA ---

def evaluate_predictions(predictions_df, ground_truth_df, dev_texts_dict):
    """Calculates F1-score and provides EDA on predictions vs. ground truth."""
    print("\n--- Starting Character-Level Evaluation & EDA ---")
    y_true = []
    y_pred = []

    all_qids = set(ground_truth_df['Question_ID']) | set(predictions_df['Question_ID'])

    for qid in sorted(list(all_qids)):
        text = dev_texts_dict.get(qid, "")
        if not text:
            continue

        # Initialize character-level labels for the text
        true_labels = ['Neither'] * len(text)
        pred_labels = ['Neither'] * len(text)

        # Populate true labels from the ground truth
        gt_spans = ground_truth_df[ground_truth_df['Question_ID'] == qid]
        for _, row in gt_spans.iterrows():
            start, end, label = int(row['Span_Start']), int(row['Span_End']), row['Label']
            if end > 0:
                for i in range(start, min(end, len(true_labels))):
                    true_labels[i] = label

        # Populate predicted labels from model output
        pred_spans = predictions_df[predictions_df['Question_ID'] == qid]
        for _, row in pred_spans.iterrows():
            start, end, label = int(row['Span_Start']), int(row['Span_End']), row['Span_Type']
            if label != 'No_Spans' and end > 0:
                for i in range(start, min(end, len(pred_labels))):
                    pred_labels[i] = label

        y_true.extend(true_labels)
        y_pred.extend(pred_labels)

    # --- EDA: Distribution Analysis ---
    print(f"\nTotal characters evaluated: {len(y_true)}")
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)

    print("\n--- Ground Truth Distribution (EDA) ---")
    for label, count in zip(unique_true, counts_true):
        print(f"  {label:<10}: {count:>8} characters ({count/len(y_true)*100:.2f}%)")

    print("\n--- Prediction Distribution (EDA) ---")
    for label, count in zip(unique_pred, counts_pred):
        print(f"  {label:<10}: {count:>8} characters ({count/len(y_pred)*100:.2f}%)")

    # --- Performance Metrics ---
    labels = ['Neither', 'Ayah', 'Hadith']
    print("\n" + "="*60)
    print("      CHARACTER-LEVEL CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=4)
    print(report)

    # --- Final F1 Score Calculation ---
    f1_scores = [f1_score(y_true, y_pred, labels=[label], average='macro', zero_division=0) for label in labels]
    macro_f1 = np.mean(f1_scores)

    print("="*60)
    print(f"** Macro-Averaged F1 Score (Ayah, Hadith, Neither): {macro_f1:.4f} **")
    print("="*60)

    return macro_f1


# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model directory not found at '{MODEL_PATH}'")
        exit()

    print("Loading fine-tuned model for prediction...")
    # These labels MUST match the labels the model was trained on
    label_list = ['O', 'B-Ayah', 'I-Ayah', 'B-Hadith', 'I-Hadith']

    try:
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Successfully loaded model and tokenizer. Using device: {device}")
    except Exception as e:
        print(f"Error loading the model or tokenizer from {MODEL_PATH}: {e}")
        exit()

    # Load development data (texts and ground truth labels)
    dev_texts_dict = load_dev_data_from_xml(DEV_XML_PATH)
    try:
        ground_truth_df = pd.read_csv(DEV_TSV_PATH, sep='\t')
        print(f"Successfully loaded {len(ground_truth_df)} ground truth annotations.")
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at '{DEV_TSV_PATH}'")
        exit()

    if dev_texts_dict:
        # Run prediction and evaluation
        predictions_df = predict_on_dev_data(model, tokenizer, dev_texts_dict, label_list)
        print(f"\nGenerated {len(predictions_df)} predictions.")
        print("\n--- Sample Predictions ---")
        print(predictions_df.head())

        final_macro_f1 = evaluate_predictions(predictions_df, ground_truth_df, dev_texts_dict)

        print(f"\n🎯 FINAL MACRO-AVERAGED F1 SCORE: {final_macro_f1:.4f}")
        print("\n--- Script Finished ---")
    else:
        print("\n--- Script Aborted: Could not load development text data. ---")
