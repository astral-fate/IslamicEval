# 66.97% - Modified for Google Colab, Development Set Evaluation, and F1-Score Calculation

import json
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
import re
import os
import zipfile
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# --- 1. Configuration ---

# --- Input Data Paths (Updated for Google Colab and Dev Set) ---
# Path to your saved model files (the directory containing config.json, etc.)
MODEL_PATH = "/content/drive/MyDrive/FinalIslamic/best model 66"

# Path to the development set annotations
DEV_TSV_PATH = "/content/drive/MyDrive/FinalIslamic/data/dev_SubtaskA.tsv"

# You will need the XML file that contains the full text for the dev set questions.
# Please update this path to the correct location of your dev XML file.
DEV_XML_PATH = "/content/drive/MyDrive/FinalIslamic/data/dev_SubtaskA.xml"


# --- 2. Data Loading and Preprocessing ---

def load_dev_data_from_xml(xml_path):
    """Loads development data from the provided XML-like file format."""
    print(f"Loading and parsing development text data from {xml_path}...")
    try:
        # This regex assumes a similar structure to the test file. Adjust if needed.
        with open(xml_path, 'r', encoding='utf-8') as f: content = f.read()
        pattern = re.compile(r"<Question>.*?<ID>(.*?)</ID>.*?<Response>(.*?)</Response>.*?</Question>", re.DOTALL)
        matches = pattern.findall(content)
        if not matches:
            print("Warning: No matches found in the XML file. Please check the file path and structure.")
            return {}
        # Create a dictionary mapping Question_ID to its full text
        return {m[0].strip(): m[1].strip() for m in matches}
    except FileNotFoundError:
        print(f"Error: The file at '{xml_path}' was not found.")
        print("Please ensure you have uploaded the development XML file and the path is correct.")
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
            # Append a record for questions with no text to ensure they are in the results
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


# --- 4. Evaluation ---

def evaluate_predictions(predictions_df, ground_truth_df, dev_texts_dict):
    """Calculates the F1-score based on the predictions and ground truth at character level."""
    print("\n--- Starting Character-Level Evaluation ---")
    y_true = []
    y_pred = []

    all_qids = set(ground_truth_df['Question_ID']) | set(predictions_df['Question_ID'])

    for qid in sorted(list(all_qids)):
        text = dev_texts_dict.get(qid, "")
        if not text:
            continue

        # Initialize character-level labels for the text (Neither/Ayah/Hadith)
        true_labels = ['Neither'] * len(text)
        pred_labels = ['Neither'] * len(text)

        # Populate true labels from the golden dataset
        gt_spans = ground_truth_df[ground_truth_df['Question_ID'] == qid]
        for _, row in gt_spans.iterrows():
            start, end, label = int(row['Span_Start']), int(row['Span_End']), row['Label']
            if end > 0: # Ignore 'No_Spans' entries in ground truth
                for i in range(start, min(end, len(true_labels))):
                    true_labels[i] = label

        # Populate predicted labels
        pred_spans = predictions_df[predictions_df['Question_ID'] == qid]
        for _, row in pred_spans.iterrows():
            start, end, label = int(row['Span_Start']), int(row['Span_End']), row['Span_Type']
            if label != 'No_Spans' and end > 0:
                for i in range(start, min(end, len(pred_labels))):
                    pred_labels[i] = label

        y_true.extend(true_labels)
        y_pred.extend(pred_labels)

    # --- Calculate and Display Character-Level Metrics ---
    print(f"Total characters evaluated: {len(y_true)}")
    
    # Count distribution
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    
    print("\nGround Truth Distribution:")
    for label, count in zip(unique_true, counts_true):
        print(f"  {label}: {count} characters ({count/len(y_true)*100:.2f}%)")
    
    print("\nPrediction Distribution:")
    for label, count in zip(unique_pred, counts_pred):
        print(f"  {label}: {count} characters ({count/len(y_pred)*100:.2f}%)")

    # Calculate metrics for each class
    labels = ['Neither', 'Ayah', 'Hadith']
    
    # Generate detailed classification report
    print("\n" + "="*60)
    print("CHARACTER-LEVEL CLASSIFICATION REPORT")
    print("="*60)
    
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=4)
    print(report)

    # Calculate individual F1 scores for each class
    f1_scores = []
    for label in labels:
        f1 = f1_score(y_true, y_pred, labels=[label], average='macro', zero_division=0)
        f1_scores.append(f1)
        print(f"F1-Score for {label}: {f1:.4f}")

    # Calculate the final Macro-Averaged F1 Score
    macro_f1 = np.mean(f1_scores)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"**Macro-Averaged F1 Score (Character-Level): {macro_f1:.4f}**")
    print("="*60)

    # Also calculate micro-averaged F1 for comparison
    micro_f1 = f1_score(y_true, y_pred, labels=labels, average='micro', zero_division=0)
    print(f"Micro-Averaged F1 Score (Character-Level): {micro_f1:.4f}")

    print("\n--- Evaluation Finished ---")
    
    return macro_f1


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model directory not found at '{MODEL_PATH}'")
        print("Please make sure the path is correct and your Google Drive is mounted.")
        exit()

    print("Loading fine-tuned model for prediction...")
    # These labels must match what the model was trained on
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

    # Load the development data (both text and ground truth labels)
    dev_texts_dict = load_dev_data_from_xml(DEV_XML_PATH)
    try:
        ground_truth_df = pd.read_csv(DEV_TSV_PATH, sep='\t')
        print(f"Successfully loaded {len(ground_truth_df)} ground truth annotations.")
    except FileNotFoundError:
        print(f"Error: The ground truth file was not found at '{DEV_TSV_PATH}'")
        print("Please ensure the path is correct.")
        exit()


    if dev_texts_dict:
        # Get predictions from the model
        predictions_df = predict_on_dev_data(model, tokenizer, dev_texts_dict, label_list)
        print(f"\nGenerated {len(predictions_df)} predictions.")
        print("Sample Predictions:")
        print(predictions_df.head())

        # Evaluate the predictions against the golden labels
        final_macro_f1 = evaluate_predictions(predictions_df, ground_truth_df, dev_texts_dict)

        print(f"\n🎯 FINAL MACRO-AVERAGED F1 SCORE: {final_macro_f1:.4f}")
        print("\n--- Script Finished ---")
    else:
        print("\n--- Script Aborted: Could not load development text data. ---")
