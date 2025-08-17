#
# SCRIPT: evaluate_finetuned_model.py
#
# Purpose: Evaluate the performance of a fine-tuned transformer model on the development set
#          using the official character-level scoring logic.
#

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re
import os
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# --- Configuration ---
# Path to your fine-tuned model directory
FINETUNED_MODEL_PATH = "/content/drive/MyDrive/FinalIslamic/arabert_finetuned_model_best_v2"

# Paths to the development set files
DEV_XML_PATH = "/content/drive/MyDrive/FinalIslamic/data/dev_SubtaskA.xml"
DEV_TSV_PATH = "/content/drive/MyDrive/FinalIslamic/data/dev_SubtaskA.tsv"

def load_dev_data_from_xml(xml_path):
    """Loads development data from XML file."""
    print(f"📖 Loading development text data from {xml_path}...")
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        pattern = re.compile(r"<Question>.*?<ID>(.*?)</ID>.*?<Response>(.*?)</Response>.*?</Question>", re.DOTALL)
        matches = pattern.findall(content)
        dev_texts = {m[0].strip(): m[1].strip() for m in matches}
        print(f"✅ Successfully loaded {len(dev_texts)} development questions")
        return dev_texts
    except FileNotFoundError:
        print(f"❌ Error: XML file not found at {xml_path}")
        return {}

def predict_with_finetuned_model(model, tokenizer, dev_texts_dict):
    """Generates predictions using the fine-tuned transformer model."""
    print("🤖 Predicting spans using fine-tuned model...")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    label_list = list(model.config.id2label.values())
    all_predictions = []

    for question_id, text in tqdm(dev_texts_dict.items(), desc="Predicting on Dev Set"):
        if not text or not text.strip():
            all_predictions.append({"Question_ID": question_id, "Span_Start": 0, "Span_End": 0, "Span_Type": "No_Spans"})
            continue

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits

        preds = torch.argmax(logits, dim=2)[0].cpu().numpy()

        spans = []
        current_span = None
        for i, pred_id in enumerate(preds):
            label = label_list[pred_id]
            word_id = inputs.word_ids(batch_index=0)[i]

            if word_id is None:  # Skip special tokens
                continue

            if label.startswith('B-'):
                if current_span:  # Close previous span if it exists
                    spans.append(current_span)

                # Start a new span
                char_span = inputs.token_to_chars(i)
                current_span = {'type': label[2:], 'start': char_span.start, 'end': char_span.end}

            elif label.startswith('I-') and current_span and current_span['type'] == label[2:]:
                # Extend the current span
                char_span = inputs.token_to_chars(i)
                current_span['end'] = char_span.end

            else:  # 'O' label or mismatched 'I-' label
                if current_span:
                    spans.append(current_span)
                current_span = None

        if current_span:  # Add the last span if the text ends with it
            spans.append(current_span)

        if spans:
            for span in spans:
                all_predictions.append({
                    "Question_ID": question_id,
                    "Span_Start": span['start'],
                    "Span_End": span['end'],
                    "Span_Type": span['type']
                })
        else:
            all_predictions.append({
                "Question_ID": question_id,
                "Span_Start": 0,
                "Span_End": 0,
                "Span_Type": "No_Spans"
            })

    return pd.DataFrame(all_predictions)

def evaluate_using_scoring_logic(predictions_df, reference_df, qid_response_mapping):
    """Evaluates predictions using the same logic as the official scoring script."""
    print("\n🎯 Starting Evaluation using official scoring logic...")
    print("=" * 60)

    Normal_Text_Tag, Ayah_Tag, Hadith_Tag = 0, 1, 2
    all_y_true, all_y_pred = [], []

    span_stats = {
        'total_questions': 0, 'questions_with_annotations': 0, 'questions_with_predictions': 0,
        'no_annotation_questions': 0, 'correct_no_spans': 0, 'total_true_spans': 0,
        'total_pred_spans': 0, 'per_question_f1': []
    }

    total_f1, count_valid_question = 0, 0

    for question_id in reference_df['Question_ID'].unique():
        span_stats['total_questions'] += 1

        if question_id not in predictions_df['Question_ID'].values or question_id not in qid_response_mapping:
            continue

        count_valid_question += 1
        question_result = reference_df[reference_df['Question_ID'] == question_id]

        if len(question_result) > 0 and question_result['Label'].values[0] == 'NoAnnotation':
            span_stats['no_annotation_questions'] += 1
            pred_spans = predictions_df[predictions_df['Question_ID'] == question_id]
            if len(pred_spans) > 0 and pred_spans['Span_Type'].values[0] == 'No_Spans':
                total_f1 += 1.0
                span_stats['correct_no_spans'] += 1
                span_stats['per_question_f1'].append(1.0)
            else:
                span_stats['per_question_f1'].append(0.0)
            continue

        span_stats['questions_with_annotations'] += 1
        response_text = qid_response_mapping[question_id]

        # Create prediction character array
        pred_char_array = [Normal_Text_Tag] * len(response_text)
        pred_result = predictions_df[predictions_df['Question_ID'] == question_id]

        pred_span_count = 0
        if len(pred_result) > 0 and pred_result['Span_Type'].values[0] != 'No_Spans':
            span_stats['questions_with_predictions'] += 1
            for _, row in pred_result.iterrows():
                pred_span_count += 1
                start, end, type = int(row['Span_Start']), int(row['Span_End']), row['Span_Type']
                if start >= 0 and end <= len(response_text):
                    tag = Ayah_Tag if type == 'Ayah' else Hadith_Tag
                    pred_char_array[start:end] = [tag] * (end - start)
        span_stats['total_pred_spans'] += pred_span_count

        # Create truth character array
        truth_char_array = [Normal_Text_Tag] * len(response_text)
        true_span_count = 0
        for _, row in question_result.iterrows():
            true_span_count += 1
            start, end, type = int(row['Span_Start']), int(row['Span_End']), row['Label']
            if end <= len(response_text) and start >= 0:
                tag = Ayah_Tag if type == 'Ayah' else Hadith_Tag
                truth_char_array[start:end] = [tag] * (end - start)
        span_stats['total_true_spans'] += true_span_count

        f1 = f1_score(truth_char_array, pred_char_array, average='macro', zero_division=0)
        total_f1 += f1
        span_stats['per_question_f1'].append(f1)

        all_y_true.extend(truth_char_array)
        all_y_pred.extend(pred_char_array)

    f1_score_value = total_f1 / count_valid_question if count_valid_question > 0 else 0.0
    generate_comprehensive_stats(all_y_true, all_y_pred, span_stats, f1_score_value)
    return f1_score_value

def generate_comprehensive_stats(y_true, y_pred, span_stats, final_f1):
    """Generates and prints the detailed EDA and evaluation statistics."""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE EVALUATION STATISTICS (EDA)")
    print("="*60)

    label_map = {0: 'Neither', 1: 'Ayah', 2: 'Hadith'}
    y_true_labels = [label_map[label] for label in y_true]
    y_pred_labels = [label_map[label] for label in y_pred]

    print(f"\n📈 CHARACTER-LEVEL CLASSIFICATION REPORT")
    print("-" * 60)
    labels = ['Neither', 'Ayah', 'Hadith']
    print(classification_report(y_true_labels, y_pred_labels, labels=labels, zero_division=0, digits=4))

    print(f"\n📋 SPAN-LEVEL STATISTICS")
    print("-" * 60)
    print(f"Total questions processed: {span_stats['total_questions']}")
    print(f"Questions with ground truth annotations: {span_stats['questions_with_annotations']}")
    print(f"'No annotation' questions: {span_stats['no_annotation_questions']}")
    print(f"Correct 'No_Spans' predictions: {span_stats['correct_no_spans']}/{span_stats['no_annotation_questions']}")
    print(f"Span counts (True vs. Predicted): {span_stats['total_true_spans']} vs. {span_stats['total_pred_spans']}")

    if span_stats['per_question_f1']:
        per_q_f1 = np.array(span_stats['per_question_f1'])
        print(f"\nPer-question F1 statistics:")
        print(f"  Mean F1: {np.mean(per_q_f1):.4f} | Median F1: {np.median(per_q_f1):.4f} | Std Dev: {np.std(per_q_f1):.4f}")
        print(f"  Questions with perfect F1 (1.0): {np.sum(per_q_f1 == 1.0)}")
        print(f"  Questions with zero F1 (0.0): {np.sum(per_q_f1 == 0.0)}")

    print("\n" + "="*60)
    print("🎯 FINAL SUMMARY")
    print("="*60)
    print(f"**Final Macro-Averaged F1 Score: {final_f1:.4f}**")
    print("="*60)

# --- Main Execution ---
def main():
    print("🔍 Fine-Tuned Model Evaluation on Development Set")
    print("=" * 60)

    # Load model and tokenizer
    print(f"🚀 Loading fine-tuned model from: {FINETUNED_MODEL_PATH}")
    if not os.path.exists(FINETUNED_MODEL_PATH):
        print("❌ Model directory not found. Please ensure the path is correct.")
        return

    model = AutoModelForTokenClassification.from_pretrained(FINETUNED_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)

    # Load development data
    dev_texts_dict = load_dev_data_from_xml(DEV_XML_PATH)
    if not dev_texts_dict:
        return

    try:
        ground_truth_df = pd.read_csv(DEV_TSV_PATH, sep='\t')
        print(f"✅ Successfully loaded {len(ground_truth_df)} ground truth annotations")
    except FileNotFoundError:
        print(f"❌ Error: Ground truth file not found at {DEV_TSV_PATH}")
        return

    # Generate predictions
    predictions_df = predict_with_finetuned_model(model, tokenizer, dev_texts_dict)

    # Evaluate predictions
    final_f1 = evaluate_using_scoring_logic(predictions_df, ground_truth_df, dev_texts_dict)

    print(f"\n🎉 EVALUATION COMPLETED!")
    print(f"🎯 Final Macro F1-Score on the development set: {final_f1:.4f}")

    # Save dev predictions for inspection
    output_path = '/content/finetuned_model_dev_predictions.tsv'
    predictions_df.to_csv(output_path, sep='\t', index=False, header=True)
    print(f"📁 Development set predictions saved to: {output_path}")

if __name__ == "__main__":
    main()
