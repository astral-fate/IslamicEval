# Database Lookup Approach - Development Set Evaluation
# Tests the exact-match lookup method on dev set using scoring.py logic

import json
import pandas as pd
import numpy as np
import re
import os
from sklearn.metrics import f1_score, classification_report

# --- Configuration ---
QURAN_JSON_PATH = "/content/drive/MyDrive/FinalIslamic/data/quranic_verses.json"
SIX_HADITH_BOOKS_JSON_PATH = "/content/drive/MyDrive/FinalIslamic/data/six_hadith_books.json"
DEV_XML_PATH = "/content/drive/MyDrive/FinalIslamic/data/dev_SubtaskA.xml"
DEV_TSV_PATH = "/content/drive/MyDrive/FinalIslamic/data/dev_SubtaskA.tsv"

# Set to False if you used inclusive indices in your solution
EXCLUSIVE_INDEX = True

def build_knowledge_base(quran_path, hadith_path):
    """Loads all Ayahs and Hadiths into sets for fast lookup."""
    print("🔧 Building knowledge base from source files...")
    try:
        with open(quran_path, 'r', encoding='utf-8') as f:
            quran_data = json.load(f)
        with open(hadith_path, 'r', encoding='utf-8') as f:
            hadith_data = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Error: Missing required source data file: {e}")
        return None, None

    # Extract Ayah texts
    ayah_set = set()
    for item in quran_data:
        if 'ayah_text' in item and item['ayah_text']:
            ayah_text = item['ayah_text'].strip()
            if len(ayah_text) > 5:  # Filter very short texts
                ayah_set.add(ayah_text)

    # Extract Hadith texts (prioritize Matn field)
    hadith_set = set()
    for item in hadith_data:
        hadith_text = None
        
        # First try Matn field (clean prophetic saying)
        if 'Matn' in item and item['Matn'] and item['Matn'].strip():
            hadith_text = item['Matn'].strip()
        # Fallback to hadithTxt
        elif 'hadithTxt' in item and item['hadithTxt']:
            hadith_text = item['hadithTxt'].strip()
        
        if hadith_text and len(hadith_text) > 10:  # Filter very short texts
            hadith_set.add(hadith_text)

    print(f"📊 Knowledge base built:")
    print(f"  📖 {len(ayah_set):,} unique Ayahs")
    print(f"  📜 {len(hadith_set):,} unique Hadiths")
    print(f"  📚 Total: {len(ayah_set) + len(hadith_set):,} religious texts")
    
    return ayah_set, hadith_set

def load_dev_data_from_xml(xml_path):
    """Loads development data from XML file."""
    print(f"📖 Loading development text data from {xml_path}...")
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = re.compile(r"<Question>.*?<ID>(.*?)</ID>.*?<Response>(.*?)</Response>.*?</Question>", re.DOTALL)
        matches = pattern.findall(content)

        if not matches:
            print("⚠️ Warning: No matches found in the XML file.")
            return {}

        dev_texts = {m[0].strip(): m[1].strip() for m in matches}
        print(f"✅ Successfully loaded {len(dev_texts)} development questions")
        return dev_texts

    except FileNotFoundError:
        print(f"❌ Error: XML file not found at {xml_path}")
        return {}
    except Exception as e:
        print(f"❌ Error parsing XML file: {e}")
        return {}

def predict_with_lookup(dev_texts_dict, ayah_set, hadith_set):
    """Finds spans by exact matching in the knowledge base."""
    print("🔍 Predicting spans using database lookup method...")
    all_predictions = []

    for question_id, text in dev_texts_dict.items():
        if not text:
            all_predictions.append({
                "Question_ID": question_id,
                "Span_Start": 0,
                "Span_End": 0,
                "Span_Type": "No_Spans"
            })
            continue

        spans_found = []

        # Search for Ayah matches (exact string matching)
        for ayah in ayah_set:
            if ayah in text:
                start_pos = 0
                while True:
                    start_index = text.find(ayah, start_pos)
                    if start_index == -1:
                        break
                    end_index = start_index + len(ayah)
                    spans_found.append({
                        "Question_ID": question_id,
                        "Span_Start": start_index,
                        "Span_End": end_index,
                        "Span_Type": "Ayah"
                    })
                    start_pos = start_index + 1

        # Search for Hadith matches (exact string matching)
        for hadith in hadith_set:
            if hadith in text:
                start_pos = 0
                while True:
                    start_index = text.find(hadith, start_pos)
                    if start_index == -1:
                        break
                    end_index = start_index + len(hadith)
                    spans_found.append({
                        "Question_ID": question_id,
                        "Span_Start": start_index,
                        "Span_End": end_index,
                        "Span_Type": "Hadith"
                    })
                    start_pos = start_index + 1

        if spans_found:
            all_predictions.extend(spans_found)
        else:
            all_predictions.append({
                "Question_ID": question_id,
                "Span_Start": 0,
                "Span_End": 0,
                "Span_Type": "No_Spans"
            })

    return pd.DataFrame(all_predictions)

def evaluate_using_scoring_logic(predictions_df, reference_df, qid_response_mapping):
    """Evaluates predictions using the same logic as scoring.py"""
    print("\n🎯 Starting Evaluation using scoring.py logic...")
    print("=" * 60)

    # Debug: Check data consistency
    print(f"📊 Data Overview:")
    print(f"  XML questions: {len(qid_response_mapping)}")
    print(f"  Reference TSV entries: {len(reference_df)}")
    print(f"  Unique questions in reference: {len(reference_df['Question_ID'].unique())}")
    print(f"  Prediction entries: {len(predictions_df)}")
    print(f"  Unique questions in predictions: {len(predictions_df['Question_ID'].unique())}")

    # Validate prediction data
    if any(elem not in ['Ayah', 'Hadith', 'No_Spans'] for elem in predictions_df['Span_Type'].values):
        raise ValueError('Prediction file "Span_Type" column must contain only "Hadith", "Ayah", or "No_Spans".')

    Normal_Text_Tag = 0
    Ayah_Tag = 1
    Hadith_Tag = 2

    # For character-level evaluation
    all_y_true = []
    all_y_pred = []
    
    # For span-level statistics
    span_stats = {
        'total_questions': 0,
        'questions_with_annotations': 0,
        'questions_with_predictions': 0,
        'no_annotation_questions': 0,
        'correct_no_spans': 0,
        'total_true_spans': 0,
        'total_pred_spans': 0,
        'per_question_f1': []
    }

    total_f1 = 0
    count_valid_question = 0
    
    for question_id in reference_df['Question_ID'].unique():
        span_stats['total_questions'] += 1
        
        if question_id not in predictions_df['Question_ID'].values:
            print(f'Question ID {question_id} is missing from the prediction file.')
            continue

        # Check if question exists in XML mapping
        if question_id not in qid_response_mapping:
            print(f'Question ID {question_id} is missing from XML file.')
            continue

        count_valid_question += 1
        question_result = reference_df[reference_df['Question_ID'] == question_id]

        # Handle NoAnnotation case
        if len(question_result) > 0 and question_result['Label'].values[0] == 'NoAnnotation':
            span_stats['no_annotation_questions'] += 1
            pred_spans = predictions_df[predictions_df['Question_ID'] == question_id]
            if len(pred_spans) > 0 and pred_spans['Span_Type'].values[0] == 'No_Spans':
                total_f1 += 1
                span_stats['correct_no_spans'] += 1
                span_stats['per_question_f1'].append(1.0)
            else:
                span_stats['per_question_f1'].append(0.0)
            continue
        
        span_stats['questions_with_annotations'] += 1
        
        # Get response text
        response_text = qid_response_mapping[question_id]
        
        # Initialize prediction character array
        pred_char_array = [Normal_Text_Tag for _ in range(len(response_text))]
        
        pred_result = predictions_df[predictions_df['Question_ID'] == question_id]
        
        # Count prediction spans
        pred_span_count = 0
        if len(pred_result) > 0 and pred_result['Span_Type'].values[0] != 'No_Spans':
            span_stats['questions_with_predictions'] += 1
            for _, row in pred_result.iterrows():
                if row['Span_Type'] != 'No_Spans':
                    pred_span_count += 1
                    span_start, span_end, span_type = int(row['Span_Start']), int(row['Span_End']), row['Span_Type']
                    
                    # Validate span bounds
                    if span_start >= 0 and span_end <= len(response_text):
                        if span_type == 'Ayah':
                            pred_char_array[span_start:span_end] = [Ayah_Tag] * (span_end - span_start)
                        elif span_type == 'Hadith':
                            pred_char_array[span_start:span_end] = [Hadith_Tag] * (span_end - span_start)
        
        span_stats['total_pred_spans'] += pred_span_count

        # Initialize truth character array
        truth_char_array = [Normal_Text_Tag for _ in range(len(response_text))]

        # Count true spans
        true_span_count = 0
        for _, row in question_result.iterrows():
            span_start, span_end, span_type = int(row['Span_Start']), int(row['Span_End']), row['Label']
            
            if span_end <= len(response_text) and span_start >= 0:
                true_span_count += 1
                if span_type == 'Ayah':
                    truth_char_array[span_start:span_end] = [Ayah_Tag] * (span_end - span_start)
                elif span_type == 'Hadith':
                    truth_char_array[span_start:span_end] = [Hadith_Tag] * (span_end - span_start)
        
        span_stats['total_true_spans'] += true_span_count

        # Calculate F1 score for this question
        f1 = f1_score(truth_char_array, pred_char_array, average='macro')
        total_f1 += f1
        span_stats['per_question_f1'].append(f1)
        
        # Collect character-level labels for overall statistics
        all_y_true.extend(truth_char_array)
        all_y_pred.extend(pred_char_array)

    # Calculate final F1 score
    if count_valid_question > 0:
        f1_score_value = total_f1 / count_valid_question
    else:
        f1_score_value = 0.0

    # Generate comprehensive statistics
    generate_comprehensive_stats(all_y_true, all_y_pred, span_stats, f1_score_value)
    
    return f1_score_value

def generate_comprehensive_stats(y_true, y_pred, span_stats, final_f1):
    """Generate comprehensive statistics and classification report"""
    print("\n" + "="*60)
    print("📊 DATABASE LOOKUP EVALUATION STATISTICS")
    print("="*60)
    
    # Map numeric labels to string labels
    label_map = {0: 'Neither', 1: 'Ayah', 2: 'Hadith'}
    y_true_labels = [label_map[label] for label in y_true]
    y_pred_labels = [label_map[label] for label in y_pred]
    
    # Character-level statistics
    print(f"\n📈 CHARACTER-LEVEL STATISTICS")
    print("-" * 40)
    print(f"Total characters evaluated: {len(y_true):,}")
    
    # Distribution analysis
    unique_true, counts_true = np.unique(y_true_labels, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred_labels, return_counts=True)
    
    print(f"\n📊 GROUND TRUTH DISTRIBUTION:")
    for label, count in zip(unique_true, counts_true):
        percentage = count/len(y_true)*100
        print(f"  {label:>8}: {count:>6,} characters ({percentage:>5.1f}%)")
    
    print(f"\n🎯 DATABASE LOOKUP PREDICTION DISTRIBUTION:")
    for label, count in zip(unique_pred, counts_pred):
        percentage = count/len(y_pred)*100
        print(f"  {label:>8}: {count:>6,} characters ({percentage:>5.1f}%)")
    
    # Classification Report
    print(f"\n" + "="*60)
    print("📈 DATABASE LOOKUP CLASSIFICATION REPORT")
    print("="*60)
    
    labels = ['Neither', 'Ayah', 'Hadith']
    report = classification_report(
        y_true_labels, y_pred_labels, 
        labels=labels, 
        zero_division=0, 
        digits=4
    )
    print(report)
    
    # Individual F1 scores
    individual_f1_scores = f1_score(y_true_labels, y_pred_labels, labels=labels, average=None, zero_division=0)
    for label, f1 in zip(labels, individual_f1_scores):
        print(f"F1-Score for {label}: {f1:.4f}")
    
    # Span-level statistics
    print(f"\n" + "="*60)
    print("📋 SPAN-LEVEL STATISTICS")
    print("="*60)
    
    print(f"Total questions processed: {span_stats['total_questions']}")
    print(f"Questions with annotations: {span_stats['questions_with_annotations']}")
    print(f"Questions with predictions: {span_stats['questions_with_predictions']}")
    print(f"'No annotation' questions: {span_stats['no_annotation_questions']}")
    print(f"Correct 'No_Spans' predictions: {span_stats['correct_no_spans']}/{span_stats['no_annotation_questions']}")
    
    print(f"\nSpan counts:")
    print(f"  Total true spans: {span_stats['total_true_spans']}")
    print(f"  Total predicted spans: {span_stats['total_pred_spans']}")
    
    if span_stats['total_true_spans'] > 0:
        span_recall = span_stats['total_pred_spans'] / span_stats['total_true_spans']
        print(f"  Span recall (rough): {span_recall:.4f}")
    
    # Per-question F1 statistics
    if span_stats['per_question_f1']:
        per_q_f1 = np.array(span_stats['per_question_f1'])
        print(f"\nPer-question F1 statistics:")
        print(f"  Mean F1: {np.mean(per_q_f1):.4f}")
        print(f"  Median F1: {np.median(per_q_f1):.4f}")
        print(f"  Std F1: {np.std(per_q_f1):.4f}")
        print(f"  Min F1: {np.min(per_q_f1):.4f}")
        print(f"  Max F1: {np.max(per_q_f1):.4f}")
        
        # F1 distribution
        perfect_questions = np.sum(per_q_f1 == 1.0)
        zero_questions = np.sum(per_q_f1 == 0.0)
        print(f"  Questions with perfect F1 (1.0): {perfect_questions}")
        print(f"  Questions with zero F1 (0.0): {zero_questions}")
    
    # Summary
    print(f"\n" + "="*60)
    print("🎯 DATABASE LOOKUP FINAL SUMMARY")
    print("="*60)
    print(f"**Final Macro-Averaged F1 Score: {final_f1:.4f}**")
    
    # Calculate micro-averaged F1 for comparison
    micro_f1 = f1_score(y_true_labels, y_pred_labels, labels=labels, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true_labels, y_pred_labels, labels=labels, average='weighted', zero_division=0)
    print(f"Micro-Averaged F1 Score: {micro_f1:.4f}")
    print(f"Weighted-Averaged F1 Score: {weighted_f1:.4f}")
    print("="*60)

# --- Main Execution ---
def main():
    print("🔍 DATABASE LOOKUP APPROACH - DEVELOPMENT SET EVALUATION")
    print("=" * 60)
    print("🎯 Method: Exact string matching against knowledge base")
    print("📊 Task: Evaluate lookup performance on dev set")
    print("=" * 60)

    # Build knowledge base
    ayah_set, hadith_set = build_knowledge_base(QURAN_JSON_PATH, SIX_HADITH_BOOKS_JSON_PATH)
    if ayah_set is None:
        print("❌ Failed to build knowledge base. Exiting.")
        return

    # Load development data
    dev_texts_dict = load_dev_data_from_xml(DEV_XML_PATH)
    if not dev_texts_dict:
        print("❌ Failed to load development texts. Exiting.")
        return

    try:
        ground_truth_df = pd.read_csv(DEV_TSV_PATH, sep='\t')
        print(f"✅ Successfully loaded {len(ground_truth_df)} ground truth annotations")
    except FileNotFoundError:
        print(f"❌ Error: Ground truth file not found at {DEV_TSV_PATH}")
        return

    # Generate predictions using database lookup
    predictions_df = predict_with_lookup(dev_texts_dict, ayah_set, hadith_set)
    
    print(f"\n📊 Database Lookup Predictions:")
    pred_counts = predictions_df['Span_Type'].value_counts()
    for span_type, count in pred_counts.items():
        print(f"  {span_type}: {count} spans")

    print(f"\n📋 Sample Predictions:")
    sample_preds = predictions_df[predictions_df['Span_Type'] != 'No_Spans'].head(10)
    for _, row in sample_preds.iterrows():
        qid = row['Question_ID']
        text = dev_texts_dict.get(qid, "")
        span_text = text[row['Span_Start']:row['Span_End']] if text else ""
        print(f"  {row['Question_ID']}: {row['Span_Type']} - \"{span_text[:50]}...\"")

    # Evaluate predictions
    final_f1 = evaluate_using_scoring_logic(predictions_df, ground_truth_df, dev_texts_dict)

    print(f"\n🎉 DATABASE LOOKUP EVALUATION COMPLETED!")
    print(f"🎯 Final Macro F1-Score: {final_f1:.4f}")

    # Save results
    predictions_df.to_csv('/content/lookup_dev_predictions.tsv', sep='\t', index=False, header=False)
    print("📁 Predictions saved to: /content/lookup_dev_predictions.tsv")

if __name__ == "__main__":
    main()
