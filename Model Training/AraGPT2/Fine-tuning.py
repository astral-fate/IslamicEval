# -*- coding: utf-8 -*-
"""
This script fine-tunes the AraBERT v2 model for a token classification task.
It learns to identify spans of Ayahs and Hadiths by training on data created
using a rule-based template system. It then predicts on the official test set.

It requires the following data files:
1. quran.json and six_hadith_books.json: For the training phase.
2. test_SubtaskA.xml: The official test file for the final prediction phase.
"""

import json
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset, DatasetDict
import re
import os
import zipfile
import random
import time

# --- 1. Configuration ---

MODEL_NAME = "aubmindlab/bert-base-arabertv2"
OUTPUT_DIR = "/content/drive/MyDrive/FinalIslamic/arabert_finetuned_model_v_temp4"

# --- Input Data Paths ---
QURAN_JSON_PATH = "/content/drive/MyDrive/FinalIslamic/data/quran.json"
SIX_HADITH_BOOKS_JSON_PATH = "/content/drive/MyDrive/FinalIslamic/data/six_hadith_books.json"
TEST_XML_PATH = "/content/drive/MyDrive/FinalIslamic/data/test_SubtaskA.xml"

# --- Output Paths ---
SUBMISSION_TSV_PATH = "/content/drive/MyDrive/FinalIslamic/data/submission.tsv"
SUBMISSION_ZIP_PATH = "/content/drive/MyDrive/FinalIslamic/data/submission.zip"


# --- 2. Data Loading and Preprocessing ---

def create_dummy_files():
    """Creates dummy data files for demonstration."""
    if not os.path.exists(QURAN_JSON_PATH):
        print(f"Creating dummy '{QURAN_JSON_PATH}'...")
        quran_data = [{"ayah_text": "بِسْمِ اللَّهِ الرَّحْمَـٰنِ الرَّحِيمِ"}]
        with open(QURAN_JSON_PATH, 'w', encoding='utf-8') as f: json.dump(quran_data, f, ensure_ascii=False, indent=4)

    if not os.path.exists(SIX_HADITH_BOOKS_JSON_PATH):
        print(f"Creating dummy '{SIX_HADITH_BOOKS_JSON_PATH}'...")
        six_hadith_data = [{"Matn": "من سلك طريقا يلتمس فيه علما سهل الله له به طريقا إلى الجنة"}]
        with open(SIX_HADITH_BOOKS_JSON_PATH, 'w', encoding='utf-8') as f: json.dump(six_hadith_data, f, ensure_ascii=False, indent=4)

    if not os.path.exists(TEST_XML_PATH):
        print(f"Creating dummy '{TEST_XML_PATH}'...")
        xml_content = """<Question><ID>A-Q001</ID><Response>نعم، يُذكر في القرآن الكريم: {وَإِنَّكَ لَعَلَىٰ خُلُقٍ عَظِيمٍ}.</Response></Question>"""
        with open(TEST_XML_PATH, 'w', encoding='utf-8') as f: f.write(xml_content)

def load_and_prepare_training_data(quran_path, six_books_path, tokenizer):
    """
    Creates a high-quality training dataset using a rule-based template system.
    """
    print("Loading and preparing training data using rule-based templates...")
    try:
        with open(quran_path, 'r', encoding='utf-8') as f: quran_data = json.load(f)
        with open(six_books_path, 'r', encoding='utf-8') as f: six_books_data = json.load(f)
    except Exception as e:
        print(f"Error loading source data files: {e}.")
        return None, None

    ayah_texts = [item['ayah_text'] for item in quran_data if 'ayah_text' in item]
    # MODIFIED: Use 'Matn' field and filter out null/empty values
    hadith_texts = [item['Matn'] for item in six_books_data if 'Matn' in item and item['Matn']]

    # --- Filter out extremely long texts to create balanced examples ---
    MAX_TEXT_LENGTH = 1500 # Characters
    ayah_texts = [t for t in ayah_texts if len(t) < MAX_TEXT_LENGTH]
    hadith_texts = [t for t in hadith_texts if len(t) < MAX_TEXT_LENGTH]
    print(f"Loaded and filtered {len(ayah_texts)} Ayahs and {len(hadith_texts)} Hadiths.")

    processed_data = []
    label_list = ['O', 'B-Ayah', 'I-Ayah', 'B-Hadith', 'I-Hadith']
    label_to_id = {l: i for i, l in enumerate(label_list)}

    # --- Rule-Based Templates for High-Quality Data ---
    prefixes = ["", "قال الله تعالى:", "كما ورد في قوله:", "والدليل على ذلك هو قول النبي صلى الله عليه وسلم:"]
    suffixes = ["", "صدق الله العظيم.", "وهذا يوضح أهمية الموضوع.", "رواه البخاري."]
    neutral_sentences = ["وبناء على ذلك، يمكننا أن نستنتج.", "اختلف العلماء في هذه المسألة."]

    def create_example(text, label_type):
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)

        # Construct a clean, well-formed sentence with proper spacing
        if random.random() > 0.5:
            context = random.choice(neutral_sentences)
            full_text = f"{prefix} {context} {text} {suffix}"
        else:
            full_text = f"{prefix} {text} {suffix}"

        full_text = full_text.strip()

        char_start = full_text.find(text)
        if char_start == -1: return

        char_end = char_start + len(text)

        tokenized_input = tokenizer(full_text, truncation=True, max_length=512)
        labels = [label_to_id['O']] * len(tokenized_input['input_ids'])

        start_token = tokenized_input.char_to_token(char_start)
        end_token = tokenized_input.char_to_token(char_end - 1)

        if start_token is not None and end_token is not None:
            labels[start_token] = label_to_id[f'B-{label_type}']
            for i in range(start_token + 1, end_token + 1):
                if i < len(labels): labels[i] = label_to_id[f'I-{label_type}']

        final_labels = []
        word_ids = tokenized_input.word_ids()
        for i, word_id in enumerate(word_ids):
            if word_id is None or (i > 0 and word_id == word_ids[i - 1]):
                final_labels.append(-100)
            else:
                final_labels.append(labels[i])

        processed_data.append({
            "input_ids": tokenized_input['input_ids'],
            "attention_mask": tokenized_input['attention_mask'],
            "labels": final_labels
        })

    print("Generating training examples...")
    for ayah in ayah_texts: create_example(ayah, 'Ayah')
    for hadith in hadith_texts: create_example(hadith, 'Hadith')
    print(f"✅ Generated {len(processed_data)} examples.")

    if not processed_data:
        print("Warning: No training data could be processed.")
        return None, None

    dataset = Dataset.from_list(processed_data)
    return DatasetDict({'train': dataset}), label_list


# --- 3. Fine-Tuning ---
def fine_tune_model(dataset_dict, label_list):
    """Initializes and fine-tunes the AraBERT model."""
    print("Starting fine-tuning process...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for i, l in enumerate(label_list)}

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Model fine-tuned and saved to {OUTPUT_DIR}")


# --- 4. Prediction ---
def load_test_data_from_xml(xml_path):
    """Loads test data from the provided XML-like file format."""
    print(f"Loading and parsing test data from {xml_path}...")
    try:
        with open(xml_path, 'r', encoding='utf-8') as f: content = f.read()
        pattern = re.compile(r"<Question>.*?<ID>(.*?)</ID>.*?<Response>(.*?)</Response>.*?</Question>", re.DOTALL)
        matches = pattern.findall(content)
        if not matches: return []
        return [{'Question_ID': m[0].strip(), 'Text': m[1].strip()} for m in matches]
    except Exception as e:
        print(f"Error parsing test file: {e}")
        return []

def predict_on_test_data(model, tokenizer, test_data, label_list):
    """Predicts spans on the loaded test data."""
    print("Predicting spans on the test set...")
    model.eval()
    device = model.device
    all_predictions = []
    for item in test_data:
        qid, text = item["Question_ID"], item["Text"]
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
            if word_id is None: continue
            if label.startswith('B-'):
                if current_span: spans.append(current_span)
                cs = inputs.token_to_chars(i)
                current_span = {'type': label[2:], 'start': cs.start, 'end': cs.end}
            elif label.startswith('I-') and current_span and current_span['type'] == label[2:]:
                cs = inputs.token_to_chars(i)
                current_span['end'] = cs.end
            elif current_span:
                spans.append(current_span)
                current_span = None
        if current_span: spans.append(current_span)
        if spans:
            for span in spans: all_predictions.append({"Question_ID": qid, "Span_Start": span['start'], "Span_End": span['end'], "Span_Type": span['type']})
        else:
            all_predictions.append({"Question_ID": qid, "Span_Start": 0, "Span_End": 0, "Span_Type": "No_Spans"})
    return all_predictions


# --- 5. Submission File Generation ---
def generate_submission_file(predictions, output_path, zip_path):
    """Generates the final TSV submission file and zips it."""
    if not predictions:
        print("No predictions to save.")
        return
    print(f"Generating submission file at {output_path}...")
    df = pd.DataFrame(predictions)[["Question_ID", "Span_Start", "Span_End", "Span_Type"]]
    df.to_csv(output_path, sep='\t', index=False, header=False)
    with zipfile.ZipFile(zip_path, 'w') as zf: zf.write(output_path, os.path.basename(output_path))
    print("Submission zip created successfully.")


# --- Main Execution ---
if __name__ == "__main__":
    create_dummy_files()
    label_list = ['O', 'B-Ayah', 'I-Ayah', 'B-Hadith', 'I-Hadith']
    if not os.path.exists(OUTPUT_DIR):
        print("No fine-tuned model found. Starting data generation and training...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        dataset_dict, loaded_label_list = load_and_prepare_training_data(QURAN_JSON_PATH, SIX_HADITH_BOOKS_JSON_PATH, tokenizer)
        if dataset_dict:
            label_list = loaded_label_list if loaded_label_list else label_list
            fine_tune_model(dataset_dict, label_list)
        else:
            print("Aborting: Could not load training data.")
            exit()
    else:
        print(f"Found existing fine-tuned model in {OUTPUT_DIR}. Skipping training.")

    print("Loading fine-tuned model for prediction...")
    model = AutoModelForTokenClassification.from_pretrained(OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")
    test_data = load_test_data_from_xml(TEST_XML_PATH)
    if test_data:
        predictions = predict_on_test_data(model, tokenizer, test_data, label_list)
        generate_submission_file(predictions, SUBMISSION_TSV_PATH, SUBMISSION_ZIP_PATH)
        print("\n--- Script Finished ---")
    else:
        print("\n--- Script Aborted: Could not load test data. ---")
