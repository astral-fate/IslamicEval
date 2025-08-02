# -*- coding: utf-8 -*-
"""
SCRIPT 2: FAST MODEL TRAINING (Modified for Google Colab)

This script loads the pre-generated dataset (`preprocessed_dataset.json`)
and then fine-tunes and evaluates the model within a Google Colab environment.

It assumes your data is located in a folder named 'Islamic' in your Google Drive.
"""
import json
import pandas as pd
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import re
import os
import zipfile
from google.colab import drive

# --- Mount Google Drive ---
# This allows the script to access files stored in your Google Drive.
print("Mounting Google Drive...")
drive.mount('/content/drive')
print("Google Drive mounted successfully.")

# --- Configuration ---
# All paths are updated to point to the 'Islamic' folder in your Google Drive.
BASE_FOLDER = "/content/drive/MyDrive/Islamic/"
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
OUTPUT_DIR = os.path.join(BASE_FOLDER, "arabert_finetuned_model_v1_tempalte")
PREPROCESSED_DATA_PATH = os.path.join(BASE_FOLDER, "preprocessed_dataset_templated.json") # Input for this script
TEST_XML_PATH = os.path.join(BASE_FOLDER, "test_SubtaskA.xml")
SUBMISSION_TSV_PATH = os.path.join(BASE_FOLDER, "submission_tem.tsv")
SUBMISSION_ZIP_PATH = os.path.join(BASE_FOLDER, "submission_tem.zip")


# --- Data Loading and Final Tokenization ---
def load_and_tokenize_preprocessed_data(data_path, tokenizer):
    """
    Loads the pre-generated dataset and tokenizes it for the model.
    """
    print(f"Loading pre-processed data from {data_path}...")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            preprocessed_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{data_path}'.")
        print("Please make sure 'preprocessed_dataset.json' is in your 'Islamic' folder on Google Drive.")
        return None, None

    label_list = ['O', 'B-Ayah', 'I-Ayah', 'B-Hadith', 'I-Hadith']
    label_to_id = {l: i for i, l in enumerate(label_list)}
    final_tokenized_data = []

    for item in preprocessed_data:
        full_text = item['full_text']
        label_type = item['label_type']
        char_start = item['char_start']
        char_end = item['char_end']

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
            if word_id is None or (i > 0 and word_id == word_ids[i-1]):
                final_labels.append(-100)
            else:
                final_labels.append(labels[i])

        final_tokenized_data.append({
            "input_ids": tokenized_input['input_ids'],
            "attention_mask": tokenized_input['attention_mask'],
            "labels": final_labels
        })

    dataset = Dataset.from_list(final_tokenized_data).train_test_split(test_size=0.1, seed=42)
    return dataset, label_list

# --- The rest of the script is for training and prediction ---
def fine_tune_model(dataset_dict, label_list):
    print("Starting fine-tuning process...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label={i: l for i, l in enumerate(label_list)},
        label2id={l: i for i, l in enumerate(label_list)}
    )
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=15,
        weight_decay=0.01,
        save_total_limit=1
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model fine-tuned and saved to {OUTPUT_DIR}")

def load_test_data_from_xml(xml_path):
    print(f"Loading test data from {xml_path}...")
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        pattern = re.compile(r"<Question>.*?<ID>(.*?)</ID>.*?<Response>(.*?)</Response>.*?</Question>", re.DOTALL)
        matches = pattern.findall(content)
        if not matches:
            print("Warning: No questions found in the test XML file.")
            return []
        return [{'Question_ID': m[0].strip(), 'Text': m[1].strip()} for m in matches]
    except FileNotFoundError:
        print(f"Error: Test XML file not found at {xml_path}")
        return []

def predict_on_test_data(model, tokenizer, test_data, label_list):
    print("Predicting spans on the test set...")
    model.eval()
    device = model.device
    all_predictions = []
    for item in test_data:
        qid, text = item["Question_ID"], item["Text"]
        if not text:
            all_predictions.append({"Question_ID": qid, "Span_Start": 0, "Span_End": 0, "Span_Type": "No_Spans"})
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=2)[0]
        pred_labels = [label_list[p.item()] for p in preds]
        spans, current_span = [], {}
        for i, label in enumerate(pred_labels):
            word_id = inputs.word_ids(batch_index=0)[i]
            if word_id is None: continue
            if label.startswith('B-'):
                if current_span: spans.append(current_span)
                cs = inputs.token_to_chars(i)
                current_span = {'type': label[2:], 'start': cs.start}
            elif not label.startswith('I-') and current_span:
                spans.append(current_span)
                current_span = {}
            if current_span and label.startswith('I-'):
                cs = inputs.token_to_chars(i)
                current_span['end'] = cs.end
        if current_span: spans.append(current_span)
        if spans:
            for span in spans:
                if 'start' in span and 'end' in span:
                    all_predictions.append({"Question_ID": qid, "Span_Start": span['start'], "Span_End": span['end'], "Span_Type": span['type']})
        else:
            all_predictions.append({"Question_ID": qid, "Span_Start": 0, "Span_End": 0, "Span_Type": "No_Spans"})
    return all_predictions

def generate_submission_file(predictions, output_path, zip_path):
    if not predictions:
        print("No predictions were generated, skipping submission file creation.")
        return
    print(f"Generating submission file at {output_path}...")
    pd.DataFrame(predictions)[["Question_ID", "Span_Start", "Span_End", "Span_Type"]].to_csv(output_path, sep='\t', index=False, header=False)
    print(f"Compressing submission file to {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_path, os.path.basename(output_path))
    print("Submission zip created successfully.")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        print(f"No fine-tuned model found in {OUTPUT_DIR}. Starting the training process...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        dataset_dict, label_list = load_and_tokenize_preprocessed_data(PREPROCESSED_DATA_PATH, tokenizer)
        if dataset_dict:
            fine_tune_model(dataset_dict, label_list)
        else:
            print("\n--- Script Aborted: Could not load training data. ---")
            exit()
    else:
        print(f"Found existing fine-tuned model in {OUTPUT_DIR}. Skipping training.")

    print("Loading fine-tuned model for prediction...")
    model = AutoModelForTokenClassification.from_pretrained(OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

    # Ensure model is on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on device: {device}")

    test_data = load_test_data_from_xml(TEST_XML_PATH)
    if test_data:
        label_list = ['O', 'B-Ayah', 'I-Ayah', 'B-Hadith', 'I-Hadith']
        predictions = predict_on_test_data(model, tokenizer, test_data, label_list)
        generate_submission_file(predictions, SUBMISSION_TSV_PATH, SUBMISSION_ZIP_PATH)
        print("\n--- Script Finished ---")
    else:
        print("\n--- Script Finished: No test data to predict on. ---")
