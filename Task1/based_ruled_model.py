# 66.97%


```python
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
import random
import time

# --- 1. Configuration ---

MODEL_NAME = "aubmindlab/bert-base-arabertv2"
OUTPUT_DIR = "./arabert_finetuned_model_best_v2"

# --- Input Data Paths ---
QURAN_JSON_PATH = "quran.json"
SIX_HADITH_BOOKS_JSON_PATH = "six_hadith_books.json"
TEST_XML_PATH = "test_SubtaskA.xml"

# --- Output Paths ---
SUBMISSION_TSV_PATH = "submission.tsv"
SUBMISSION_ZIP_PATH = "submission.zip"

# --- 2. Data Loading and Preprocessing ---

def create_dummy_files():
    """Creates dummy data files for demonstration."""
    if not os.path.exists(QURAN_JSON_PATH):
        print(f"Creating dummy '{QURAN_JSON_PATH}'...")
        quran_data = [{"ayah_text": "بِسْمِ اللَّهِ الرَّحْمَـٰنِ الرَّحِيمِ"}]
        with open(QURAN_JSON_PATH, 'w', encoding='utf-8') as f: json.dump(quran_data, f, ensure_ascii=False, indent=4)

    if not os.path.exists(SIX_HADITH_BOOKS_JSON_PATH):
        print(f"Creating dummy '{SIX_HADITH_BOOKS_JSON_PATH}'...")
        six_hadith_data = [{"hadithTxt": "من سلك طريقا يلتمس فيه علما سهل الله له به طريقا إلى الجنة"}]
        with open(SIX_HADITH_BOOKS_JSON_PATH, 'w', encoding='utf-8') as f: json.dump(six_hadith_data, f, ensure_ascii=False, indent=4)

    if not os.path.exists(TEST_XML_PATH):
        print(f"Creating dummy '{TEST_XML_PATH}'...")
        xml_content = """<Question><ID>A-Q001</ID><Response>نعم، يُذكر في القرآن الكريم: {وَإِنَّكَ لَعَلَىٰ خُلُقٍ عَظِيمٍ}.</Response></Question>"""
        with open(TEST_XML_PATH, 'w', encoding='utf-8') as f: f.write(xml_content)

def _create_example(text, label_type, tokenizer, label_to_id, prefixes, suffixes, neutral_sentences):
    """Helper function to create a single tokenized example."""
    prefix = random.choice(prefixes)
    suffix = random.choice(suffixes)

    if random.random() > 0.5:
        context = random.choice(neutral_sentences)
        full_text = f"{prefix} {context} {text} {suffix}"
    else:
        full_text = f"{prefix} {text} {suffix}"

    full_text = full_text.strip()
    char_start = full_text.find(text)
    if char_start == -1: return None

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
    
    return {
        "input_ids": tokenized_input['input_ids'],
        "attention_mask": tokenized_input['attention_mask'],
        "labels": final_labels
    }


def load_and_prepare_training_data(quran_path, six_books_path, tokenizer, label_to_id):
    """Creates a high-quality training dataset from all available data."""
    print("Loading and preparing full training data...")
    try:
        with open(quran_path, 'r', encoding='utf-8') as f: quran_data = json.load(f)
        with open(six_books_path, 'r', encoding='utf-8') as f: six_books_data = json.load(f)
    except Exception as e:
        print(f"Error loading source data files: {e}.")
        return None

    ayah_texts = [item['ayah_text'] for item in quran_data if 'ayah_text' in item]
    hadith_texts = [item['hadithTxt'] for item in six_books_data if 'hadithTxt' in item]

    MAX_TEXT_LENGTH = 1500
    ayah_texts = [t for t in ayah_texts if len(t) < MAX_TEXT_LENGTH]
    hadith_texts = [t for t in hadith_texts if len(t) < MAX_TEXT_LENGTH]
    print(f"Loaded and filtered {len(ayah_texts)} Ayahs and {len(hadith_texts)} Hadiths for training.")
    
    processed_data = []
    prefixes = ["", "قال الله تعالى:", "كما ورد في قوله:", "والدليل على ذلك هو قول النبي صلى الله عليه وسلم:"]
    suffixes = ["", "صدق الله العظيم.", "وهذا يوضح أهمية الموضوع.", "رواه البخاري."]
    neutral_sentences = ["وبناء على ذلك، يمكننا أن نستنتج.", "اختلف العلماء في هذه المسألة."]

    for ayah in ayah_texts:
        example = _create_example(ayah, 'Ayah', tokenizer, label_to_id, prefixes, suffixes, neutral_sentences)
        if example: processed_data.append(example)

    for hadith in hadith_texts:
        example = _create_example(hadith, 'Hadith', tokenizer, label_to_id, prefixes, suffixes, neutral_sentences)
        if example: processed_data.append(example)

    print(f"✅ Generated {len(processed_data)} training examples.")
    if not processed_data: return None
    return Dataset.from_list(processed_data)

def create_synthetic_validation_data(tokenizer, label_to_id):
    """Creates a small, synthetic validation set."""
    print("Generating synthetic validation data...")
    synthetic_ayahs = [
        "إِنَّ اللَّهَ عَلَىٰ كُلِّ شَيْءٍ قَدِيرٌ",
        "فَاصْبِرْ صَبْرًا جَمِيلًا",
        "وَاللَّهُ غَفُورٌ رَحِيمٌ",
        "وَمَا النَّصْرُ إِلَّا مِنْ عِنْدِ اللَّهِ"
    ]
    synthetic_hadiths = [
        "إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ",
        "طَلَبُ الْعِلْمِ فَرِيضَةٌ عَلَىٰ كُلِّ مُسْلِمٍ",
        "الدِّينُ النَّصِيحَةُ",
        "مَنْ غَشَّنَا فَلَيْسَ مِنَّا"
    ]
    
    validation_data = []
    prefixes = ["", "قال الله تعالى:", "كما ورد في قوله:", "والدليل على ذلك هو قول النبي صلى الله عليه وسلم:"]
    suffixes = ["", "صدق الله العظيم.", "وهذا يوضح أهمية الموضوع.", "رواه البخاري."]
    neutral_sentences = ["وبناء على ذلك، يمكننا أن نستنتج.", "اختلف العلماء في هذه المسألة."]
    
    # Create multiple variations for a more robust set
    for _ in range(25): # Generate 100 validation samples total
        for ayah in synthetic_ayahs:
            example = _create_example(ayah, 'Ayah', tokenizer, label_to_id, prefixes, suffixes, neutral_sentences)
            if example: validation_data.append(example)
        for hadith in synthetic_hadiths:
            example = _create_example(hadith, 'Hadith', tokenizer, label_to_id, prefixes, suffixes, neutral_sentences)
            if example: validation_data.append(example)

    print(f"✅ Generated {len(validation_data)} synthetic validation examples.")
    return Dataset.from_list(validation_data)


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
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=200,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Model fine-tuned and saved to {OUTPUT_DIR}")


# --- 4. Prediction & 5. Submission (No changes needed in these sections) ---
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
    label_to_id = {l: i for i, l in enumerate(label_list)}

    if not os.path.exists(OUTPUT_DIR):
        print("No fine-tuned model found. Starting data generation and training...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        train_dataset = load_and_prepare_training_data(QURAN_JSON_PATH, SIX_HADITH_BOOKS_JSON_PATH, tokenizer, label_to_id)
        validation_dataset = create_synthetic_validation_data(tokenizer, label_to_id)

        if train_dataset and validation_dataset:
            dataset_dict = DatasetDict({'train': train_dataset, 'validation': validation_dataset})
            fine_tune_model(dataset_dict, label_list)
        else:
            print("Aborting: Could not load or generate training/validation data.")
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
```

    No fine-tuned model found. Starting data generation and training...
    Loading and preparing full training data...
    Loaded and filtered 6236 Ayahs and 34662 Hadiths for training.
    ✅ Generated 29815 training examples.
    Generating synthetic validation data...
    ✅ Generated 200 synthetic validation examples.
    Starting fine-tuning process...
    
```


    
