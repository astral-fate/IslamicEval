import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import load_from_disk
import re
import os
import zipfile
import time
from tqdm import tqdm

# --- 1. Configuration ---
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
OUTPUT_DIR = "/content/drive/MyDrive/FinalIslamic/arabert_finetuned_model_best_v2"
TEST_XML_PATH = "test_SubtaskA.xml"

# Paths for data prepared by preprocessing.py
PREPROCESSED_TRAIN_PATH = "/content/drive/MyDrive/FinalIslamic/prepros/preprocessed_train_dataset"
PREPROCESSED_VAL_PATH = "/content/drive/MyDrive/FinalIslamic/prepros/preprocessed_val_dataset"

# Output paths for submission
SUBMISSION_TSV_PATH = "/content/drive/MyDrive/FinalIslamic/test/submission.tsv"
SUBMISSION_ZIP_PATH = "/content/drive/MyDrive/FinalIslamic/test/submission.zip"

def fast_train_model():
    """Fast training using preprocessed data from disk."""
    print("🚀 STEP 2: FAST TRAINING")
    print("=" * 50)

    if not os.path.exists(PREPROCESSED_TRAIN_PATH) or not os.path.exists(PREPROCESSED_VAL_PATH):
        print(f"❌ Preprocessed datasets not found at {PREPROCESSED_TRAIN_PATH}")
        print("Please run the 'preprocessing.py' script first.")
        return False

    print("📥 Loading preprocessed datasets...")
    train_dataset = load_from_disk(PREPROCESSED_TRAIN_PATH)
    val_dataset = load_from_disk(PREPROCESSED_VAL_PATH)
    print(f"✅ Loaded {len(train_dataset)} training and {len(val_dataset)} validation examples.")

    label_list = ['O', 'B-Ayah', 'I-Ayah', 'B-Hadith', 'I-Hadith']
    id_to_label = {i: l for i, l in enumerate(label_list)}
    label_to_id = {l: i for i, l in enumerate(label_list)}

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_list), id2label=id_to_label, label2id=label_to_id
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=6,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        fp16=True,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("🏃‍♂️ Starting training...")
    start_time = time.time()
    trainer.train()
    print(f"⏱️ Training completed in {(time.time() - start_time)/60:.1f} minutes")

    trainer.save_model(OUTPUT_DIR)
    print(f"✅ Best model saved to {OUTPUT_DIR}")
    return True

def load_test_data_from_xml(xml_path):
    """Loads test data from the provided XML file format."""
    if not os.path.exists(xml_path):
        print(f"❌ Test file not found at {xml_path}")
        return []
    with open(xml_path, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = re.compile(r"<Question>.*?<ID>(.*?)</ID>.*?<Response>(.*?)</Response>.*?</Question>", re.DOTALL)
    matches = pattern.findall(content)
    return [{'Question_ID': m[0].strip(), 'Text': m[1].strip()} for m in matches]

def predict_on_test_data(model, tokenizer, test_data):
    """Predicts spans on the test data."""
    print("🔮 Predicting spans on test set...")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    label_list = list(model.config.id2label.values())
    all_predictions = []

    for item in tqdm(test_data, desc="Predicting"):
        qid, text = item["Question_ID"], item["Text"]
        if not text.strip():
            all_predictions.append({"Question_ID": qid, "Span_Start": 0, "Span_End": 0, "Span_Type": "No_Spans"})
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
            if word_id is None: continue

            if label.startswith('B-'):
                if current_span: spans.append(current_span)
                cs = inputs.token_to_chars(i)
                current_span = {'type': label[2:], 'start': cs.start, 'end': cs.end}
            elif label.startswith('I-') and current_span and current_span['type'] == label[2:]:
                cs = inputs.token_to_chars(i)
                current_span['end'] = cs.end
            elif current_span:
                spans.append(current_span)
                current_span = None
        if current_span:
            spans.append(current_span)

        if spans:
            for span in spans:
                all_predictions.append({"Question_ID": qid, "Span_Start": span['start'], "Span_End": span['end'], "Span_Type": span['type']})
        else:
            all_predictions.append({"Question_ID": qid, "Span_Start": 0, "Span_End": 0, "Span_Type": "No_Spans"})

    return all_predictions

def generate_submission_file(predictions, output_path, zip_path):
    """Generates the final submission.tsv and submission.zip files."""
    print(f"📦 Generating submission file at {output_path}...")
    df = pd.DataFrame(predictions)[["Question_ID", "Span_Start", "Span_End", "Span_Type"]]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False, header=False)
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(output_path, os.path.basename(output_path))
    print(f"✅ Submission zip created successfully at {zip_path}")


def main_finetuning():
    """Main function to run training and prediction."""
    # --- Training ---
    if not os.path.exists(OUTPUT_DIR):
        if not fast_train_model():
            print("❌ Training failed. Exiting.")
            return
    else:
        print(f"✅ Found existing fine-tuned model at {OUTPUT_DIR}. Skipping training.")

    # --- Prediction ---
    print("\n" + "="*50)
    print("🚀 STEP 3: PREDICTION")
    print("="*50)
    model = AutoModelForTokenClassification.from_pretrained(OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

    test_data = load_test_data_from_xml(TEST_XML_PATH)
    if test_data:
        predictions = predict_on_test_data(model, tokenizer, test_data)
        generate_submission_file(predictions, SUBMISSION_TSV_PATH, SUBMISSION_ZIP_PATH)
        print("\n🎉 Full pipeline completed successfully!")
    else:
        print("❌ Could not load test data. Prediction step skipped.")


if __name__ == "__main__":
    main_finetuning()
