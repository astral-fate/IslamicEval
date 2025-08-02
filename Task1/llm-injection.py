# SCRIPT 2: generate_synthetic_texts.py (MODIFIED FOR FULL DATASET)
# This script generates contextualized training examples for the ENTIRE dataset.

import json
import torch
from transformers import pipeline
import random
import os
import time

# --- 1. Configuration ---
GENERATOR_MODEL_NAME = "aubmindlab/aragpt2-base"
QURAN_JSON_PATH = "quran.json"
SIX_HADITH_BOOKS_JSON_PATH = "six_hadith_books.json"
OUTPUT_SYNTHETIC_DATASET_PATH = "synthetic_dataset_full.json" # New output file name

# --- 2. Helper Functions ---
def format_time(seconds):
    """Formats seconds into a readable minutes and seconds format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    if minutes > 0:
        return f"{minutes} minute(s) and {seconds} second(s)"
    return f"{seconds} second(s)"

def create_dummy_files():
    """Creates dummy data files if they don't exist, for demonstration."""
    if not os.path.exists(QURAN_JSON_PATH):
        print(f"Creating dummy '{QURAN_JSON_PATH}'...")
        quran_data = [{"ayah_text": "بِسْمِ اللَّهِ الرَّحْمَـٰنِ الرَّحِيمِ"}]
        with open(QURAN_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(quran_data, f, ensure_ascii=False, indent=4)

    if not os.path.exists(SIX_HADITH_BOOKS_JSON_PATH):
        print(f"Creating dummy '{SIX_HADITH_BOOKS_JSON_PATH}'...")
        six_hadith_data = [{"hadithTxt": "من سلك طريقا يلتمس فيه علما سهل الله له به طريقا إلى الجنة"}]
        with open(SIX_HADITH_BOOKS_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(six_hadith_data, f, ensure_ascii=False, indent=4)

# --- 3. Main Synthetic Data Generation Function ---
def generate_synthetic_data(quran_path, hadith_path, output_path):
    """
    Creates and saves a dataset of synthetic training examples for the entire dataset.
    """
    print("Starting synthetic data generation process for the FULL dataset...")
    try:
        with open(quran_path, 'r', encoding='utf-8') as f: quran_data = json.load(f)
        with open(hadith_path, 'r', encoding='utf-8') as f: six_books_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure source files are in the same directory.")
        return

    ayah_texts = [item['ayah_text'] for item in quran_data if 'ayah_text' in item and item['ayah_text']]
    hadith_texts = [item['hadithTxt'] for item in six_books_data if 'hadithTxt' in item and item['hadithTxt']]
    print(f"Loaded {len(ayah_texts)} Ayahs and {len(hadith_texts)} Hadiths.")

    processed_data = []

    # --- Initialize Generator Model ---
    model_load_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing text generator ({GENERATOR_MODEL_NAME}) on device: {device}...")
    try:
        text_generator = pipeline('text-generation', model=GENERATOR_MODEL_NAME, device=device)
    except Exception as e:
        print(f"Could not load generator model. Error: {e}")
        return
    print(f"✅ Generator model loaded in {format_time(time.time() - model_load_start)}\n")

    prompt_templates = [
        "اشرح المفهوم التالي مستشهداً بالنص: {text}",
        "في سياق الحديث عن الفضائل، يُذكر النص التالي: {text}، وهذا يعني أن",
        "كيف يمكن تطبيق هذا القول في حياتنا اليومية؟ القول هو: {text}",
        "من الأدلة الشرعية على ذلك، النص التالي: {text}"
    ]

    def create_example(text, label_type):
        """Generates a single synthetic example using the LLM."""
        prompt = random.choice(prompt_templates).format(text=text)
        try:
            outputs = text_generator(
                prompt, max_new_tokens=40, num_return_sequences=1,
                pad_token_id=text_generator.tokenizer.eos_token_id, truncation=True
            )
            generated_context = outputs[0]['generated_text']
            full_text = generated_context if text in generated_context else prompt
        except Exception:
            full_text = prompt

        char_start = full_text.find(text)
        if char_start != -1:
            return {
                "full_text": full_text, "span_text": text, "char_start": char_start,
                "char_end": char_start + len(text), "label_type": label_type
            }
        return None

    # --- Generate Examples for ALL Ayahs ---
    print(f"Generating synthetic examples for all {len(ayah_texts)} Ayahs...")
    ayah_gen_start = time.time()
    for i, ayah in enumerate(ayah_texts):
        if (i + 1) % 500 == 0: print(f"  ... processing Ayah {i + 1}/{len(ayah_texts)}")
        example = create_example(ayah, 'Ayah')
        if example: processed_data.append(example)
    print(f"✅ All synthetic Ayah examples generated in {format_time(time.time() - ayah_gen_start)}\n")

    # --- Generate Examples for ALL Hadiths ---
    short_hadiths = [h for h in hadith_texts if len(h) < 1000]
    print(f"Generating synthetic examples for {len(short_hadiths)} Hadiths (filtered by length)...")
    hadith_gen_start = time.time()
    for i, hadith in enumerate(short_hadiths):
        if (i + 1) % 500 == 0: print(f"  ... processing Hadith {i + 1}/{len(short_hadiths)}")
        example = create_example(hadith, 'Hadith')
        if example: processed_data.append(example)
    print(f"✅ All synthetic Hadith examples generated in {format_time(time.time() - hadith_gen_start)}\n")

    # --- Save the final combined dataset ---
    print(f"Saving {len(processed_data)} total examples to '{output_path}'...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print("✅ Full synthetic dataset saved successfully.")

# --- 4. Main Execution ---
if __name__ == "__main__":
    create_dummy_files()
    generate_synthetic_data(QURAN_JSON_PATH, SIX_HADITH_BOOKS_JSON_PATH, OUTPUT_SYNTHETIC_DATASET_PATH)
    print("\n--- Full Synthetic Data Generation Script Finished ---")
