# -*- coding: utf-8 -*-
"""
SCRIPT 1: DATA GENERATION

This script generates a combined training dataset for the token classification task.
It performs two main steps:
1.  Adds the raw, unaltered Quran and Hadith texts for specialization.
2.  Uses a generative LLM (AraGPT2-base) with sophisticated prompt templates
    to create synthetic, contextualized examples.

The final output is a single JSON file containing all training examples.
"""

import json
import pandas as pd
import torch
from transformers import pipeline
import random
import os
import time

# --- 1. Configuration ---
GENERATOR_MODEL_NAME = "aubmindlab/bert-base-arabertv2"
QURAN_JSON_PATH = "quran.json"
SIX_HADITH_BOOKS_JSON_PATH = "six_hadith_books.json"
OUTPUT_DATASET_PATH = "training_dataset.json" # The final output file

# --- 2. Helper Functions ---
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

def format_time(seconds):
    """Formats seconds into a readable minutes and seconds format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    if minutes > 0:
        return f"{minutes} minute(s) and {seconds} second(s)"
    else:
        return f"{seconds} second(s)"

# --- 3. Main Data Generation Function ---
def generate_and_save_dataset(quran_path, hadith_path, output_path):
    """
    Creates and saves a combined dataset of raw and synthetic training examples.
    """
    print("Starting data generation process...")
    try:
        with open(quran_path, 'r', encoding='utf-8') as f: quran_data = json.load(f)
        with open(hadith_path, 'r', encoding='utf-8') as f: six_books_data = json.load(f)
    except Exception as e:
        print(f"Error loading source data files: {e}. Aborting.")
        return

    ayah_texts = [item['ayah_text'] for item in quran_data if 'ayah_text' in item]
    hadith_texts = [item['hadithTxt'] for item in six_books_data if 'hadithTxt' in item]
    print(f"Loaded {len(ayah_texts)} Ayahs and {len(hadith_texts)} Hadiths.")
    
    processed_data = []

    # --- Stage 1: Add Raw Texts for Specialization ---
    print("Adding raw religious texts to the dataset...")
    for ayah in ayah_texts:
        processed_data.append({
            "full_text": ayah, "span_text": ayah, "char_start": 0,
            "char_end": len(ayah), "label_type": "Ayah"
        })
    for hadith in hadith_texts:
        processed_data.append({
            "full_text": hadith, "span_text": hadith, "char_start": 0,
            "char_end": len(hadith), "label_type": "Hadith"
        })
    print(f"✅ Added {len(processed_data)} raw text examples.\n")

    # --- Stage 2: Generate Synthetic Data for Contextualization ---
    model_load_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing text generator ({GENERATOR_MODEL_NAME}) on device: {device}...")
    try:
        text_generator = pipeline('text-generation', model=GENERATOR_MODEL_NAME, device=device)
    except Exception as e:
        print(f"Could not load generator model. Using fallback method. Error: {e}")
        text_generator = None
    model_load_end = time.time()
    print(f"✅ Generator model loaded. Time taken: {format_time(model_load_end - model_load_start)}\n")

    prompt_templates = [
        "اشرح المفهوم التالي مستشهداً بالنص: {text}",
        "في سياق الحديث عن الفضائل، يُذكر النص التالي: {text}، وهذا يعني أن",
        "كيف يمكن تطبيق هذا القول في حياتنا اليومية؟ القول هو: {text}",
        "من الأدلة الشرعية على ذلك، النص التالي: {text}"
    ]

    def create_synthetic_example(text, label_type):
        if not text_generator:
            full_text = f"النص هو: {text}"
        else:
            prompt = random.choice(prompt_templates).format(text=text)
            try:
                outputs = text_generator(prompt, max_new_tokens=50, num_return_sequences=1, pad_token_id=text_generator.tokenizer.eos_token_id)
                generated_context = outputs[0]['generated_text']
                full_text = generated_context if text in generated_context else f"{prompt.split(':')[0]}: {text}"
            except Exception:
                full_text = f"النص هو: {text}"

        char_start = full_text.find(text)
        if char_start != -1:
            processed_data.append({
                "full_text": full_text, "span_text": text, "char_start": char_start,
                "char_end": char_start + len(text), "label_type": label_type
            })

    print("Generating synthetic Ayah examples...")
    ayah_gen_start = time.time()
    sample_size = min(2000, len(ayah_texts))
    for i, ayah in enumerate(random.sample(ayah_texts, sample_size)):
        if (i + 1) % 250 == 0: print(f"  Processing synthetic Ayah {i + 1}/{sample_size}...")
        create_synthetic_example(ayah, 'Ayah')
    ayah_gen_end = time.time()
    print(f"✅ Synthetic Ayah examples generated. Time taken: {format_time(ayah_gen_end - ayah_gen_start)}\n")

    print("Generating synthetic Hadith examples...")
    hadith_gen_start = time.time()
    sample_size = min(2000, len(hadith_texts))
    for i, hadith in enumerate(random.sample(hadith_texts, sample_size)):
        if (i + 1) % 250 == 0: print(f"  Processing synthetic Hadith {i + 1}/{sample_size}...")
        create_synthetic_example(hadith, 'Hadith')
    hadith_gen_end = time.time()
    print(f"✅ Synthetic Hadith examples generated. Time taken: {format_time(hadith_gen_end - hadith_gen_start)}\n")

    # --- Save the final combined dataset ---
    print(f"Saving {len(processed_data)} total examples to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print("✅ Dataset saved successfully.")

# --- Main Execution ---
if __name__ == "__main__":
    create_dummy_files()
    generate_and_save_dataset(QURAN_JSON_PATH, SIX_HADITH_BOOKS_JSON_PATH, OUTPUT_DATASET_PATH)
    print("\n--- Data Generation Script Finished ---")
