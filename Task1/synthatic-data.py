# -*- coding: utf-8 -*-
"""
SCRIPT: ENHANCED OFFLINE DATA GENERATION

This script generates a synthetic dataset by embedding Quranic Ayahs and Hadiths
into natural-sounding sentences using a variety of sophisticated LLM templates.
It uses 'quran.json' and 'six_hadith_books.json' as data sources.
"""
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline
import random
import os
import time

# --- Configuration ---
GENERATOR_MODEL_NAME = "aubmindlab/bert-base-arabertv2"
# MODIFIED: Corrected the input data paths as requested
QURAN_JSON_PATH = "quran.json"
SIX_HADITH_BOOKS_JSON_PATH = "six_hadith_books.json" 
OUTPUT_DATASET_PATH = "preprocessed_dataset_templated.json"

# --- Helper function for logging ---
def format_time(seconds):
    """Formats seconds into a readable minutes and seconds format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    if minutes > 0:
        return f"{minutes} minute(s) and {seconds} second(s)"
    else:
        return f"{seconds} second(s)"

def generate_and_save_dataset(quran_path, hadith_path, output_path):
    """
    Uses a generative LLM with improved templates to create and save a synthetic dataset.
    """
    print("Starting OFFLINE data generation with enhanced templates...")

    # Load source texts
    try:
        with open(quran_path, 'r', encoding='utf-8') as f:
            quran_data = json.load(f)
        # MODIFIED: Load hadith from the specified JSON file
        with open(hadith_path, 'r', encoding='utf-8') as f:
            six_books_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Missing data file: {e}. Aborting.")
        return

    ayah_texts = [item['ayah_text'] for item in quran_data if 'ayah_text' in item]
    # MODIFIED: Extract hadith text from the correct key in the JSON file
    hadith_texts = [item['hadithTxt'] for item in six_books_data if 'hadithTxt' in item]

    # --- Timer Start: Model Loading ---
    model_load_start = time.time()
    print(f"Loading generator model '{GENERATOR_MODEL_NAME}'. This may take a moment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Force CPU usage to prevent GPU errors
    # device = torch.device("cpu")
    text_generator = pipeline("text-generation", model=GENERATOR_MODEL_NAME, device=device)
    model_load_end = time.time()
    print(f"✅ Generator model loaded on device: {device}.")
    print(f"   └── Time taken: {format_time(model_load_end - model_load_start)}\n")
    # --- Timer End: Model Loading ---

    processed_data = []

    # --- MODIFIED: Using improved prompt templates for more natural context ---
    prompt_templates = [
        "اشرح المفهوم التالي مستشهداً بالنص: {text}",
        "في سياق الحديث عن الفضائل، يُذكر النص التالي: {text}، وهذا يعني أن",
        "كيف يمكن تطبيق هذا القول في حياتنا اليومية؟ القول هو: {text}",
        "من الأدلة الشرعية على ذلك، النص التالي: {text}"
    ]

    def create_example(text, label_type, generator):
        # Randomly choose a template for each example
        prompt = random.choice(prompt_templates).format(text=text)
        try:
            generated_outputs = generator(prompt, max_new_tokens=60, num_return_sequences=1, pad_token_id=generator.tokenizer.eos_token_id)
            full_text = generated_outputs[0]['generated_text']
            # Fallback: Ensure the original text is present if the model hallucinates
            if text not in full_text:
                full_text = f"{prompt.split(':')[0]}: {text}"
        except Exception:
            full_text = f"النص هو: {text}" # General fallback
        
        char_start = full_text.find(text)
        if char_start != -1:
            processed_data.append({
                "full_text": full_text,
                "span_text": text,
                "char_start": char_start,
                "char_end": char_start + len(text),
                "label_type": label_type
            })

    # --- Timer Start: Ayah Generation ---
    print("Generating Ayah examples...")
    ayah_gen_start = time.time()
    for i, ayah in enumerate(ayah_texts):
        if (i + 1) % 250 == 0: print(f"  Processing Ayah {i + 1}/{len(ayah_texts)}")
        create_example(ayah, 'Ayah', text_generator)
    ayah_gen_end = time.time()
    print("✅ Ayah examples generated.")
    print(f"   └── Time taken: {format_time(ayah_gen_end - ayah_gen_start)}\n")
    # --- Timer End: Ayah Generation ---

    # --- Timer Start: Hadith Generation ---
    print("Generating Hadith examples...")
    hadith_gen_start = time.time()
    for i, hadith in enumerate(hadith_texts):
        if (i + 1) % 250 == 0: print(f"  Processing Hadith {i + 1}/{len(hadith_texts)}")
        create_example(hadith, 'Hadith', text_generator)
    hadith_gen_end = time.time()
    print("✅ Hadith examples generated.")
    print(f"   └── Time taken: {format_time(hadith_gen_end - hadith_gen_start)}\n")
    # --- Timer End: Hadith Generation ---

    # --- Timer Start: File Saving ---
    print(f"Saving {len(processed_data)} generated examples to {output_path}...")
    save_start = time.time()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    save_end = time.time()
    print("✅ Dataset saved.")
    print(f"   └── Time taken: {format_time(save_end - save_start)}\n")
    # --- Timer End: File Saving ---


if __name__ == "__main__":
    script_start_time = time.time()
    generate_and_save_dataset(QURAN_JSON_PATH, SIX_HADITH_BOOKS_JSON_PATH, OUTPUT_DATASET_PATH)
    script_end_time = time.time()
    print("------------------------------------------------------------")
    print("✔️ Offline data generation complete!")
    print(f"Total script runtime: {format_time(script_end_time - script_start_time)}")
    print("------------------------------------------------------------")
