# SCRIPT 2: generate_synthetic_texts.py (MODIFIED WITH CUDA FIX)
# This script generates contextualized training examples for the ENTIRE dataset.

import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import random
import os
import time

# --- 1. Configuration ---
GENERATOR_MODEL_NAME = "aubmindlab/aragpt2-base"
RAW_DATASET_PATH = "raw_dataset.json"
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
    """Creates a dummy raw_dataset.json if it doesn't exist."""
    if not os.path.exists(RAW_DATASET_PATH):
        print(f"Creating dummy '{RAW_DATASET_PATH}'...")
        dummy_data = [
            {"full_text": "بِسْمِ اللَّهِ الرَّحْمَـٰنِ الرَّحِيمِ", "label_type": "Ayah"},
            {"full_text": "من سلك طريقا يلتمس فيه علما سهل الله له به طريقا إلى الجنة", "label_type": "Hadith"}
        ]
        with open(RAW_DATASET_PATH, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, ensure_ascii=False, indent=4)

# --- 3. Main Synthetic Data Generation Function ---
def generate_synthetic_data(raw_data_path, output_path):
    """
    Creates and saves a dataset of synthetic training examples for the entire dataset.
    """
    print("Starting synthetic data generation process for the FULL dataset...")
    try:
        with open(raw_data_path, 'r', encoding='utf-8') as f: loaded_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure '{RAW_DATASET_PATH}' is in the same directory.")
        return

    ayah_texts = [item['full_text'] for item in loaded_data if item.get('label_type') == 'Ayah' and item.get('full_text')]
    hadith_texts = [item['full_text'] for item in loaded_data if item.get('label_type') == 'Hadith' and item.get('full_text')]
    print(f"Loaded {len(ayah_texts)} Ayahs and {len(hadith_texts)} Hadiths from '{raw_data_path}'.")

    processed_data = []

    # --- Initialize Generator Model ---
    model_load_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing text generator ({GENERATOR_MODEL_NAME}) on device: {device}...")
    try:
        # ==================== FIX STARTS HERE ====================
        # Load tokenizer and model separately to configure them before creating the pipeline.
        tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL_NAME)

        # GPT-2 models often lack a pad token, so we set it to the end-of-sequence token.
        # This is the most common fix for the CUDA assert error.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.config.pad_token_id = tokenizer.pad_token_id

        # Now, create the pipeline with the correctly configured model and tokenizer.
        text_generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        # ===================== FIX ENDS HERE =====================
    except Exception as e:
        print(f"Could not load generator model. Error: {e}")
        return
    print(f"✅ Generator model loaded in {format_time(time.time() - model_load_start)}\n")

    prompt_templates = [
        # General Explanation & Meaning
        "اشرح النص التالي مبيناً معناه العام والدروس المستفادة منه: {text}",
        "ما هي أبرز الفوائد والعبر من النص التالي؟ النص هو: {text}",
        "يتناول النص التالي قضية مهمة، وهي: {text}، وهذا يدل على أن",
    
        # Contextual & Evidentiary
        "يُستشهد بالنص التالي: {text}، في سياق الحديث عن",
        "من الأدلة الشرعية على هذه المسألة، النص التالي: {text}",
        "ورد في الأثر قوله: {text}، ويُفهم من ذلك أن",
    
        # Application & Relevance
        "كيف يمكن تجسيد هذا التوجيه: {text}، في واقعنا المعاصر؟",
        "يعتبر النص التالي: {text}، قاعدة أساسية في باب",
    ]

    def create_example(text, label_type):
        prompt = random.choice(prompt_templates).format(text=text)
        try:
            outputs = text_generator(
                prompt, max_new_tokens=40, num_return_sequences=1,
                truncation=True
                # No need to set pad_token_id here anymore, it's in the model's config
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
    # For more detailed error messages, you can uncomment the line below
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    create_dummy_files()
    generate_synthetic_data(RAW_DATASET_PATH, OUTPUT_SYNTHETIC_DATASET_PATH)
    print("\n--- Full Synthetic Data Generation Script Finished ---")
