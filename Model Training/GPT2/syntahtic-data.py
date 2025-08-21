# SCRIPT 1: generate_synthetic_data_with_llm.py (Corrected with true generation)

import json
import pandas as pd
import re
import os
import random
from tqdm import tqdm
import torch
from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel

# --- 1. Configuration ---
GENERATOR_MODEL_NAME = "aubmindlab/aragpt2-base"

# Input data paths
QURAN_JSON_PATH = "/content/drive/MyDrive/FinalIslamic/data/quran.json"
SIX_HADITH_BOOKS_JSON_PATH = "/content/drive/MyDrive/FinalIslamic/data/six_hadith_books.json"

# Output CSV file paths
QURAN_CSV_OUTPUT_PATH = "/content/drive/MyDrive/FinalIslamic/quran_synthetic_data_llm.csv"
HADITH_CSV_OUTPUT_PATH = "/content/drive/MyDrive/FinalIslamic/hadith_synthetic_data_llm.csv"
OUTPUT_DIR = os.path.dirname(QURAN_CSV_OUTPUT_PATH)

# --- 2. Helper Functions (Unchanged) ---
def normalize_arabic(text):
    text = re.sub(r'[\u064B-\u0652\u0640]', '', text)
    return text

def split_long_texts(texts, tokenizer, max_tokens=25, label_type="Ayah"):
    print(f"🔪 Splitting {label_type} texts longer than {max_tokens} tokens...")
    split_texts = []
    split_count = 0
    for text in tqdm(texts, desc=f"Processing {label_type}s"):
        # We need a tokenizer instance here for the splitting logic
        tokens = tokenizer.tokenize(text)
        if len(tokens) <= max_tokens:
            split_texts.append(text)
        else:
            mid_point = len(text) // 2
            split_pos = text.rfind(' ', 0, mid_point)
            if split_pos == -1: split_pos = mid_point
            part1 = text[:split_pos].strip()
            part2 = text[split_pos:].strip()
            if part1: split_texts.append(part1)
            if part2: split_texts.append(part2)
            split_count += 1
    print(f"✅ Splitting complete. Original: {len(texts)}, New total: {len(split_texts)}. ({split_count} texts split).")
    return split_texts

# --- 3. Main Data Generation Logic ---
def main():
    """Main function to run the entire data generation pipeline using an LLM."""
    print("🚀 STARTING: Generative Synthetic Data Creation with LLM")
    print("=" * 60)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- NEW: Initialize the Generative Model and Pipeline ---
    print(f"🔄 Initializing generative model: {GENERATOR_MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    try:
        model = GPT2LMHeadModel.from_pretrained(GENERATOR_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
        # The pipeline will now handle the text generation task
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        print("✅ Generative pipeline initialized successfully.")
    except Exception as e:
        print(f"❌ ERROR: Could not initialize the model or pipeline: {e}")
        return

    # --- Load Raw Data ---
    print("\n🔄 Loading raw Quran and Hadith data...")
    try:
        with open(QURAN_JSON_PATH, 'r', encoding='utf-8') as f: quran_data = json.load(f)
        with open(SIX_HADITH_BOOKS_JSON_PATH, 'r', encoding='utf-8') as f: hadith_data = json.load(f)
        quran_texts = [item['ayah_text'].strip() for item in quran_data if isinstance(item, dict) and 'ayah_text' in item and item['ayah_text']]
        hadith_texts = [item['Matn'].strip() for item in hadith_data if isinstance(item, dict) and 'Matn' in item and item['Matn']]
        print(f"✅ Loaded {len(quran_texts):,} Quran Ayahs and {len(hadith_texts):,} Hadith Matn.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find data file: {e}")
        return

    # --- Preprocessing ---
    print("\n🔄 Starting Preprocessing Steps...")
    # Use the generator's tokenizer for splitting logic
    quran_texts_split = split_long_texts(quran_texts, tokenizer, max_tokens=25, label_type="Ayah")
    normalized_ayah_texts = [normalize_arabic(text) for text in tqdm(quran_texts_split, desc="Normalizing")]
    final_quran_texts = quran_texts_split + normalized_ayah_texts
    print(f"✅ Ayah preprocessing complete. Total unique texts: {len(set(final_quran_texts)):,}")

    # --- NEW: Generative Prompt Templates ---
    # These are designed to be open-ended to encourage the model to continue writing.
    prompt_templates = [
        "يُستشهد بالنص التالي: {text}، وهذا يدل على أن",
        "من الأدلة الشرعية على هذه المسألة، النص التالي: {text}، ويُفهم من ذلك أن",
        "يتناول النص التالي قضية مهمة، وهي: {text}، حيث أن",
        "يمكن الاستفادة من قوله: {text}، في واقعنا المعاصر عن طريق",
    ]

    # --- NEW: Function to create examples using the LLM pipeline ---
    def create_generative_example(text, templates):
        prompt = random.choice(templates).format(text=text)
        try:
            # Generate new text based on the prompt
            outputs = text_generator(
                prompt,
                max_new_tokens=30,  # Generate a short continuation
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2 # Prevent repetitive loops
            )
            generated_text = outputs[0]['generated_text']

            # Ensure the original text is still present
            char_start = generated_text.find(text)
            if char_start != -1:
                return {
                    "full_text": generated_text,
                    "span_text": text,
                    "char_start": char_start,
                    "char_end": char_start + len(text)
                }
        except Exception as e:
            # If generation fails, we can fall back to the prompt itself
            print(f"Generation failed for text, falling back. Error: {e}")
            pass

        # Fallback if generation fails or text is lost
        char_start_fallback = prompt.find(text)
        return {
            "full_text": prompt, "span_text": text,
            "char_start": char_start_fallback, "char_end": char_start_fallback + len(text)
        }

    # --- Generate Synthetic Data using the LLM ---
    print("\n🔄 Generating synthetic data using the LLM (this will take time)...")
    quran_synthetic_list = [create_generative_example(text, prompt_templates) for text in tqdm(final_quran_texts, desc="Generating Quran examples")]
    hadith_synthetic_list = [create_generative_example(text, prompt_templates) for text in tqdm(hadith_texts, desc="Generating Hadith examples")]

    # Filter out any None results from failed generations
    quran_synthetic_list = [ex for ex in quran_synthetic_list if ex and ex['char_start'] != -1]
    hadith_synthetic_list = [ex for ex in hadith_synthetic_list if ex and ex['char_start'] != -1]

    for item in quran_synthetic_list: item['label_type'] = 'Ayah'
    for item in hadith_synthetic_list: item['label_type'] = 'Hadith'

    print(f"✅ Generated {len(quran_synthetic_list):,} synthetic Quran examples.")
    print(f"✅ Generated {len(hadith_synthetic_list):,} synthetic Hadith examples.")

    # --- Save to CSV ---
    print("\n💾 Saving synthetic data to CSV files...")
    pd.DataFrame(quran_synthetic_list).to_csv(QURAN_CSV_OUTPUT_PATH, index=False, encoding='utf-8-sig')
    pd.DataFrame(hadith_synthetic_list).to_csv(HADITH_CSV_OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ Quran data saved to: {QURAN_CSV_OUTPUT_PATH}")
    print(f"✅ Hadith data saved to: {HADITH_CSV_OUTPUT_PATH}")
    print("\n🎉 Generative data creation complete!")

if __name__ == "__main__":
    main()
