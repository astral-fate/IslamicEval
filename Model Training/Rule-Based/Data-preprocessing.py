
import json
import pandas as pd
from datasets import Dataset
import re
import os
import random
from tqdm import tqdm
from transformers import AutoTokenizer
import nltk

# --- 1. Configuration ---
MODEL_NAME = "aubmindlab/bert-base-arabertv2"

# Input data paths
QURAN_JSON_PATH = "/content/drive/MyDrive/FinalIslamic/data/quran.json"
SIX_HADITH_BOOKS_JSON_PATH = "/content/drive/MyDrive/FinalIslamic/data/six_hadith_books.json"

# Output paths for processed data
PREPROCESSED_TRAIN_PATH = "/content/drive/MyDrive/FinalIslamic/prepros/preprocessed_train_30p_dataset"
PREPROCESSED_VAL_PATH = "/content/drive/MyDrive/FinalIslamic/prepros/preprocessed_val_30p_dataset"
CSV_OUTPUT_DIR = "/content/drive/MyDrive/FinalIslamic/preprocessed_csv_30p/"


# --- NEW HELPER FUNCTION ---
def normalize_arabic(text):
    """Removes Arabic diacritics (Tashkeel) and Tatweel from the text."""
    # This regex targets the Unicode range for Arabic diacritics and the Tatweel character.
    text = re.sub(r'[\u064B-\u0652\u0640]', '', text)
    return text


def split_long_texts(texts, tokenizer, max_tokens=25, label_type="Ayah"):
    """
    Splits long texts into smaller chunks based purely on token length.
    It finds the nearest space to the middle of the text to create a clean split.
    """
    print(f"🔪 Splitting {label_type} texts longer than {max_tokens} tokens...")
    split_texts = []
    split_count = 0
    for text in tqdm(texts, desc=f"Processing {label_type}s"):
        tokens = tokenizer.tokenize(text)
        if len(tokens) <= max_tokens:
            split_texts.append(text)
        else:
            mid_point = len(text) // 2
            split_pos = text.rfind(' ', 0, mid_point)
            if split_pos == -1:
                split_pos = mid_point

            part1 = text[:split_pos].strip()
            part2 = text[split_pos:].strip()

            if part1: split_texts.append(part1)
            if part2: split_texts.append(part2)
            split_count += 1

    print(f"✅ Splitting complete. Original: {len(texts)} texts, New total: {len(split_texts)} texts. ({split_count} texts were split).")
    return split_texts


def _create_example_fixed(text, label_type, tokenizer, label_to_id, prefixes, suffixes, neutral_sentences, save_details=False):
    """Creates a single tokenized example with context."""
    try:
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        if not cleaned_text:
            return None

        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)

        if random.random() > 0.3:
            context = random.choice(neutral_sentences)
            if random.random() > 0.5:
                full_text = f'{prefix} {context} "{cleaned_text}" {suffix}'
            else:
                full_text = f'{prefix} "{cleaned_text}" {context} {suffix}'
        else:
            full_text = f'{prefix} "{cleaned_text}" {suffix}'

        full_text = re.sub(r'\s+', ' ', full_text).strip()
        char_start = full_text.find(cleaned_text)
        if char_start == -1:
            return None
        char_end = char_start + len(cleaned_text)

        tokenized_input = tokenizer(full_text, truncation=True, max_length=512)
        input_ids = tokenized_input['input_ids']
        attention_mask = tokenized_input['attention_mask']
        labels = [label_to_id['O']] * len(input_ids)

        start_token = tokenized_input.char_to_token(char_start)
        end_token = tokenized_input.char_to_token(char_end - 1)

        if start_token is not None and end_token is not None:
            labels[start_token] = label_to_id[f'B-{label_type}']
            for i in range(start_token + 1, min(end_token + 1, len(labels))):
                labels[i] = label_to_id[f'I-{label_type}']

        word_ids = tokenized_input.word_ids()
        final_labels = []
        for i, word_id in enumerate(word_ids):
            if word_id is None or (i > 0 and word_id == word_ids[i - 1]):
                final_labels.append(-100)
            else:
                final_labels.append(labels[i] if i < len(labels) else label_to_id['O'])

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": final_labels
        }

        if save_details:
            result.update({
                "original_text": text,
                "full_text": full_text,
                "prefix": prefix,
                "suffix": suffix,
                "char_start": char_start,
                "char_end": char_end,
                "label_type": label_type,
                "target_span": cleaned_text
            })
        return result
    except Exception:
        return None

def create_validation_examples(tokenizer, label_to_id, val_ayah_texts, val_hadith_texts):
    """Creates validation examples using a different set of patterns to test generalization."""
    print("🔄 Creating generalization-focused validation examples...")

    val_ayah_prefixes = ["", "وفي القرآن الكريم نجد:", "ومن آيات الله:", "وقد أنزل الله:", "ويقول الحق تبارك وتعالى:", "وفي الذكر الحكيم:", "وفي كتاب الله نقرأ:", "والدليل على ذلك قوله تعالى:"]
    val_ayah_suffixes = ["", "هذا من كلام الله", "آية عظيمة", "من القرآن الكريم", "كلام رب العالمين", "من الذكر الحكيم", "آية كريمة", "(صدق الله العظيم)"]
    val_hadith_prefixes = ["", "وفي السنة النبوية:", "ومن هدي النبي صلى الله عليه وسلم:", "وقد علمنا الرسول صلى الله عليه وسلم:", "وفي الحديث الشريف نجد:", "كما جاء في الحديث:"]
    val_hadith_suffixes = ["", "من السنة النبوية", "حديث نبوي شريف", "من هدي المصطفى", "صلى الله عليه وسلم", "(رواه الترمذي)"]
    val_transitions = ["ولنتأمل معاً", "وفي هذا السياق", "وللتوضيح", "وإليكم المثال", "وفي هذا الصدد", "وهذا يبين لنا أهمية الموضوع."]

    validation_data = []
    validation_csv_data = []

    for ayah in tqdm(val_ayah_texts, desc="Val Ayahs"):
        for variation_num in range(3):
            example = _create_example_fixed(ayah, 'Ayah', tokenizer, label_to_id, val_ayah_prefixes, val_ayah_suffixes, val_transitions, save_details=True)
            if example:
                validation_data.append({k: v for k, v in example.items() if k in ["input_ids", "attention_mask", "labels"]})
                details = {k: v for k, v in example.items() if k not in ["input_ids", "attention_mask", "labels"]}
                details.update({"variation_number": variation_num + 1, "dataset_split": "validation"})
                validation_csv_data.append(details)

    for hadith in tqdm(val_hadith_texts, desc="Val Hadiths"):
        for variation_num in range(3):
            example = _create_example_fixed(hadith, 'Hadith', tokenizer, label_to_id, val_hadith_prefixes, val_hadith_suffixes, val_transitions, save_details=True)
            if example:
                validation_data.append({k: v for k, v in example.items() if k in ["input_ids", "attention_mask", "labels"]})
                details = {k: v for k, v in example.items() if k not in ["input_ids", "attention_mask", "labels"]}
                details.update({"variation_number": variation_num + 1, "dataset_split": "validation"})
                validation_csv_data.append(details)

    print(f"✅ Created {len(validation_data)} validation examples.")
    return validation_data, validation_csv_data


def main_preprocessing():
    """Main function to run the entire preprocessing pipeline."""
    print("🔄 STEP 1: OFFLINE PREPROCESSING WITH 30% BALANCED VALIDATION")
    print("=" * 60)

    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    label_list = ['O', 'B-Ayah', 'I-Ayah', 'B-Hadith', 'I-Hadith']
    label_to_id = {l: i for i, l in enumerate(label_list)}

    print("Loading raw data...")
    with open(QURAN_JSON_PATH, 'r', encoding='utf-8') as f:
        quran_data = json.load(f)
    with open(SIX_HADITH_BOOKS_JSON_PATH, 'r', encoding='utf-8') as f:
        six_books_data = json.load(f)

    ayah_texts = [item['ayah_text'] for item in quran_data if 'ayah_text' in item]
    hadith_texts = [item['Matn'].strip() for item in six_books_data if 'Matn' in item and item['Matn'] and item['Matn'].strip()]

    # Step 1: Split long texts
    ayah_texts = split_long_texts(ayah_texts, tokenizer, max_tokens=25, label_type="Ayah")

    # --- NEW: NORMALIZE AND AUGMENT AYAH DATA ---
    print("🔄 Normalizing Ayah texts for data augmentation...")
    # Create a new list containing Ayahs with Tashkeel removed
    normalized_ayah_texts = [normalize_arabic(text) for text in tqdm(ayah_texts, desc="Normalizing")]

    # Combine the original (with Tashkeel) and normalized (without Tashkeel) lists
    original_count = len(ayah_texts)
    ayah_texts.extend(normalized_ayah_texts)
    print(f"✅ Normalization complete. Ayah count increased from {original_count} to {len(ayah_texts)}.")
    # --- END OF NEW LOGIC ---

    MAX_TEXT_LENGTH = 1500
    ayah_texts = [t for t in ayah_texts if len(t) < MAX_TEXT_LENGTH]
    hadith_texts = [t for t in hadith_texts if len(t) < MAX_TEXT_LENGTH]
    print(f"Filtered: {len(ayah_texts)} Ayahs, {len(hadith_texts)} Hadiths")

    # --- MODIFIED: 30% BALANCED VALIDATION SPLIT ---
    random.seed(42)

    # Calculate validation size based on 30% of the total unique texts
    total_texts = len(ayah_texts) + len(hadith_texts)
    total_val_size = int(total_texts * 0.30)
    # Ensure the total size is an even number for a perfect 50/50 split
    if total_val_size % 2 != 0:
        total_val_size += 1
    val_size_per_class = total_val_size // 2

    print(f"🎯 Creating 30% BALANCED validation split:")
    print(f"   - Total available texts: {total_texts:,}")
    print(f"   - Target validation size (30%): {total_val_size:,} texts ({val_size_per_class} per class)")
    print(f"   - Target validation examples (x3): {total_val_size * 3:,} examples")
    print(f"   - Available Ayah texts: {len(ayah_texts):,}")
    print(f"   - Available Hadith texts: {len(hadith_texts):,}")

    # Ensure we have enough texts in each class
    if len(ayah_texts) < val_size_per_class:
        print(f"❌ WARNING: Not enough Ayah texts for a balanced 30% split!")
        print(f"   - Need {val_size_per_class}, have {len(ayah_texts)}. Adjusting validation size.")
        val_size_per_class = len(ayah_texts)
        total_val_size = val_size_per_class * 2
        print(f"   - Reduced validation size to: {total_val_size} texts ({val_size_per_class} per class)")

    if len(hadith_texts) < val_size_per_class:
        print(f"❌ WARNING: Not enough Hadith texts for a balanced 30% split!")
        print(f"   - Need {val_size_per_class}, have {len(hadith_texts)}. Adjusting validation size.")
        val_size_per_class = min(val_size_per_class, len(hadith_texts))
        total_val_size = val_size_per_class * 2
        print(f"   - Reduced validation size to: {total_val_size} texts ({val_size_per_class} per class)")

    # Sample equal numbers from each class
    val_ayah_texts = random.sample(ayah_texts, val_size_per_class)
    val_hadith_texts = random.sample(hadith_texts, val_size_per_class)

    print(f"✅ 30% balanced validation split created:")
    print(f"   - Validation Ayah texts: {len(val_ayah_texts):,}")
    print(f"   - Validation Hadith texts: {len(val_hadith_texts):,}")
    print(f"   - Total validation texts: {len(val_ayah_texts) + len(val_hadith_texts):,}")
    print(f"   - Validation examples (3x): {(len(val_ayah_texts) + len(val_hadith_texts)) * 3:,}")

    # Create training sets (remove validation texts from training)
    val_ayah_set = set(val_ayah_texts)
    val_hadith_set = set(val_hadith_texts)

    train_ayah_texts = [text for text in ayah_texts if text not in val_ayah_set]
    train_hadith_texts = [text for text in hadith_texts if text not in val_hadith_set]

    print(f"📊 Training data after removing validation:")
    print(f"   - Training Ayah texts: {len(train_ayah_texts):,}")
    print(f"   - Training Hadith texts: {len(train_hadith_texts):,}")
    print(f"   - Training examples (3x): {(len(train_ayah_texts) + len(train_hadith_texts)) * 3:,}")
    # --- END OF MODIFIED VALIDATION SPLIT ---

    quran_train_prefixes = ["", "قال الله تعالى:", "وقال الله عز وجل:", "كما ورد في القرآن الكريم:", "وفي كتاب الله:", "ومن آيات الله:", "يقول سبحانه وتعالى:", "وفي هذا الشأن يقول الله:"]
    quran_train_suffixes = ["", "صدق الله العظيم", "آية كريمة", "من القرآن الكريم", "كلام الله عز وجل", "من الذكر الحكيم", "ولذلك عبرة للمعتبرين", "وهذا بيان للناس"]
    hadith_train_prefixes = ["", "قال رسول الله صلى الله عليه وسلم:", "وقال النبي صلى الله عليه وسلم:", "عن النبي صلى الله عليه وسلم:", "روى أن النبي صلى الله عليه وسلم قال:", "وفي الحديث الشريف:", "وعن أبي هريرة رضي الله عنه قال:"]
    hadith_train_suffixes = ["", "رواه البخاري", "رواه مسلم", "حديث صحيح", "صلى الله عليه وسلم", "من السنة النبوية", "(متفق عليه)", "أو كما قال صلى الله عليه وسلم"]
    neutral_sentences = ["وبناء على ذلك، يمكننا أن نستنتج.", "وهذا يوضح عظمة التشريع.", "وفي هذا هداية للمؤمنين.", "إن في ذلك لآيات لقوم يعقلون.", "وهذا هو القول الراجح."]


    print("🔄 Preprocessing training examples...")
    train_examples = []
    ayah_csv_data, hadith_csv_data = [], []
    failed_examples = 0

    for ayah in tqdm(train_ayah_texts, desc="Training Ayahs"):
        for variation in range(3):
            example = _create_example_fixed(ayah, 'Ayah', tokenizer, label_to_id, quran_train_prefixes, quran_train_suffixes, neutral_sentences, save_details=True)
            if example:
                train_examples.append({k: v for k, v in example.items() if k in ["input_ids", "attention_mask", "labels"]})
                details = {k: v for k, v in example.items() if k not in ["input_ids", "attention_mask", "labels"]}
                details.update({"variation_number": variation + 1, "dataset_split": "training"})
                ayah_csv_data.append(details)
            else:
                failed_examples += 1

    for hadith in tqdm(train_hadith_texts, desc="Training Hadiths"):
        for variation in range(3):
            example = _create_example_fixed(hadith, 'Hadith', tokenizer, label_to_id, hadith_train_prefixes, hadith_train_suffixes, neutral_sentences, save_details=True)
            if example:
                train_examples.append({k: v for k, v in example.items() if k in ["input_ids", "attention_mask", "labels"]})
                details = {k: v for k, v in example.items() if k not in ["input_ids", "attention_mask", "labels"]}
                details.update({"variation_number": variation + 1, "dataset_split": "training"})
                hadith_csv_data.append(details)
            else:
                failed_examples += 1

    print(f"✅ Generated {len(train_examples)} training examples")
    print(f"❌ Failed to create {failed_examples} examples")

    validation_examples, validation_csv_data = create_validation_examples(tokenizer, label_to_id, val_ayah_texts, val_hadith_texts)

    print("💾 Saving preprocessing details to CSV files...")
    pd.DataFrame(ayah_csv_data).to_csv(os.path.join(CSV_OUTPUT_DIR, "ayah_training_details.csv"), index=False, encoding='utf-8')
    pd.DataFrame(hadith_csv_data).to_csv(os.path.join(CSV_OUTPUT_DIR, "hadith_training_details.csv"), index=False, encoding='utf-8')
    pd.DataFrame(validation_csv_data).to_csv(os.path.join(CSV_OUTPUT_DIR, "validation_details.csv"), index=False, encoding='utf-8')
    print("✅ CSV files saved.")

    print("💾 Saving final tokenized datasets...")
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(validation_examples)
    train_dataset.save_to_disk(PREPROCESSED_TRAIN_PATH)
    val_dataset.save_to_disk(PREPROCESSED_VAL_PATH)
    print(f"✅ Datasets saved to {PREPROCESSED_TRAIN_PATH} and {PREPROCESSED_VAL_PATH}")

    # Updated summary with balanced validation info
    summary_data = [
        {"dataset": "Training_Ayah", "total_examples": len(ayah_csv_data), "unique_texts": len(train_ayah_texts)},
        {"dataset": "Training_Hadith", "total_examples": len(hadith_csv_data), "unique_texts": len(train_hadith_texts)},
        {"dataset": "Validation_Ayah", "total_examples": len(val_ayah_texts) * 3, "unique_texts": len(val_ayah_texts)},
        {"dataset": "Validation_Hadith", "total_examples": len(val_hadith_texts) * 3, "unique_texts": len(val_hadith_texts)},
        {"dataset": "TOTAL", "total_examples": len(train_examples) + len(validation_examples), "failed_examples": failed_examples}
    ]
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(CSV_OUTPUT_DIR, "preprocessing_summary_balanced_30p.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Preprocessing summary saved to: {summary_path}")

    # Print final balanced statistics
    print("\n🎉 30% BALANCED PREPROCESSING COMPLETE!")
    print("📊 FINAL DATASET STATISTICS:")
    print(f"   Training:   {len(train_ayah_texts):,} Ayahs + {len(train_hadith_texts):,} Hadiths = {len(train_examples):,} examples")
    print(f"   Validation: {len(val_ayah_texts):,} Ayahs + {len(val_hadith_texts):,} Hadiths = {len(validation_examples):,} examples")
    print(f"   Validation balance: {len(val_ayah_texts)/(len(val_ayah_texts)+len(val_hadith_texts))*100:.1f}% Ayah, {len(val_hadith_texts)/(len(val_ayah_texts)+len(val_hadith_texts))*100:.1f}% Hadith")
    print(f"   🎯 Validation set is ~{round((len(val_ayah_texts) + len(val_hadith_texts)) / total_texts * 100)}% of the total unique texts.")


if __name__ == "__main__":
    main_preprocessing()
