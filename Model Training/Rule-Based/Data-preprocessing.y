🔄 STEP 1: OFFLINE PREPROCESSING WITH BALANCED VALIDATION (10K)
============================================================
Loading raw data...
🔪 Splitting Ayah texts longer than 25 tokens...
Processing Ayahs: 100%|██████████| 6236/6236 [00:00<00:00, 8236.34it/s]
✅ Splitting complete. Original: 6236 texts, New total: 6910 texts. (674 texts were split).
🔄 Normalizing Ayah texts for data augmentation...
Normalizing: 100%|██████████| 6910/6910 [00:00<00:00, 89103.61it/s]
✅ Normalization complete. Ayah count increased from 6910 to 13820.
Filtered: 13820 Ayahs, 31317 Hadiths
🎯 Creating BALANCED validation split (10K examples):
   - Target validation size: 3334 texts (1667 per class)
   - Target validation examples: 10002 examples (5001 per class)
   - Available Ayah texts: 13820
   - Available Hadith texts: 31317
✅ Balanced validation split created (10K examples):
   - Validation Ayah texts: 1,667
   - Validation Hadith texts: 1,667
   - Total validation texts: 3,334
   - Validation examples (3x): 10,002
📊 Training data after removing validation:
   - Training Ayah texts: 12015
   - Training Hadith texts: 29540
   - Training examples (3x): 124665
🔄 Preprocessing training examples...
Training Ayahs: 100%|██████████| 12015/12015 [00:08<00:00, 1427.06it/s]
Training Hadiths: 100%|██████████| 29540/29540 [00:37<00:00, 787.61it/s]
✅ Generated 124665 training examples
❌ Failed to create 0 examples
🔄 Creating generalization-focused validation examples...
Val Ayahs: 100%|██████████| 1667/1667 [00:01<00:00, 1471.95it/s]
Val Hadiths: 100%|██████████| 1667/1667 [00:02<00:00, 797.42it/s]
✅ Created 10002 validation examples.
💾 Saving preprocessing details to CSV files...
✅ CSV files saved.
💾 Saving final tokenized datasets...
Saving the dataset (1/1 shards): 100%
 124665/124665 [00:00<00:00, 472261.81 examples/s]
Saving the dataset (1/1 shards): 100%
 10002/10002 [00:00<00:00, 294298.92 examples/s]
✅ Datasets saved to /content/drive/MyDrive/FinalIslamic/prepros/preprocessed_train_10k_dataset and /content/drive/MyDrive/FinalIslamic/prepros/preprocessed_val_10k_dataset
✅ Preprocessing summary saved to: /content/drive/MyDrive/FinalIslamic/preprocessed_csv_10k/preprocessing_summary_balanced.csv

🎉 LARGE BALANCED PREPROCESSING COMPLETE!
📊 FINAL DATASET STATISTICS:
   Training:   12,015 Ayahs + 29,540 Hadiths = 124,665 examples
   Validation: 1,667 Ayahs + 1,667 Hadiths = 10,002 examples
   Validation balance: 50.0% Ayah, 50.0% Hadith
   🎯 Large validation set: ~10K examples for robust evaluation!






```

🔄 STEP 1: OFFLINE PREPROCESSING WITH BALANCED VALIDATION (10K)
============================================================
Loading raw data...
🔪 Splitting Ayah texts longer than 25 tokens...
Processing Ayahs: 100%|██████████| 6236/6236 [00:00<00:00, 8236.34it/s]
✅ Splitting complete. Original: 6236 texts, New total: 6910 texts. (674 texts were split).
🔄 Normalizing Ayah texts for data augmentation...
Normalizing: 100%|██████████| 6910/6910 [00:00<00:00, 89103.61it/s]
✅ Normalization complete. Ayah count increased from 6910 to 13820.
Filtered: 13820 Ayahs, 31317 Hadiths
🎯 Creating BALANCED validation split (10K examples):
   - Target validation size: 3334 texts (1667 per class)
   - Target validation examples: 10002 examples (5001 per class)
   - Available Ayah texts: 13820
   - Available Hadith texts: 31317
✅ Balanced validation split created (10K examples):
   - Validation Ayah texts: 1,667
   - Validation Hadith texts: 1,667
   - Total validation texts: 3,334
   - Validation examples (3x): 10,002
📊 Training data after removing validation:
   - Training Ayah texts: 12015
   - Training Hadith texts: 29540
   - Training examples (3x): 124665
🔄 Preprocessing training examples...
Training Ayahs: 100%|██████████| 12015/12015 [00:08<00:00, 1427.06it/s]
Training Hadiths: 100%|██████████| 29540/29540 [00:37<00:00, 787.61it/s]
✅ Generated 124665 training examples
❌ Failed to create 0 examples
🔄 Creating generalization-focused validation examples...
Val Ayahs: 100%|██████████| 1667/1667 [00:01<00:00, 1471.95it/s]
Val Hadiths: 100%|██████████| 1667/1667 [00:02<00:00, 797.42it/s]
✅ Created 10002 validation examples.
💾 Saving preprocessing details to CSV files...
✅ CSV files saved.
💾 Saving final tokenized datasets...
Saving the dataset (1/1 shards): 100%
 124665/124665 [00:00<00:00, 472261.81 examples/s]
Saving the dataset (1/1 shards): 100%
 10002/10002 [00:00<00:00, 294298.92 examples/s]
✅ Datasets saved to /content/drive/MyDrive/FinalIslamic/prepros/preprocessed_train_10k_dataset and /content/drive/MyDrive/FinalIslamic/prepros/preprocessed_val_10k_dataset
✅ Preprocessing summary saved to: /content/drive/MyDrive/FinalIslamic/preprocessed_csv_10k/preprocessing_summary_balanced.csv

🎉 LARGE BALANCED PREPROCESSING COMPLETE!
📊 FINAL DATASET STATISTICS:
   Training:   12,015 Ayahs + 29,540 Hadiths = 124,665 examples
   Validation: 1,667 Ayahs + 1,667 Hadiths = 10,002 examples
   Validation balance: 50.0% Ayah, 50.0% Hadith
   🎯 Large validation set: ~10K examples for robust evaluation!

```
