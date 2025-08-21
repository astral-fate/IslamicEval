# Rule-Based error analysis

 
# 📊 ANALYSIS 1: CHARACTER-LEVEL PERFORMANCE
 ```
📈 Character-Level Classification Report:

              precision    recall  f1-score   support

     Neither     0.8489    0.9646    0.9031     45131
        Ayah     0.8098    0.5574    0.6603     12381
      Hadith     0.4723    0.3333    0.3908      7798

    accuracy                         0.8121     65310
   macro avg     0.7103    0.6184    0.6514     65310
weighted avg     0.7965    0.8121    0.7959     65310


📈 Character-Level Confusion Matrix:

```
 
# 📝 ANALYSIS 2: SPAN-LEVEL ERROR LOGGING (FP/FN)
 
 <img width="666" height="547" alt="download" src="https://github.com/user-attachments/assets/ae38f0dc-3f0b-4cbe-bc63-d41a239cdffd" />
<img width="1004" height="624" alt="download" src="https://github.com/user-attachments/assets/afc457dd-0619-4f93-9769-fe02e8adad27" />

 
# 📏 ANALYSIS 3: PERFORMANCE BY SPAN LENGTH
 
```
📈 Descriptive Statistics for Span Lengths:
  - True Positives:
    - Count: 78
    - Mean: 108.59, Median: 97.00
    - Min: 20, Max: 541
  - False Positives:
    - Count: 61
    - Mean: 69.62, Median: 15.00
    - Min: 3, Max: 795
  - False Negatives:
    - Count: 101
    - Mean: 104.78, Median: 70.00
    - Min: 6, Max: 690
 
```
 
# Look-up method




# 📊 ANALYSIS 1: CHARACTER-LEVEL PERFORMANCE
```

📈 Character-Level Classification Report:

              precision    recall  f1-score   support

     Neither     0.7412    0.8774    0.8035     45131
        Ayah     0.7958    0.3043    0.4403     12381
      Hadith     0.3363    0.3084    0.3218      7798

    accuracy                         0.7008     65310
   macro avg     0.6244    0.4967    0.5219     65310
weighted avg     0.7032    0.7008    0.6771     65310


📈 Character-Level Confusion Matrix:
```


# 📝 ANALYSIS 2: SPAN-LEVEL ERROR LOGGING (FP/FN)

Analyzing Spans: 100%|██████████| 50/50 [00:00<00:00, 623.44it/s]✅ Detailed error log saved to: /content/enhanced_lookup_error_analysis.csv
Found 57 TPs, 467 FPs, and 109 FNs.


# ANALYSIS 3: PERFORMANCE BY SPAN LENGTH


<img width="666" height="547" alt="download" src="https://github.com/user-attachments/assets/2c9f0c21-934d-4cfb-ade7-a33b870c21f7" />

<img width="1004" height="624" alt="download" src="https://github.com/user-attachments/assets/71c618d4-1c3d-4497-9793-d277e375c76a" />

```
📈 Descriptive Statistics for Span Lengths:
  - True Positives:
    - Count: 57
    - Mean: 130.49, Median: 93.00
    - Min: 21, Max: 690
  - False Positives:
    - Count: 467
    - Mean: 12.19, Median: 3.00
    - Min: 2, Max: 71
  - False Negatives:
    - Count: 109
    - Mean: 93.11, Median: 70.00
    - Min: 6, Max: 677

📈 Boxplot of Span Lengths by Category:

```



# Basic fine-tuning 



 
# 📊 ANALYSIS 1: CHARACTER-LEVEL PERFORMANCE

```
📈 Character-Level Classification Report:

              precision    recall  f1-score   support

     Neither     0.6933    0.9464    0.8004     45131
        Ayah     0.8715    0.0876    0.1593     12381
      Hadith     0.0390    0.0123    0.0187      7798

    accuracy                         0.6721     65310
   macro avg     0.5346    0.3488    0.3261     65310
weighted avg     0.6490    0.6721    0.5855     65310


 ```
 

# 📏 ANALYSIS 3: PERFORMANCE BY SPAN LENGTH

<img width="666" height="547" alt="download" src="https://github.com/user-attachments/assets/a3bfae91-10bf-4c6c-9ebd-35541164464f" />

<img width="1004" height="624" alt="download" src="https://github.com/user-attachments/assets/f0171720-7cbf-4c59-8a37-28848fa41c3d" />

```

📈 Descriptive Statistics for Span Lengths:
  - True Positives:
    - Count: 12
    - Mean: 111.33, Median: 100.50
    - Min: 33, Max: 247
  - False Positives:
    - Count: 13
    - Mean: 184.46, Median: 17.00
    - Min: 6, Max: 1488
  - False Negatives:
    - Count: 173
    - Mean: 110.21, Median: 87.00
    - Min: 6, Max: 690

📈 Boxplot of Span Lengths by Category:

```
