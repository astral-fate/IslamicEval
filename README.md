# Isnad AI: Identifying Islamic Citation in LLM Outputs

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Hugging%20Face-blue)](https://huggingface.co/datasets/FatimahEmadEldin/Isnad-AI-Identifying-Islamic-Citation)
[![Colab](https://img.shields.io/badge/Open%20in-Google%20Colab-orange?logo=google-colab)](https://colab.research.google.com/drive/12uHze_8apJ41_oLe6QJbAjZLYqtp_g3q?usp=sharing)
[![Paper](https://img.shields.io/badge/📄%20Paper-arXiv-red)](https://arxiv.org/abs/your-paper-link)

## Overview

Isnad AI is a specialized system designed to identify character-level spans of Quranic verses (Ayahs) and Prophetic sayings (Hadiths) within Large Language Model outputs. The project was developed for the IslamicEval 2025 Shared Task 1A and achieved an F1 score of 66.97% on the official test set.

### Key Features

- **Rule-based synthetic data generation** for creating high-quality training datasets
- **Fine-tuned AraBERTv2 model** for token classification
- **Comprehensive evaluation framework** with multiple baseline comparisons
- **Character-level span detection** with precise boundary identification

## Architecture

The system employs a multi-stage approach:

1. **Data Preprocessing Pipeline**: Transforms raw Islamic texts into contextualized training data
2. **Template-based Generation**: Creates synthetic examples using curated Arabic templates
3. **Model Training**: Fine-tunes AraBERTv2 for token classification using BIO schema
4. **Span Detection**: Identifies precise character boundaries of religious citations

## Dataset

The training dataset is built from:
- **Quranic Corpus**: 6,236 verses (Ayahs) from the Holy Quran
- **Hadith Collection**: 34,662 prophetic narrations from the Six Major Books

After preprocessing and augmentation:
- **Training Set**: 93,099 examples (20,622 Ayah + 72,477 Hadith)
- **Validation Set**: 40,626 examples (20,313 Ayah + 20,313 Hadith)
- **Total**: 133,725 synthetic examples from 45,137 unique texts

## Quick Start

### Installation

```bash
git clone https://github.com/astral-fate/IslamicEval/
cd IslamicEval
pip install -r requirements.txt
```

### Training Data Generation

```bash
cd "Model Training/Rule-Based"
python Data-preprocessing.py
```

### Model Training

```bash
python Fine-tuning.py
```

### Evaluation

```bash
python Evaluate_finetuned.py
```

## Methodology Comparison

The project implements and compares three approaches:

### 1. Rule-Based Model (Primary)
- **F1 Score**: 66.97% (test) / 65.08% (dev)
- Uses template-based synthetic data generation
- Employs curated Arabic prefixes and suffixes
- Includes text normalization and augmentation

### 2. Database Lookup (Ablation study)
- **F1 Score**: 34.80% (test) / 41.76% (dev)
- Enhanced knowledge base with normalization
- Overlapping text segmentation
- Direct string matching approach

### 3. Basic Fine-tuning (Ablation study)
- **F1 Score**: 44.70% (test) / 53.71% (dev)
- Standard fine-tuning without synthetic augmentation
- Limited training data diversity

## Project Structure

```
isnad-ai/
├── Model Training/
│   ├── Rule-Based/           # Primary rule-based approach
│   │   ├── Data-preprocessing.py
│   │   ├── Fine-tuning.py
│   │   ├── Evaluate_finetuned.py
│   │   └── EDA.py
│   ├── Arabert/             # Ablation study: AraBERT fine-tuning
│   │   ├── Fine-tuning.py
│   │   └── Evalaute_Arabert.py
│   ├── Look-up/             # Ablation study: Database lookup
│   │   ├── look-up method.py
│   │   └── EDA.py
│   └── GPT2/                # Ablation study: Generative augmentation experiments
│       └── syntahtic-data.py
└── README.md
```

## Key Results

| Method | Dev F1 | Test F1 | Improvement |
|--------|--------|---------|-------------|
| Rule-Based Model | 65.08% | 66.97% | - |
| Basic Fine-tuning | 53.71% | 44.70% | +22.27% |
| Database Lookup | 41.76% | 34.80% | +32.17% |

### Performance Analysis

- **Class Imbalance Impact**: 'Neither' class (67.8%), 'Ayah' class (20.2%), 'Hadith' class (12.0%)
- **Main Challenge**: Hadith identification (F1: 0.39) due to textual variation across different books
- **Strength**: Quranic verse detection (F1: 0.67) benefits from textual uniformity

## Technical Details

### Model Configuration
- **Base Model**: AraBERTv2 (aubmindlab/bert-base-arabertv2)
- **Architecture**: Token classification with 5-class BIO schema
- **Training**: 10 epochs with early stopping, learning rate 2×10⁻⁵
- **Hardware**: CUDA-enabled GPU recommended

### Data Augmentation
- Text splitting for long sequences (>25 tokens)
- Arabic script normalization (Tashkeel removal)
- Template-based contextual generation
- Balanced 70/30 train/validation split

## Evaluation Metrics

The system uses **Macro-Averaged F1 Score** computed at the character level:

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Macro-F1 = (F1_Ayah + F1_Hadith + F1_Neither) / 3
```

## Limitations and Future Work

### Current Limitations
- **Hadith Variation**: Performance limited by textual diversity across different Hadith collections
- **Paraphrased Citations**: System designed for exact or near-exact matches
- **Language Scope**: Focused specifically on Arabic Islamic texts

### Proposed Improvements
1. Confine training to single Hadith book (e.g., Sahih al-Bukhari)
2. Implement class-balanced sampling techniques
3. Add fuzzy matching for corrupted/paraphrased citations
4. Enhance synthetic data with textual variations

## Citation

If you use this work, please cite:

```bibtex
coming soon
```

## Resources

- **Dataset**: [Hugging Face](https://huggingface.co/datasets/FatimahEmadEldin/Isnad-AI-Identifying-Islamic-Citation)
- **Model**: [Hugging Face Collection](https://huggingface.co/collections/FatimahEmadEldin/isnad-ai-at-islamiceval-68a64677910651f034b9cdf4)
- **Code**: [GitHub Repository](https://github.com/astral-fate/IslamicEval)
- **Interactive Demo**: [Google Colab](https://colab.research.google.com/drive/12uHze_8apJ41_oLe6QJbAjZLYqtp_g3q?usp=sharing)

## Contributing

Contributions are welcome! Please feel free to submit pull requests, open issues, or suggest improvements.

 

## Acknowledgments

Special thanks to the organizers of IslamicEval 2025 Shared Task for providing this important benchmark for Arabic religious text processing.
