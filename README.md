# Improving-CBM-with-VLMs

This repository contains the implementation of the project for `Advance Topics in Machine Learning and Optimization'.
The goal is to use VLMs to improve CBMs, trained and tested on a syntetic dataset, Shapes3d, and a real one, CUB-200


## Project Structure

```
Improving-CBM-with-VLMs
├── cub
│   ├── cub_enhanced_llava_feedback_adjust.json
│   ├── cub_training_history.json
│   └── results.out
├── shapes3d
│   ├── results.out
│   ├── shapes3d_enhanced_llava_feedback_adjust.json
│   ├── shapes3d_training_history.json
│   ├── test_split_cl_clean.npy
│   ├── test_split_cl.npy
│   ├── test_split_imgs_clean.npy
│   ├── test_split_imgs.npy
│   ├── train_split_cl_clean.npy
│   ├── train_split_cl.npy
│   ├── train_split_imgs_clean.npy
│   ├── train_split_imgs.npy
│   ├── val_split_cl_clean.npy
│   ├── val_split_cl.npy
│   ├── val_split_imgs_clean.npy
│   └── val_split_imgs.npy
│ 
├── datasets.py     # Dataset loading and balancing utilities
├── llava.py        # LLaVA feedback and annotation logic
├── model.py        # Model architectures (CBM, predictors)
├── main.py         # Main training, evaluation, and pipeline script
└── readme.md       # This file

```


## Setup

**Download datasets:**
   - **SHAPES3D:** Place the dataset in a folder named `shapes3d` in your working directory.
   - **CUB-200-2011:** Place the dataset in a folder named `CUB_200_2011` in your working directory.

**Install dependencies:**
  - make sure to have installed all needed packages
     - PyTorch
     - torchvision
     - scikit-learn
     - tqdm
     - transformers
     - pillow
     - numpy
     - pandas
---

## Usage

Run experiments via the main script:

```bash
python main.py --dataset [shapes3d|cub] --data_dir [OUTPUT_DIR] [other options]
```

### Example

```bash
python main.py --dataset shapes3d --data_dir results_shapes3d
python main.py --dataset cub --data_dir results_cub
```

### Key Arguments

- `--dataset`: Dataset to use (`shapes3d` or `cub`)
- `--data_dir`: Output directory for results and logs
- `--epochs_3d`: Number of training epochs for SHAPES3D (default: 15)
- `--epochs_cub`: Number of training epochs for CUB (default: 30)
- `--patience_3d`: Early stopping patience for SHAPES3D (default: 10)
- `--patience_cub`: Early stopping patience for CUB (default: 15)
- `--batch_size_3d`: Batch size for SHAPES3D (default: 16)
- `--batch_size_cub`: Batch size for CUB (default: 32)
- `--train_samples_3d`: Number of SHAPES3D training samples (default: 150)
- `--val_samples_3d`: Number of SHAPES3D validation samples (default: 150)
- `--test_samples_3d`: Number of SHAPES3D test samples (default: 100)
- `--min_confidence`: Minimum confidence for LLaVA feedback (default: 0.5)


## Pipeline Overview

1. **Training:**  
   Trains a Concept Bottleneck Model (CBM) on the selected dataset.

2. **Evaluation:**  
   Evaluates the trained model on the test set before fine-tuning.

3. **LLaVA Feedback Collection:**  
   Collects feedback using LLaVA strategies on the validation set.

4. **Fine-tuning:**  
   Fine-tunes the CBM using LLaVA feedback and re-evaluates on the test set.

5. **Evaluation - after finetuning**  
   Evaluates the model on the test set.
