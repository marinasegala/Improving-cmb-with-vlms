import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
import os
import itertools
import json
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import logging
from typing import List, Dict, Optional
from transformers import BitsAndBytesConfig

from datasets import SHAPES3DMini, CUBDataset, CUB_CONCEPT_NAMES, SHAPES3D_CONCEPT_NAMES
from model import ModelXtoCtoY
from llava import LLaVAFeedbackDataset, EnhancedLLaVAAnnotator


# =================
# SUPPORT FN 
# =================
def tensor_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_serializable(v) for v in obj]
    else:
        return obj

# =================
# TRAINING FN
# =================
def train_concept_bottleneck_model(model, train_loader, val_loader, data_dir, 
                                   learning_rate, weight_decay, num_epochs=50, 
                                   concept_weight=1.0, label_weight=1.0, device='cuda', patience=10):
    """
    Train the concept bottleneck model
    Args:
        model: The model to train
        train_loader: DataLoader for the training set
        val_loader: DataLoader for the validation set
        data_dir: Directory for saving
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        num_epochs: Number of epochs to train
        concept_weight: Weight for the concept loss
        label_weight: Weight for the label loss
        device: Device to train on (e.g., 'cuda' or 'cpu')
        patience: Patience for early stopping
    """

    model.to(device)

    # Loss functions
    concept_criterion = nn.BCELoss() 
    # Compute class weights for CrossEntropyLoss
    label_counts = torch.zeros(2)
    for _, _, labels in train_loader:
        for l in labels:
            label_counts[int(l)] += 1
    class_weights = (1.0 / (label_counts + 1e-6))
    class_weights = class_weights / class_weights.sum() * 2  # Normalize to sum to num_classes
    class_weights = class_weights.to(torch.float32).to(device)
    label_criterion = nn.CrossEntropyLoss(weight=class_weights) 


    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=weight_decay)
    if 'cub' in data_dir:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    concept_accuracies = []
    f1_scores = []
    best_f1 = -1.0
    best_accuracy = 0.0
    best_epoch = 0
    counter = 0
    image_save_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_concept_loss = 0.0
        train_label_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]', leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            images, concepts, labels = batch
            images, concepts, labels = images.to(device), concepts.to(device), labels.to(device)
            if concepts.dtype != torch.float32:
                concepts = concepts.float()
            labels = labels.long()
            optimizer.zero_grad()

            # Forward pass
            pred_concepts, pred_labels = model(images)

            # Calculate losses
            concept_loss = concept_criterion(pred_concepts, concepts)
            label_loss = label_criterion(pred_labels, labels)

            total_loss = concept_weight * concept_loss + label_weight * label_loss

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()
            train_concept_loss += concept_loss.item()
            train_label_loss += label_loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_concept_loss = 0.0
        val_label_loss = 0.0
        correct_labels = 0
        correct_concepts = 0
        total_samples = 0
        total_concept_predictions = 0
        all_true_labels = []
        all_pred_labels = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch} [VAL]', leave=False)
            for index, batch in enumerate(val_pbar):
                images, concepts, labels = batch
                images, concepts, labels = images.to(device), concepts.to(device), labels.to(device)
                batch_size = images.shape[0]
                images_dir = os.path.join(data_dir, 'images')
                os.makedirs(images_dir, exist_ok=True)
                for i in range(batch_size):
                    img_tensor = images[i].cpu()
                    if 'shapes3d' in data_dir: 
                        img_tensor = (img_tensor * 0.5) + 0.5
                    else: 
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_tensor = (img_tensor * std) + mean
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
                    img.save(os.path.join(images_dir, f'image_{image_save_counter:05d}.png'))
                    image_save_counter += 1
                if concepts.dtype != torch.float32:
                    concepts = concepts.float()
                labels = labels.long()
                pred_concepts, pred_labels = model(images)

                concept_loss = concept_criterion(pred_concepts, concepts)
                label_loss = label_criterion(pred_labels, labels)

                val_loss += (concept_weight * concept_loss + label_weight * label_loss).item()
                val_concept_loss += concept_loss.item()
                val_label_loss += label_loss.item()

                # Calculate accuracies
                _, predicted_labels = torch.max(pred_labels.data, 1)
                total_samples += labels.size(0)
                correct_labels += (predicted_labels == labels).sum().item()

                all_true_labels.extend(labels.cpu().numpy().tolist())
                all_pred_labels.extend(predicted_labels.cpu().numpy().tolist())
                
                predicted_concepts = (pred_concepts > 0.5).float()
                concept_matches = (predicted_concepts == concepts).float()
                correct_concepts += concept_matches.sum().item()
                total_concept_predictions += concept_matches.numel()

                val_pbar.set_postfix({
                    'Loss': f'{val_loss/(total_samples//labels.size(0)):.4f}',
                    'Acc': f'{100*correct_labels/total_samples:.1f}%'
                })

            # F1-score
            f1_macro = f1_score(all_true_labels, all_pred_labels, average='macro')
            f1_pos = f1_score(all_true_labels, all_pred_labels, pos_label=1, average='binary')
            f1_scores.append(f1_pos)
            print(f"Val F1-macro: {f1_macro:.3f} | F1-pos: {f1_pos:.3f}")

        # Update learning rate
        if 'cub' in data_dir:
            scheduler.step()
        else:
            scheduler.step(val_loss)

        # Record metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_labels / total_samples
        concept_accuracy = 100 * correct_concepts / total_concept_predictions

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        concept_accuracies.append(concept_accuracy)

        # early stopping 
        if f1_pos > best_f1:
            best_f1 = f1_pos
            best_accuracy = val_accuracy
            best_epoch = epoch
            counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'concept_accuracies': concept_accuracies,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'concept_accuracy': concept_accuracy,
                'f1_macro': f1_macro,
                'f1_pos': f1_pos
            }
            torch.save(checkpoint, os.path.join(data_dir, f'best_model_epoch_{best_epoch}.pth'))
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"\n Early stopping at epoch {epoch}")
                print(f"   Best epoch: {best_epoch}")
                print(f"   Best val F1-pos: {best_f1:.4f}")
                print(f"   Best val accuracy: {best_accuracy:.2f}%")
                break

    
    # Save final model
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'concept_accuracies': concept_accuracies
    }
    torch.save(final_checkpoint, os.path.join(data_dir, 'final_model.pth'))
    print("Final model saved as 'final_model.pth'")

    best_epoch_idx = np.argmax(f1_scores)
    print(f"\n TRAINING SUMMARY:")
    print(f"   Completed epochs: {len(train_losses)}")
    print(f"   Best epoch: {best_epoch_idx}")
    print(f"   Best val loss: {val_losses[best_epoch_idx]:.4f}")
    print(f"   Best val accuracy: {val_accuracies[best_epoch_idx]:.2f}%")
    print(f"   Best concept accuracy: {concept_accuracies[best_epoch_idx]:.2f}%")
    print(f"   Best f1: {f1_scores[best_epoch_idx]:.2f}%")

    return train_losses, val_losses, val_accuracies, concept_accuracies, best_epoch_idx, f1_scores


# ===========================
# LLAVA USAGE
# ===========================

def enhanced_feedback_collection(logger, output_dir, cbm_model, dataloader, 
                               llava_annotator: EnhancedLLaVAAnnotator, 
                               dataset_type: str, device: str,
                               concept_names: List[str],
                               model_path: Optional[str] = None,
                               min_confidence: float = 0.5,
                               use_uncertainty_sampling: bool = True) -> List[Dict]:
    """
    Enhanced feedback collection with multiple strategies
    
    Args:
        logger: Logger for logging information
        output_dir: Directory to save output files
        cbm_model: The CBM model to use for predictions
        dataloader: DataLoader for the input data
        llava_annotator: The LLaVA annotator for feedback collection
        dataset_type: Type of the dataset (e.g., "shapes3d" or "cub")
        device: Device to run the model on (e.g., "cuda" or "cpu")
        concept_names: List of concept names for the task
        model_path: Optional path to a pre-trained model
        min_confidence: Minimum confidence threshold for predictions
        use_uncertainty_sampling: Whether to use uncertainty sampling

    Returns:
        List of feedback data collected from the LLaVA annotator
    """
    
    if model_path:
        logger.info(f"Loading CBM model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            cbm_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            cbm_model.load_state_dict(checkpoint)
    
    cbm_model.to(device)
    cbm_model.eval()
    
    # collect all CBM predictions
    all_predictions = {}
    feedback_image_counter = 0  # Global counter for feedback images
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, concepts, labels = batch
            
            images = images.to(device)
            pred_concepts, pred_labels = cbm_model(images)
            
            for i in range(images.shape[0]):
                sample_id =f'{feedback_image_counter:05d}'
                all_predictions[sample_id] = {
                    'image': images[i],
                    'pred_concepts': pred_concepts[i].cpu().numpy(),
                    'pred_labels': torch.argmax(pred_labels[i]).item(),
                    'concept_probs': pred_concepts[i].cpu().numpy(),
                    'label_probs': torch.softmax(pred_labels[i], dim=0).cpu().numpy(),
                    'true_concepts': concepts[i].numpy(),
                    'true_labels': labels[i].item()
                }
                feedback_image_counter += 1
    if use_uncertainty_sampling:
        uncertain_samples = llava_annotator.uncertainty_based_sampling(
            all_predictions, uncertainty_threshold=0.3, max_samples=300
        )
        ids_to_evaluate = [s['sample_id'] for s in uncertain_samples]
        logger.info(f"Selected {len(ids_to_evaluate)} uncertain samples for detailed LLaVA evaluation")
    else:
        ids_to_evaluate = list(all_predictions.keys())[:500] 
    
    # Enhanced feedback collection
    feedback_data = []
    for sample_id in ids_to_evaluate:
        sample = all_predictions[sample_id]
        
        # Convert tensor image to PIL
        image_tensor = sample['image']
        if image_tensor.shape[0] == 3:  # CHW
            image_tensor = image_tensor.permute(1, 2, 0)  # HWC
        
        # Denormalize based on dataset
        if dataset_type.lower() == "shapes3d":
            image_tensor = (image_tensor * 0.5) + 0.5  # From [-1,1] to [0,1]
        else: 
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            mean = mean.to(image_tensor.device)
            std = std.to(image_tensor.device)
            image_tensor = image_tensor * std + mean
        
        image_tensor = torch.clamp(image_tensor, 0, 1)
        image_pil = Image.fromarray((image_tensor.cpu() * 255).byte().numpy())
        
        enhanced_feedback = llava_annotator.multi_prompt_evaluation(
            image_pil, sample['pred_labels'], sample['concept_probs'],
            concept_names, dataset_type, sample['true_labels']
        )
        
        # Combine all feedback
        comprehensive_feedback = {
            'sample_id': sample_id,
            'image_path': f"{output_dir}/images/image_{sample_id}.png",
            'cbm_prediction': sample['pred_labels'],
            'cbm_concepts': sample['pred_concepts'].tolist(),
            'concept_probabilities': sample['concept_probs'].tolist(),
            'true_label': sample['true_labels'],
            'true_concepts': sample['true_concepts'].tolist(),
            # Metrics
            'concept_accuracy': float(np.mean(sample['true_concepts'] == sample['pred_concepts'])),
            'cbm_label_correct': bool(sample['true_labels'] == sample['pred_labels']),

            # Enhanced LLaVA feedback
            'llava_strategies': enhanced_feedback['strategies'],
            'llava_final_prediction': enhanced_feedback['final_prediction'],
            'llava_confidence': enhanced_feedback['confidence'],
            'needs_correction': enhanced_feedback['needs_correction'],
        }
        
        # Save image
        # image_pil.save(comprehensive_feedback['image_path'])
        feedback_data.append(comprehensive_feedback)
    
    # Save comprehensive feedback
    feedback_file = f"{output_dir}/{dataset_type}_enhanced_llava_feedback_adjust.json"
    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f, indent=2)
    
    logger.info(f"Saved {len(feedback_data)} enhanced feedback samples to {feedback_file}")
    
    # Print statistics
    print_enhanced_feedback_statistics(feedback_data, dataset_type, min_confidence)
    
    return feedback_data

def print_enhanced_feedback_statistics(feedback_data: List[Dict], dataset_type: str, min_confidence: float):
    """
    Print comprehensive statistics about LLaVA feedback
    
    Args:
        feedback_data: List of feedback data samples
        dataset_type: Type of the dataset (e.g., "shapes3d" or "cub")
        min_confidence: Minimum confidence threshold for filtering samples
    """
    
    print(f"\n=== ENHANCED LLAVA FEEDBACK STATISTICS ({dataset_type.upper()}) ===")
    
    total_samples = len(feedback_data)
    corrections = sum(1 for sample in feedback_data if sample['needs_correction'])
    high_conf_corrections = sum(1 for sample in feedback_data 
                               if sample['needs_correction'] and sample['llava_confidence'] >= min_confidence)
    
    print(f"Total samples evaluated: {total_samples}")
    print(f"Corrections suggested: {corrections} ({100*corrections/total_samples:.1f}%)")
    print(f"High-confidence corrections: {high_conf_corrections} ({100*high_conf_corrections/total_samples:.1f}%)")
    
    # Strategy agreement analysis
    strategy_agreements = {}
    for sample in feedback_data:
        strategies = sample['llava_strategies']
        cbm_pred = sample['cbm_prediction']
        
        for strategy_name, strategy_result in strategies.items():
            if strategy_name not in strategy_agreements:
                strategy_agreements[strategy_name] = {'agree': 0, 'disagree': 0}
            
            # Check if strategy agrees with CBM
            if strategy_name == 'direct':
                agrees = strategy_result['prediction'] == cbm_pred
            elif strategy_name == 'complete':
                agrees = strategy_result['prediction'] == cbm_pred
            elif strategy_name == 'guided':
                agrees = strategy_result['prediction'] == cbm_pred
            
            if agrees:
                strategy_agreements[strategy_name]['agree'] += 1
            else:
                strategy_agreements[strategy_name]['disagree'] += 1
    
    print(f"\nStrategy Agreement with CBM:")
    for strategy, counts in strategy_agreements.items():
        total = counts['agree'] + counts['disagree']
        if total > 0:
            agreement_rate = 100 * counts['agree'] / total
            print(f"  {strategy}: {agreement_rate:.1f}% agreement ({counts['agree']}/{total})")
    
    # Confidence distribution
    confidences = [sample['llava_confidence'] for sample in feedback_data]
    print(f"\nLLaVA Confidence Distribution:")
    print(f"  Mean: {np.mean(confidences):.3f}")
    print(f"  Std: {np.std(confidences):.3f}")
    print(f"  High confidence (>{min_confidence}): {sum(1 for c in confidences if c > min_confidence)} samples")
    print(f"  Low confidence (<0.4): {sum(1 for c in confidences if c < 0.4)} samples")


# ================================
# FINE-TUNING WITH LLAVA FEEDBACK
# ================================

def create_mixed_finetune_loader(feedback_dataset, original_train_dataset, batch_size=16, feedback_ratio=0.5):
    """
    Create a DataLoader that mixes feedback and original training data for fine-tuning.
    Args:
        feedback_dataset: Dataset containing feedback samples
        original_train_dataset: Dataset containing original training samples
        batch_size: Total batch size for fine-tuning
        feedback_ratio: fraction of each batch from feedback data (e.g., 0.5 means half feedback, half original)

    Returns:
        A mixed DataLoader for fine-tuning
    """
    feedback_size = int(batch_size * feedback_ratio)
    original_size = batch_size - feedback_size

    feedback_loader = DataLoader(feedback_dataset, batch_size=feedback_size, shuffle=False, drop_last=False)
    original_loader = DataLoader(original_train_dataset, batch_size=original_size, shuffle=False, drop_last=False)

    # Use image tensors as unique identifiers (by hash)
    def image_hash(img_tensor):
        return hash(img_tensor.cpu().numpy().tobytes())

    feedback_image_hashes = set()
    for i in range(len(feedback_dataset)):
        img = feedback_dataset[i]['image']
        feedback_image_hashes.add(image_hash(img))

    only_original_indices = []
    for i in range(len(original_train_dataset)):
        img = original_train_dataset[i][0]
        if image_hash(img) not in feedback_image_hashes:
            only_original_indices.append(i)


    def mixed_batch_generator():
        # all feedback samples
        feedback_indices = list(range(len(feedback_dataset)))
        for i in range(0, len(feedback_indices), feedback_size):
            batch_idxs = feedback_indices[i:i+feedback_size]
            batch = [feedback_dataset[j] for j in batch_idxs]
            images = torch.stack([b['image'] for b in batch])
            labels = torch.stack([b['corrected_label'] for b in batch])
            concepts = torch.stack([b['true_concepts'] for b in batch])
            confidence = torch.stack([b['confidence'] for b in batch])
            yield {'image': images, 'label': labels, 'concepts': concepts, 'confidence': confidence}

        # original samples not present in feedback
        for i in range(0, len(only_original_indices), original_size):
            batch_idxs = only_original_indices[i:i+original_size]
            batch = [original_train_dataset[j] for j in batch_idxs]
            images = torch.stack([b[0] for b in batch])
            labels = torch.stack([b[2] for b in batch])
            concepts = torch.stack([b[1] for b in batch])
            confidence = torch.ones(len(batch), dtype=torch.float32)
            yield {'image': images, 'label': labels, 'concepts': concepts, 'confidence': confidence}

    return mixed_batch_generator

def finetune_cbm_with_llava_feedback(output_dir, logger, cbm_model, feedback_data: List[Dict], 
                                   original_train_loader, val_loader, num_epochs,
                                   learning_rate: float = 1e-3, weight_decay: float = 1e-3, device: str = 'cuda',
                                   concept_weight: float = 1.0, label_weight: float = 1.0, min_confidence: float = 0.6,
                                   patience: int = 5, selected_dataset: str = None) -> Dict:
    """
    Fine-tune CBM model using LLaVA feedback
    
    Args:
        output_dir: Directory to save output files
        logger: Logger for logging information
        cbm_model: Pre-trained CBM model
        feedback_data: LLaVA feedback annotations
        original_train_loader: Original training data loader
        val_loader: Validation data loader
        num_epochs: Number of fine-tuning epochs
        learning_rate: Learning rate for fine-tuning
        weight_decay: Weight decay for optimizer
        device: Device for training
        concept_weight: Weight for concept loss (aligned with original training)
        label_weight: Weight for label loss (aligned with original training)
        min_confidence: Minimum confidence threshold for using feedback
        patience: Patience for early stopping
        selected_dataset: Name of the selected dataset

    Returns:
        Dictionary with fine-tuning results 
    """
    cbm_model.to(device)
    
    # Create feedback dataset
    feedback_transform = original_train_loader.dataset.transform
    feedback_dataset = LLaVAFeedbackDataset(feedback_data, feedback_transform, logger, min_confidence)
    
    correction_summary = feedback_dataset.get_correction_summary()
    logger.info("=== LLaVA Feedback Dataset Summary ===")
    logger.info(f"Total corrections: {correction_summary['total_corrections']}")
    logger.info(f"Average confidence: {correction_summary.get('avg_confidence', 0):.3f}")
    logger.info(f"Confidence std: {correction_summary.get('confidence_std', 0):.3f}")
    logger.info(f"Correction patterns: {correction_summary.get('correction_patterns', {})}")
    
    if len(feedback_dataset) == 0:
        logger.warning("No high-confidence corrections found. Skipping fine-tuning.")
        return {'message': 'No corrections to apply', 'correction_summary': correction_summary}
    
    
    feedback_loader = DataLoader(feedback_dataset, batch_size=8, shuffle=True) #num_workers=2

    # Setup for fine-tuning
    concept_criterion = nn.BCELoss() 
                                       
    label_counts = torch.zeros(2)
    for _, _, labels in original_train_loader:
        for l in labels:
            label_counts[int(l)] += 1
    class_weights = (1.0 / (label_counts + 1e-6))
    class_weights = class_weights / class_weights.sum() * 2
    class_weights = class_weights.to(torch.float32).to(device)
    label_criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Mix 25% feedback and 75% original data in each batch for fine-tuning
    feedback_loader = itertools.cycle(
        create_mixed_finetune_loader(
            feedback_dataset, original_train_loader.dataset, batch_size=8, feedback_ratio=0.4
        )()
    )
    
    optimizer = optim.Adam(cbm_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if 'cub' in selected_dataset:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    concept_accuracies = []
    f1_scores = []
    best_f1 = -1.0
    best_accuracy = 0.0
    best_epoch = 0
    counter = 0
    
    logger.info(f"Starting CBM fine-tuning with {len(feedback_dataset)} correction samples")

    for epoch in range(num_epochs):
        cbm_model.train()
        
        train_loss = 0.0
        train_concept_loss = 0.0
        train_label_loss = 0.0
        
        num_batches = len(original_train_loader)
        train_pbar = tqdm(range(num_batches), desc=f'Epoch {epoch} [TRAIN]', leave=False)
        for batch_idx in train_pbar:
            feedback_batch = next(feedback_loader)
            images = feedback_batch['image'].to(device)
            corrected_labels = feedback_batch['label'].to(device).long()
            true_concepts = feedback_batch['concepts'].to(device).float()
            confidence_weights = feedback_batch['confidence'].to(device)

            optimizer.zero_grad()
            
            # Forward pass
            pred_concepts, pred_labels = cbm_model(images)
            
            # Calculate losses 
            concept_loss = concept_criterion(pred_concepts, true_concepts)
            label_loss = label_criterion(pred_labels, corrected_labels)
            
            total_loss = concept_weight * concept_loss + label_weight * label_loss
            weighted_total_loss = total_loss * confidence_weights.mean() 
            
            # Backward pass
            weighted_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(cbm_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += weighted_total_loss.item()
            train_concept_loss += concept_loss.item()
            train_label_loss += label_loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{weighted_total_loss.item():.4f}',
                'C_Loss': f'{concept_loss.item():.4f}',
                'L_Loss': f'{label_loss.item():.4f}'
            })
        
        # === VALIDATION PHASE ===
        cbm_model.eval()
        val_loss = 0.0
        val_concept_loss = 0.0
        val_label_loss = 0.0
        correct_labels = 0
        correct_concepts = 0
        total_samples = 0
        total_concept_predictions = 0
        all_true_labels = []
        all_pred_labels = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch} [VAL]', leave=False)

            for batch_idx, batch in enumerate(val_pbar):
                images, concepts, labels = batch
                images, concepts, labels = images.to(device), concepts.to(device), labels.to(device)
                if concepts.dtype != torch.float32:
                    concepts = concepts.float()
                labels = labels.long()

                pred_concepts, pred_labels = cbm_model(images)
                
                # Calculate losses
                concept_loss = concept_criterion(pred_concepts, concepts)
                label_loss = label_criterion(pred_labels, labels)

                val_loss += (concept_weight * concept_loss + label_weight * label_loss).item()
                val_concept_loss += concept_loss.item()
                val_label_loss += label_loss.item()

                # Calculate accuracies
                _, predicted_labels = torch.max(pred_labels.data, 1)
                total_samples += labels.size(0)
                correct_labels += (predicted_labels == labels).sum().item()

                all_true_labels.extend(labels.cpu().numpy().tolist())
                all_pred_labels.extend(predicted_labels.cpu().numpy().tolist())

                predicted_concepts = (pred_concepts > 0.5).float()
                concept_matches = (predicted_concepts == concepts).float()
                correct_concepts += concept_matches.sum().item()
                total_concept_predictions += concept_matches.numel()

                val_pbar.set_postfix({
                    'Loss': f'{val_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100*correct_labels/total_samples:.1f}%'
                })

            # F1-score calculation 
            f1_macro = f1_score(all_true_labels, all_pred_labels, average='macro')
            f1_pos = f1_score(all_true_labels, all_pred_labels, pos_label=1, average='binary')
            f1_scores.append(f1_pos)
            logger.info(f"Val F1-macro: {f1_macro:.3f} | F1-pos: {f1_pos:.3f}")

        scheduler.step(val_loss)
        # Record metrics 
        avg_train_loss = train_loss / num_batches
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_labels / total_samples
        concept_accuracy = 100 * correct_concepts / total_concept_predictions

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        concept_accuracies.append(concept_accuracy)
        
        # Early stopping based on F1-score 
        if f1_pos > best_f1:
            best_f1 = f1_pos
            best_accuracy = val_accuracy
            best_epoch = epoch
            counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': cbm_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'concept_accuracies': concept_accuracies,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'concept_accuracy': concept_accuracy,
                'f1_macro': f1_macro,
                'f1_pos': f1_pos,
                'feedback_samples_used': len(feedback_dataset),
                'correction_summary': correction_summary
            }
            best_model_path = os.path.join(output_dir, f'best_finetuned_model_epoch_{best_epoch}.pth')
            torch.save(checkpoint, best_model_path)
            logger.info(f"New best model saved with F1: {best_f1:.4f}")
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                logger.info(f"Best epoch: {best_epoch}, Best F1: {best_f1:.4f}, Best accuracy: {best_accuracy:.2f}%")
                break

        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f} (Concept: {train_concept_loss/num_batches:.4f}, "
                   f"Label: {train_label_loss/num_batches:.4f})")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, "
                   f"Concept Accuracy: {concept_accuracy:.2f}%")
    
    # Save final model 
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': cbm_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'concept_accuracies': concept_accuracies,
        'f1_scores': f1_scores,
        'feedback_samples_used': len(feedback_dataset),
        'correction_summary': correction_summary
    }
    
    final_model_path = os.path.join(output_dir, 'final_finetuned_model.pth')
    torch.save(final_checkpoint, final_model_path)
    logger.info(f"Final fine-tuned model saved to: {final_model_path}")

    # Summary
    best_epoch_idx = np.argmax(f1_scores) if f1_scores else 0
    print(f"\n FINE-TUNING SUMMARY:")
    print(f"   Completed epochs: {len(train_losses)}")
    print(f"   Best epoch: {best_epoch_idx}")
    print(f"   Best val loss: {val_losses[best_epoch_idx]:.4f}")
    print(f"   Best val accuracy: {val_accuracies[best_epoch_idx]:.2f}%")
    print(f"   Best concept accuracy: {concept_accuracies[best_epoch_idx]:.2f}%")
    print(f"   Best F1: {f1_scores[best_epoch_idx]:.4f}" if f1_scores else "   No F1 scores")
    print(f"   Corrections applied: {len(feedback_dataset)}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'concept_accuracies': concept_accuracies,
        'f1_scores': f1_scores,
        'best_epoch_idx': best_epoch_idx,
        'best_f1': best_f1,
        'best_accuracy': best_accuracy,
        'corrections_applied': len(feedback_dataset),
        'correction_summary': correction_summary,
        'final_model_path': final_model_path,
        'best_model_path': best_model_path
    }

def evaluate_finetuned_model(cbm_model, test_loader, output_dir, device: str = 'cuda', logger = None) -> Dict:
    """
    Evaluate the fine-tuned CBM model on test set
    
    Args:
        cbm_model: Fine-tuned CBM model
        test_loader: DataLoader for the test set
        output_dir: Directory to save evaluation results
        device: Device for evaluation
        logger: Optional logger for logging information

    Returns:
        Dictionary with evaluation metrics
    """
    cbm_model.to(device)
    cbm_model.eval()
    
    correct = 0
    total = 0
    concept_correct = 0
    concept_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for idx, (images, concepts, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images, concepts, labels = images.to(device), concepts.to(device), labels.to(device)
            
            pred_concepts, pred_labels = cbm_model(images)
            pred_concepts, pred_labels = cbm_model(images)
            
            _, predicted = torch.max(pred_labels.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pred_concepts_binary = (torch.sigmoid(pred_concepts) > 0.5).float()
            concept_correct += (pred_concepts_binary == concepts).sum().item()
            concept_total += concepts.numel()
    
    # Calculate metrics
    accuracy = correct / total
    
    report = classification_report(all_labels, all_predictions, output_dict=True)
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'test_accuracy': accuracy,
        'total_samples': total,
        'correct_predictions': correct,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    print("=== Test Evaluation Results ===")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Total Samples: {total}")
    print(f"-------------")
    print(f"Test F1-macro: {report['macro avg']['f1-score']:.4f}")
    print(f"Test F1-weighted: {report['weighted avg']['f1-score']:.4f}")
    print(f"Test F1 - class 1: {report['1.0']['f1-score']:.4f}")
    print(f"Test F1 - class 0: {report['0.0']['f1-score']:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return results

# ================================
# MAIN SCRIPT
# ================================

def main(args):
    selected_dataset = args.dataset.lower()
    min_confidence = args.min_confidence
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if selected_dataset == 'shapes3d':
        concept_names=SHAPES3D_CONCEPT_NAMES
        output_dir = args.data_dir + '_shapes3d' if 'shapes3d' not in args.data_dir else args.data_dir
        epoch = args.epochs_3d
        epochs_finetuning = 10
        patience = args.patience_3d
        batch_size = args.batch_size_3d
        samples_trains = args.train_samples_3d
        samples_val = args.val_samples_3d
        samples_test = args.test_samples_3d

        lr_train = 0.0005
        weight_decay_train = 1e-3
        lr_finetuning = 5e-5
        weight_decay_finetuning = 1e-3

        os.makedirs(output_dir, exist_ok=True)
        
        # =============================
        # SHAPES3D TRAINING
        # =============================
        
        print("=" * 50)
        print("TRAINING ON SHAPES3D DATASET")
        print("=" * 50)
        
        class AddGaussianNoise(object):
            def __init__(self, mean=0., std=0.1):
                self.mean = mean
                self.std = std
            def __call__(self, tensor):
                return tensor + torch.randn_like(tensor) * self.std + self.mean

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2),
            transforms.ToTensor(),
            AddGaussianNoise(0., 0.15),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        transform_val = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])

        # Load Shapes3D dataset
        train = SHAPES3DMini(root='shapes3d', split='train', transform=transform_train, subset_indices=[0,samples_trains], output_dir=output_dir)
        val = SHAPES3DMini(root='shapes3d', split='val',transform=transform_val, subset_indices=[0,samples_val], output_dir=output_dir)
        test = SHAPES3DMini(root='shapes3d', split='test',transform=transform_val, subset_indices=[0,samples_test], output_dir=output_dir)

        # Create dataloaders
        # Perturb concepts in the training set (e.g., 20% noise)
        concept_noise_prob = 0.2  # 20% of concept bits will be flipped

        # Access the concepts array in your dataset
        concepts = train.dataset.concepts  # shape: (N, num_concepts)
        mask = np.random.rand(*concepts.shape) < concept_noise_prob
        concepts[mask] = 1 - concepts[mask]
        train.dataset.concepts = concepts  # Overwrite with noisy concepts
        
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

        model = ModelXtoCtoY(num_concepts=14, num_classes=2)

    if selected_dataset == 'cub':
        concept_names=CUB_CONCEPT_NAMES
        output_dir = args.data_dir + '_cub' if 'cub' not in args.data_dir else args.data_dir
        epoch = args.epochs_cub
        epochs_finetuning = epoch
        patience = args.patience_cub
        batch_size = args.batch_size_cub

        lr_train = 0.001
        weight_decay_train = 1e-2
        lr_finetuning = 1e-4
        weight_decay_finetuning = 5e-4

        os.makedirs(output_dir, exist_ok=True)
        
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load CUB200 dataset (you need to download CUB-200-2011 dataset)
        train = CUBDataset('CUB_200_2011', transform=transform_val, split='train')
        val = CUBDataset('CUB_200_2011', transform=transform_val, split='val')
        test = CUBDataset('CUB_200_2011', transform=transform_val, split='test')

        # Create dataloaders
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

        model = ModelXtoCtoY(num_concepts=12, num_classes=2)

    # ==============================
    #  TRAIN PHASE
    # ==============================
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    log_file = f"runs_{selected_dataset}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    train_losses, val_losses, val_accuracies, concept_accuracies, best_epoch, f1_scores = train_concept_bottleneck_model(
        model, train_loader, val_loader, output_dir, lr_train, weight_decay_train,
        num_epochs=epoch, device=device, patience=patience
    )
    with open(os.path.join(output_dir, f'{selected_dataset}_training_history.json'), 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'concept_accuracies': concept_accuracies,
            'f1_scores': f1_scores
        }, f, indent=2)
        
    print("Training complete! Best model saved as 'best_model.pth'")

    best_model_path = os.path.join(output_dir, f'best_model_epoch_{best_epoch}.pth')

    print(f"{selected_dataset} model setup complete!")

    # ==============================
    # PRE-FINETUNING TEST EVALUATION
    # ==============================
    print("\nEvaluating on test set BEFORE finetuning...")
    pre_finetune_results = evaluate_finetuned_model(model, test_loader, output_dir, logger=logger)
    logger.info(f"Pre-finetuning test results: {pre_finetune_results}")

    # ==============================
    #  LLaVA PHASE
    # ==============================

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Initialize LLaVA
    logger.info("Initializing LLaVA annotator...")

    llava_annotator = EnhancedLLaVAAnnotator(
        model_id="llava-hf/llava-v1.6-mistral-7b-hf",
        device=device,
        logger=logger,
        quantization=quantization_config
    )

    feedback_data = enhanced_feedback_collection(
        logger=logger,
        output_dir=output_dir,
        cbm_model=model,
        dataloader=val_loader,
        llava_annotator=llava_annotator,
        dataset_type=selected_dataset,
        device=device,
        concept_names=concept_names,
        model_path=best_model_path,
        min_confidence=min_confidence,
        use_uncertainty_sampling=False 
    )

    # ==============================
    # FINETUNING 
    # ==============================
    training_history = finetune_cbm_with_llava_feedback(
        output_dir=output_dir,
        logger=logger,
        cbm_model=model,
        feedback_data=feedback_data,
        original_train_loader=train_loader, 
        val_loader=val_loader,
        num_epochs=epochs_finetuning,
        learning_rate=lr_finetuning,
        weight_decay=weight_decay_finetuning,
        device=device,
        min_confidence=min_confidence,
        patience=5,
        selected_dataset=selected_dataset
    )

    # ==============================
    # POST-FINETUNING TEST EVALUATION
    # ==============================
    print("\nEvaluating on test set AFTER finetuning...")
    test_results = evaluate_finetuned_model(model, test_loader, output_dir, logger=logger)
    logger.info(f"Post-finetuning test results: {test_results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CBM Experiments')
    parser.add_argument('--dataset', type=str, choices=['shapes3d', 'cub', 'enhanced'], required=True,
                        help='Dataset to use for the experiment')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to the directory')
    parser.add_argument('--epochs_3d', type=int, default=15,
                       help='Number of training epochs for 3D')
    parser.add_argument('--epochs_cub', type=int, default=30,
                       help='Number of training epochs for CUB')
    parser.add_argument('--patience_3d', type=int, default=10,
                       help='Patience for early stopping with shapes')
    parser.add_argument('--patience_cub', type=int, default=15,
                       help='Patience for early stopping with cub')
    parser.add_argument('--batch_size_3d', type=int, default=16,
                       help='Batch size for 3D training')
    parser.add_argument('--batch_size_cub', type=int, default=32,
                       help='Batch size for CUB training')
    parser.add_argument('--train_samples_3d', type=int, default=150,
                       help='Samples to use for training set')
    parser.add_argument('--val_samples_3d', type=int, default=150,
                       help='Samples to use for validation set')
    parser.add_argument('--test_samples_3d', type=int, default=100,
                       help='Samples to use for testing set')
    parser.add_argument('--min_confidence', type=float, default=0.5,
                       help='Minimum confidence for feedback collection')
    args = parser.parse_args()
    main(args)
