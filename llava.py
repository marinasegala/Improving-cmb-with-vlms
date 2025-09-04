from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch 
import re
from typing import List, Dict, Tuple
import numpy as np
from torch.utils.data import Dataset

# ================================
# LLAVA FEEDBACK DATASET
# ================================

class LLaVAFeedbackDataset(Dataset):
    def __init__(self, feedback_data: List[Dict], transform=None, logger=None, min_confidence=0.6):
        """
        Dataset containing LLaVA feedback for fine-tuning CBM
        
        Args:
            feedback_data (List[Dict]): List of feedback annotations
            transform: Image transformations
            min_confidence (float): Minimum confidence threshold for corrections
        """
        self.feedback_data = feedback_data
        self.transform = transform
        self.logger = logger
        self.min_confidence = min_confidence

        # Filter for samples that need correction with high confidence
        self.corrected_samples = self.validate_corrections(
            feedback_data, 
            min_confidence=min_confidence,
            strategy_agreement_threshold=2,  # At least 2 strategies must agree
            logger=logger
        )

    def __len__(self):
        return len(self.corrected_samples)
    
    def __getitem__(self, idx):
        sample = self.corrected_samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'original_concepts': torch.FloatTensor(sample['cbm_concepts']),
            'true_concepts': torch.FloatTensor(sample['true_concepts']),   
            'corrected_label': torch.LongTensor([sample['llava_final_prediction']]).squeeze(),
            'original_label': torch.LongTensor([sample['cbm_prediction']]).squeeze(),
            'true_label': torch.LongTensor([sample['true_label']]).squeeze(),   
            'confidence': torch.FloatTensor([sample['llava_confidence']]),
            'concept_accuracy': torch.FloatTensor([sample.get('concept_accuracy', 0.0)]),
            'sample_id': sample['sample_id'],
             
            'llava_responses': {
                'direct': sample['llava_strategies']['direct']['response'],
                'guided': sample['llava_strategies']['guided']['response'],
                'complete': sample['llava_strategies']['complete']['response'],
            }
        }
    
    def validate_corrections(self, feedback_data: List[Dict], min_confidence: float = 0.7, 
                        strategy_agreement_threshold: int = 2, 
                        logger=None) -> List[Dict]:
        """
        Validate LLaVA corrections before applying them in fine-tuning
        
        Args:
            feedback_data: List of feedback samples from LLaVA
            min_confidence: Minimum confidence threshold for accepting corrections
            strategy_agreement_threshold: Minimum number of strategies that must agree
            logger: Logger instance
        
        Returns:
            List of validated corrections
        """
        validated_corrections = []
        
        total_corrections = sum(1 for sample in feedback_data if sample['needs_correction'])
        
        for sample in feedback_data:
            if not sample['needs_correction']:
                continue
                 
            if sample['llava_confidence'] < min_confidence:
                continue
                 
            strategies = sample['llava_strategies']
            cbm_pred = sample['cbm_prediction']
            llava_final = sample['llava_final_prediction']
            
            agreeing_strategies = 0
             
            if 'direct' in strategies and strategies['direct']['prediction'] == llava_final:
                agreeing_strategies += 1
                
            if 'complete' in strategies and strategies['complete']['prediction'] == llava_final:
                agreeing_strategies += 1

            if 'guided' in strategies and strategies['guided']['prediction'] == llava_final:
                agreeing_strategies += 1
                 
            if agreeing_strategies < strategy_agreement_threshold:
                continue
                
            # correction_direction = f"{cbm_pred} -> {llava_final}"
            
            validated_corrections.append(sample)
        
        if logger:
            logger.info(f"Correction validation results:")
            logger.info(f"  Total corrections suggested: {total_corrections}")
            logger.info(f"  Validated corrections: {len(validated_corrections)}")
            logger.info(f"  Rejection rate: {100*(1-len(validated_corrections)/max(1,total_corrections)):.1f}%")
            
            correction_patterns = {}
            for sample in validated_corrections:
                pattern = f"{sample['cbm_prediction']} -> {sample['llava_final_prediction']}"
                correction_patterns[pattern] = correction_patterns.get(pattern, 0) + 1
            
            logger.info(f"  Validated correction patterns: {correction_patterns}")
        
        return validated_corrections

    def get_correction_summary(self) -> Dict:
        """
        Get summary statistics about the corrections in this dataset
        
        Returns:
            Dict with correction statistics
        """
        if not self.corrected_samples:
            return {'total_corrections': 0}
        
        correction_patterns = {}
        confidence_scores = []
        concept_accuracies = []
        
        for sample in self.corrected_samples:
            original = sample['cbm_prediction']
            corrected = sample['llava_final_prediction']
            pattern = f"{original} -> {corrected}"
            correction_patterns[pattern] = correction_patterns.get(pattern, 0) + 1
            
            confidence_scores.append(sample['llava_confidence'])
            concept_accuracies.append(sample.get('concept_accuracy', 0.0))
        
        return {
            'total_corrections': len(self.corrected_samples),
            'correction_patterns': correction_patterns,
            'avg_confidence': np.mean(confidence_scores),
            'confidence_std': np.std(confidence_scores),
            'avg_concept_accuracy': np.mean(concept_accuracies),
            'confidence_distribution': {
                'min': np.min(confidence_scores),
                'max': np.max(confidence_scores),
                'median': np.median(confidence_scores)
            }
        }

    def filter_by_strategy_agreement(self, require_all_strategies_agree=True) -> 'LLaVAFeedbackDataset':
        """
        Create a filtered dataset where all LLaVA strategies agree on the correction
        
        Args:
            require_all_strategies_agree (bool): If True, all strategies must agree
            
        Returns:
            New LLaVAFeedbackDataset with filtered samples
        """
        filtered_samples = []
        
        for sample in self.corrected_samples:
            strategies = sample['llava_strategies']
             
            direct_pred = strategies['direct']['prediction']
            confidence_agrees = strategies['confidence']['agrees_with_cbm']
            
            if require_all_strategies_agree:
                if (direct_pred == sample['llava_final_prediction'] and not confidence_agrees): 
                    filtered_samples.append(sample)
            else:
                filtered_samples.append(sample)
        
        new_dataset = LLaVAFeedbackDataset([], self.transform, self.logger, self.min_confidence)
        new_dataset.corrected_samples = filtered_samples
        new_dataset.feedback_data = filtered_samples
        
        if self.logger:
            self.logger.info(f"Strategy agreement filtering: {len(filtered_samples)} samples remain")
        
        return new_dataset

# ================================
# ENHACED LLAVA
# ================================

class EnhancedLLaVAAnnotator:
    def __init__(self, model_id="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda", logger=None, quantization = None):
        """Enhanced LLaVA annotator with multiple evaluation strategies"""
        self.device = device
        self.model_id = model_id
        self.logger = logger
        self.quantization_config = quantization
        
        if logger:
            self.logger.info(f"Loading LLaVA model: {model_id}")
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        if quantization is None:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=device
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                quantization_config=quantization,
                device_map=device
            )
        
        if logger:
            self.logger.info("Enhanced LLaVA model loaded successfully!")

    def multi_prompt_evaluation(self, image: Image.Image, prediction: int, 
                               concept_probs: np.ndarray, concept_names: List[str],
                               dataset_type: str, ground_label: int) -> Dict:
        """
        Use multiple prompting strategies to get comprehensive feedback
        """
        results = {}
        
        # Strategy 1: Direct classification
        results['direct'] = self._direct_classification_prompt(image, dataset_type)

        # Strategy 2: Guided completion
        results['guided'] = self._guided_classification_prompt(image, dataset_type)

        results['complete'] = self._complete_classification_prompt(image, dataset_type)

        # Aggregate results
        final_prediction, confidence = self._aggregate_results(results, prediction)
        
        return {
            'strategies': results,
            'final_prediction': final_prediction,
            'confidence': confidence,
            'needs_correction': final_prediction == ground_label and final_prediction != prediction,
            'cbm_prediction': prediction
        }

    def _direct_classification_prompt(self, image: Image.Image, dataset_type: str) -> Dict:
        
        if dataset_type.lower() == "shapes3d":
            prompt = """[INST] <image>\nWhat is the main object? Focus on:
1. The shape of the object (cube or cylinder or sphere or pill/capsule)
2. The color of the object
Describe the object (color and shape) in one clear sentence.
[/INST]"""
        else: 
            prompt = """[INST]<image>\n What is the principal color of the bird? Focus on:
1. The predominant colors you see
2. Whether the bird appears mostly dark/black or has other prominent colors
Describe the bird's coloration in one clear sentence.
[/INST]"""
        
        response = self._generate_response(image, prompt, max_tokens=80)
        response = response.lower()
        
        if dataset_type.lower() == "shapes3d":
            is_red_pill = any(term in response.replace(',', '') for term in ['red pill', 'red capsule'])
            confidence = 0.8 if any(term in response for term in ['clearly', 'obviously', 'definitely']) else 0.7
            predictions = is_red_pill
        else:  
            if 'predominantly' in response:
                match = re.search(r'predominantly (\S+)', response)
                is_black_bird = any(term in match.group(1) for term in ['black', 'dark']) if match else False
                confidence = 0.8 if any(term in response for term in ['clearly', 'obviously', 'predominantly']) else 0.6
            else:
                is_black_bird = any(term in response for term in ['black', 'dark'])
                words = response.split()
                for i, word in enumerate(words):
                    if word in ['black', 'dark', 'darker']:
                        if i > 0:
                            confidence = 0.8 if words[i-1] in ['clearly', 'obviously'] else 0.5
                        if i < len(words) - 1:
                            confidence = 0.8 if words[i+1] in ['clearly', 'obviously'] else 0.5
            predictions = is_black_bird

        return {
            'response': response,
            'prediction': 1 if predictions else 0,
            'confidence': confidence
        }

    def _guided_classification_prompt(self, image: Image.Image, dataset_type: str) -> Dict:
        
        if dataset_type.lower() == "shapes3d":
            prompt = """[INST] <image>
The main 3d shapes object is {color} {shape}.
For the shapes you can choose from:
 - Cube: a solid object with six equal square faces and sharp edges, like a box.
 - Cylinder: a 3D shape with two flat circular ends connected by a curved side, like a can.
 - Sphere: a perfectly round 3D ball, same distance from the center in every direction.
 - Pill/Capsule: a rounded capsule shape, like a stretched sphere with flat circular ends.
Complete the description. Be concise.
[/INST]"""
        else: 
            prompt = """[INST] <image>
The main color of the bird in the image is {color}.
Complete the description. Be concise.
[/INST]"""

        response = self._generate_response(image, prompt, max_tokens=80)
        response = response.lower()
        confidence = 0.3
        if dataset_type.lower() == "shapes3d":
            red_color = any(term in response for term in ['red']) 
            pill_shape = any(term in response for term in ['pill', 'capsule'])
            predictions = red_color and pill_shape
            confidence = 0.8 if any(term in response for term in ['clearly', 'obviously', 'definitely']) else 0.7
        else: 
            if 'predominantly' in response:
                match = re.search(r'predominantly (\S+)', response)
                is_black_bird = any(term in match.group(1) for term in ['black', 'dark']) if match else False
                confidence = 0.8 if any(term in response for term in ['clearly', 'obviously', 'predominantly']) else 0.6
            else:
                is_black_bird = any(term in response for term in ['black', 'dark'])
                words = response.split()
                for i, word in enumerate(words):
                    if word in ['black', 'dark', 'darker']:
                        if i > 0:
                            confidence = 0.8 if words[i-1] in ['clearly', 'obviously'] else 0.5
                        if i < len(words) - 1:
                            confidence = 0.8 if words[i+1] in ['clearly', 'obviously'] else 0.5
            predictions = is_black_bird
        return {
            'response': response.replace("{", "").replace("}", ""),
            'prediction': 1 if predictions else 0,
            'confidence': confidence
        }

    def _complete_classification_prompt(self, image: Image.Image, dataset_type: str) -> Dict:
        
        if dataset_type.lower() == "shapes3d":
            prompt = """[INST] <image>
The main 3d object in the image is ...
Be concise.
[/INST]"""
        else: 
            prompt = """[INST] <image>
The main color of the bird in the image is ...
Be concise.
[/INST]"""

        response = self._generate_response(image, prompt, max_tokens=80)
        response = response.lower()

        if dataset_type.lower() == "shapes3d":
            red_color = any(term in response for term in ['red']) 
            pill_shape = any(term in response for term in ['pill', 'capsule'])
            predictions = red_color and pill_shape
            confidence = 0.8 if any(term in response for term in ['clearly', 'obviously', 'predominantly']) else 0.6
            
        else: 
            is_black_bird = any(term in response for term in ['black', 'dark'])
            confidence = 0.8 if any(term in response for term in ['clearly', 'obviously', 'predominantly']) else 0.6
            predictions = is_black_bird
        return {
            'response': response,
            'prediction': 1 if predictions else 0,
            'confidence': confidence
        }


    def _generate_response(self, image: Image.Image, prompt: str, max_tokens: int = 100) -> str:
        try:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generate_ids = self.model.generate(**inputs, max_new_tokens=150)
            output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            response = output[0].split("[/INST]")[-1].strip()
            return response
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating LLaVA response: {e}")
            return f"Error: {str(e)}"

    def _extract_rating(self, response: str) -> float:
        """Extract numerical rating from response"""
        patterns = [
            r'(\d+)/10',
            r'rating:?\s*(\d+)',
            r'score:?\s*(\d+)',
            r'(\d+)\s*out\s*of\s*10',
            r'as\s*a\s*(\d+)',
            r'^(\d+)', 
            r'(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                rating = float(match.group(1))
                return min(10, max(1, rating)) 
        if any(word in response.lower() for word in ['wrong', 'incorrect', 'bad', 'poor']):
            return 3.0
        elif any(word in response.lower() for word in ['correct', 'good', 'accurate', 'right']):
            return 8.0
        else:
            return 5.0  # Neutral

   
    def _aggregate_results(self, results: Dict, original_prediction: int) -> Tuple[int, float]:
        votes = []
        confidences = []
        strategies = {
            'direct': 1.0,
            'guided': 0.6,
            'complete': 0.6
        }
        
        for strategy, weight in strategies.items():
            if strategy in results:
                vote = results[strategy]['prediction']
                conf = results[strategy]['confidence']
                
                for _ in range(int(weight * 10)):  # Multiply by 10 for integer voting
                    votes.append(vote)
                    confidences.append(conf)
        
        if not votes:
            return original_prediction, 0.3
        
        # Majority vote
        final_prediction = 1 if np.mean(votes) > 0.5 else 0
        avg_confidence = np.mean(confidences)
        
        # Boost confidence if unanimous
        if len(set(votes)) == 1:
            avg_confidence = min(0.95, avg_confidence + 0.2)

        # Boost confidence based on agreement ratio among strategies
        agreement_count = sum(1 for strategy in strategies if strategy in results and results[strategy]['prediction'] == final_prediction)
        agreement_ratio = agreement_count / len(strategies)
        
        if agreement_ratio > 0.5:
            avg_confidence += 0.1 * (agreement_ratio - 0.5) / 0.5  # 0 to +0.1
            avg_confidence = min(1.0, max(0.0, avg_confidence))

        return final_prediction, avg_confidence

    def uncertainty_based_sampling(self, collected_data: Dict, 
                                  uncertainty_threshold: float = 0.3,
                                  max_samples: int = 500) -> List[str]:
        """
        Sample images where CBM is most uncertain for LLaVA evaluation
        """
        uncertain_samples = []
        
        for sample_id, sample in collected_data.items():
            concept_probs = np.array(sample['concept_probs'])
            label_probs = np.array(sample['label_probs'])
            
            # Calculate uncertainty metrics
            concept_uncertainty = np.mean(np.abs(concept_probs - 0.5))  # Distance from 0.5
            label_uncertainty = 1 - np.max(label_probs)  # 1 - max probability
            
            combined_uncertainty = (concept_uncertainty + label_uncertainty) / 2
            
            if combined_uncertainty > uncertainty_threshold:
                uncertain_samples.append({
                    'sample_id': sample_id,
                    'uncertainty': combined_uncertainty,
                    'concept_uncertainty': concept_uncertainty,
                    'label_uncertainty': label_uncertainty
                })
        
        # Sort by uncertainty (highest first)
        uncertain_samples.sort(key=lambda x: x['uncertainty'], reverse=True)

        max_samples = min(max_samples, len(uncertain_samples))
        
        return uncertain_samples[:max_samples]
