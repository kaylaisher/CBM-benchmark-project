#!/usr/bin/env python3
"""
Multi-Method Concept Generation Framework
Implements three specific research-backed methods for concept generation:

1. Label-free CBM (ICLR 2023) - https://arxiv.org/pdf/2304.06129
2. Learning Concise and Descriptive Attributes (2023) - https://arxiv.org/pdf/2308.03685  
3. Language in a Bottle (LaBo) (2023) - https://arxiv.org/pdf/2211.11158

This script allows users to select which method to use for concept generation
and applies the appropriate prompting, filtering, and optimization strategies.
"""

import os
import json
import argparse
import requests
import re
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings("ignore")


@dataclass 
class MethodConfig:
    """Configuration for different concept generation methods."""
    method: str  # 'label_free_cbm', 'concise_descriptive', 'labo'
    llama_endpoint: str = "http://localhost:11434/api/generate"
    model_name: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 500
    
    # Filtering parameters (Label-free CBM & Concise Descriptive)
    length_threshold: int = 30
    class_similarity_threshold: float = 0.85
    concept_similarity_threshold: float = 0.9
    
    # LaBo specific parameters
    concepts_per_class: int = 50
    discriminability_weight: float = 1.0
    coverage_weight: float = 1.0
    
    # Concise Descriptive specific parameters
    target_concept_count: int = 32
    mahalanobis_lambda: float = 0.01
    
    # Output parameters
    verbose: bool = False
    output_dir: str = "/DiskHDD/s112504502/local-llm/output_files/unify_concept_output"


class BaseMethod:
    """Base class for concept generation methods."""
    
    def __init__(self, config: MethodConfig):
        self.config = config
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def query_llama(self, prompt: str) -> str:
        """Query local Llama instance."""
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
        }
        
        try:
            response = requests.post(self.config.llama_endpoint, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            if self.config.verbose:
                print(f"Error querying Llama: {e}")
            return ""
    
    def extract_concepts_from_response(self, response: str) -> List[str]:
        """Extract individual noun-phrase concepts from LLM response."""
        concepts = []
        seen = set()
        
        lines = response.split('\n')
        for line in lines:
            # Remove bullets/numbers and normalize whitespace
            cleaned = re.sub(r'^[\d\.\-\*\•\s]*', '', line.strip())
            
            # Split by comma if needed
            sub_concepts = [x.strip() for x in cleaned.split(',')] if ',' in cleaned else [cleaned]
            
            for concept in sub_concepts:
                concept = concept.strip()
                if not concept:
                    continue
                
                # Capitalize if it's a noun phrase or starts with "a"/"an"/"the"
                if not re.match(r'^(a|an|the)\s', concept, re.IGNORECASE):
                    concept = 'a ' + concept
                
                # Remove multiple spaces and basic punctuation
                concept = re.sub(r'[^\w\s\-]', '', concept)
                concept = re.sub(r'\s+', ' ', concept).strip()
                
                if len(concept) < 4 or concept.lower() in seen:
                    continue

                # Filter out filler/generic phrases
                filler_phrases = ['good question', 'fun question', 'that is', 'they are', 'it is', 'i think']
                if any(f in concept.lower() for f in filler_phrases):
                    continue
                
                seen.add(concept.lower())
                concepts.append(concept)
        
        return concepts


class LabelFreeCBM(BaseMethod):
    """
    Label-free CBM implementation (ICLR 2023)
    Paper: https://arxiv.org/pdf/2304.06129
    """
    
    def generate_concepts(self, class_names: List[str]) -> Tuple[Dict[str, List[str]], Dict]:
        """Generate concepts using Label-free CBM method."""
        if self.config.verbose:
            print("🚀 Using Label-free CBM method (ICLR 2023)")
        
        all_concepts = {}
        generation_details = {
            'method': 'Label-free CBM',
            'prompts_used': self._get_prompts(),
            'classes_processed': []
        }
        
        # Step 1: Generate initial concept set
        raw_concepts = []
        for class_name in class_names:
            class_concepts = []
            class_info = {'class_name': class_name, 'raw_concepts': []}
            
            for prompt_template in self._get_prompts():
                prompt = prompt_template.format(class_name=class_name)
                if self.config.verbose:
                    print(f"  Prompting for {class_name}: {prompt}")
                
                response = self.query_llama(prompt)
                concepts = self.extract_concepts_from_response(response)
                class_concepts.extend(concepts)
                class_info['raw_concepts'].extend(concepts)
            
            all_concepts[class_name] = class_concepts
            raw_concepts.extend(class_concepts)
            generation_details['classes_processed'].append(class_info)
        
        # Step 2: Apply Label-free CBM filtering to the overall pool
        all_raw_concepts = []
        for concepts_list in all_concepts.values():
            all_raw_concepts.extend(concepts_list)
        
        # Remove duplicates while preserving class association
        unique_concepts = list(set(all_raw_concepts))
        
        if self.config.verbose:
            print(f"  📊 Generated {len(all_raw_concepts)} raw concepts, {len(unique_concepts)} unique")
        
        # Step 3: Keep class-specific concepts (proper Label-free CBM approach)
        final_concepts = {}
        for class_name in class_names:
            # Filter the class's original concepts using the same filtering steps
            class_concepts = all_concepts[class_name]
            
            # Apply same filtering to class-specific concepts
            # Step 1: Length filter
            class_filtered = [c for c in class_concepts if len(c) <= self.config.length_threshold]
            
            # Step 2: Remove concepts too similar to class names
            if class_filtered:
                concept_embeddings = self.embedding_model.encode(class_filtered)
                class_embeddings = self.embedding_model.encode(class_names)
                similarities = cosine_similarity(concept_embeddings, class_embeddings)
                max_similarities = np.max(similarities, axis=1)
                
                class_filtered = [c for i, c in enumerate(class_filtered) 
                               if max_similarities[i] < self.config.class_similarity_threshold]
            
            # Step 3: Remove concepts too similar to each other within this class
            if len(class_filtered) > 1:
                concept_embeddings = self.embedding_model.encode(class_filtered)
                similarities = cosine_similarity(concept_embeddings)
                
                to_remove = set()
                for i in range(len(similarities)):
                    for j in range(i + 1, len(similarities)):
                        if similarities[i][j] > self.config.concept_similarity_threshold:
                            to_remove.add(j)
                
                class_filtered = [c for i, c in enumerate(class_filtered) if i not in to_remove]
            
            # Keep only meaningful concepts (remove very generic ones)
            class_filtered = [c for c in class_filtered 
                            if len(c.split()) <= 6 and len(c.split()) >= 2]
            
            final_concepts[class_name] = class_filtered[:20]  # Limit to top 20 per class
        
        # Calculate total final concepts
        total_final_concepts = sum(len(concepts) for concepts in final_concepts.values())
        generation_details['final_concept_count'] = total_final_concepts
        generation_details['concepts_per_class'] = {
            class_name: len(concepts) for class_name, concepts in final_concepts.items()
        }
        
        return final_concepts, generation_details
    
    def _get_prompts(self) -> List[str]:
        """Get the three prompts from Label-free CBM paper."""
        return [
            "List the most important features for recognizing something as a {class_name}:",
            "List the things most commonly seen around a {class_name}:",
            "Give superclasses for the word {class_name}:"
        ]
    
    def _apply_label_free_filtering(self, concepts: List[str], class_names: List[str]) -> List[str]:
        """Apply the 5-step filtering process from Label-free CBM."""
        if self.config.verbose:
            print(f"  📝 Starting Label-free CBM filtering with {len(concepts)} concepts")
        
        # Step 1: Length filter (≤30 characters)
        concepts = [c for c in concepts if len(c) <= self.config.length_threshold]
        if self.config.verbose:
            print(f"  ✂️  After length filter: {len(concepts)} concepts")
        
        # Step 2: Remove concepts too similar to classes
        if concepts:
            concept_embeddings = self.embedding_model.encode(concepts)
            class_embeddings = self.embedding_model.encode(class_names)
            similarities = cosine_similarity(concept_embeddings, class_embeddings)
            max_similarities = np.max(similarities, axis=1)
            
            concepts = [c for i, c in enumerate(concepts) 
                       if max_similarities[i] < self.config.class_similarity_threshold]
            
            if self.config.verbose:
                print(f"  🎯 After class similarity filter: {len(concepts)} concepts")
        
        # Step 3: Remove concepts too similar to each other
        if len(concepts) > 1:
            concept_embeddings = self.embedding_model.encode(concepts)
            similarities = cosine_similarity(concept_embeddings)
            
            to_remove = set()
            for i in range(len(similarities)):
                for j in range(i + 1, len(similarities)):
                    if similarities[i][j] > self.config.concept_similarity_threshold:
                        to_remove.add(j)
            
            concepts = [c for i, c in enumerate(concepts) if i not in to_remove]
            
            if self.config.verbose:
                print(f"  🔄 After concept similarity filter: {len(concepts)} concepts")
        
        # Steps 4 & 5: Remove concepts not present in training data & can't project accurately
        # (Simplified - in full implementation would use CLIP scores and CLIP-Dissect)
        filtered_concepts = [c for c in concepts if len(c.split()) <= 5]  # Keep shorter concepts
        
        if self.config.verbose:
            print(f"  ✅ Final concepts after all filtering: {len(filtered_concepts)} concepts")
        
        return filtered_concepts


class ConciseDescriptiveMethod(BaseMethod):
    """
    Learning Concise and Descriptive Attributes implementation (2023)
    Paper: https://arxiv.org/pdf/2308.03685
    """
    
    def generate_concepts(self, class_names: List[str]) -> Tuple[Dict[str, List[str]], Dict]:
        """Generate concepts using Concise & Descriptive method."""
        if self.config.verbose:
            print("🚀 Using Concise & Descriptive Attributes method (2023)")
        
        generation_details = {
            'method': 'Concise & Descriptive Attributes',
            'prompts_used': self._get_prompts(),
            'target_concepts': self.config.target_concept_count,
            'classes_processed': []
        }
        
        # Step 1: Generate large attribute pool
        all_raw_concepts = []
        class_concepts = {}
        
        for class_name in class_names:
            class_raw_concepts = []
            class_info = {'class_name': class_name, 'raw_concepts': []}
            
            # Use both instance and batch prompting
            for prompt_template in self._get_prompts():
                prompt = prompt_template.format(class_name=class_name)
                if self.config.verbose:
                    print(f"  Prompting for {class_name}: {prompt}")
                
                response = self.query_llama(prompt)
                concepts = self.extract_concepts_from_response(response)
                class_raw_concepts.extend(concepts)
                class_info['raw_concepts'].extend(concepts)
            
            class_concepts[class_name] = class_raw_concepts
            all_raw_concepts.extend(class_raw_concepts)
            generation_details['classes_processed'].append(class_info)
        
        # Step 2: Apply task-guided attribute searching
        selected_concepts = self._task_guided_search(all_raw_concepts, class_names)
        
        # Step 3: Distribute selected concepts
        final_concepts = {}
        concepts_per_class = max(1, len(selected_concepts) // len(class_names))
        
        for i, class_name in enumerate(class_names):
            start_idx = i * concepts_per_class
            end_idx = min(start_idx + concepts_per_class, len(selected_concepts))
            final_concepts[class_name] = selected_concepts[start_idx:end_idx]
        
        generation_details['final_concept_count'] = len(selected_concepts)
        return final_concepts, generation_details
    
    def _get_prompts(self) -> List[str]:
        """Get prompts for Concise & Descriptive method."""
        return [
            "What are useful visual features to distinguish {class_name} in a photo?",
            "List visual attributes (color, texture, shape) that describe {class_name}:",
            "List functional attributes (purpose, behavior, interaction) of {class_name}:",
            "List contextual attributes (location, time, association) related to {class_name}:"
        ]
    
    def _task_guided_search(self, concepts: List[str], class_names: List[str]) -> List[str]:
        """
        Simplified task-guided attribute searching.
        In full implementation, this would use learnable embeddings and Mahalanobis distance.
        """
        if self.config.verbose:
            print(f"  🔍 Task-guided search from {len(concepts)} concepts")
        
        # Remove duplicates
        unique_concepts = list(set(concepts))
        
        # Filter by length and similarity (simplified)
        filtered_concepts = [c for c in unique_concepts if len(c) <= self.config.length_threshold]
        
        # Select top K concepts (simplified selection)
        target_count = min(self.config.target_concept_count, len(filtered_concepts))
        
        if len(filtered_concepts) > target_count:
            # Use embedding-based diversity selection
            embeddings = self.embedding_model.encode(filtered_concepts)
            
            # Greedy selection for diversity
            selected_indices = [0]  # Start with first concept
            
            for _ in range(target_count - 1):
                remaining_indices = [i for i in range(len(filtered_concepts)) if i not in selected_indices]
                if not remaining_indices:
                    break
                
                # Find concept most different from already selected
                max_min_distance = -1
                best_idx = remaining_indices[0]
                
                for idx in remaining_indices:
                    min_distance = min([
                        1 - cosine_similarity([embeddings[idx]], [embeddings[sel_idx]])[0][0]
                        for sel_idx in selected_indices
                    ])
                    
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_idx = idx
                
                selected_indices.append(best_idx)
            
            selected_concepts = [filtered_concepts[i] for i in selected_indices]
        else:
            selected_concepts = filtered_concepts
        
        if self.config.verbose:
            print(f"  ✅ Selected {len(selected_concepts)} diverse concepts")
        
        return selected_concepts


class LaBo(BaseMethod):
    """
    Language in a Bottle (LaBo) implementation (2023)
    Paper: https://arxiv.org/pdf/2211.11158
    """
    
    def generate_concepts(self, class_names: List[str]) -> Tuple[Dict[str, List[str]], Dict]:
        """Generate concepts using LaBo method."""
        if self.config.verbose:
            print("🚀 Using Language in a Bottle (LaBo) method (2023)")
        
        generation_details = {
            'method': 'Language in a Bottle (LaBo)',
            'prompts_used': self._get_prompts(),
            'concepts_per_class': self.config.concepts_per_class,
            'classes_processed': []
        }
        
        all_concepts = {}
        
        # Step 1: Generate candidate concepts for each class
        for class_name in class_names:
            class_info = {'class_name': class_name, 'raw_concepts': [], 'filtered_concepts': []}
            
            # Generate 500 sentences per class (as in paper)
            raw_concepts = []
            for prompt_template in self._get_prompts():
                for _ in range(100):  # Generate multiple responses per prompt
                    prompt = prompt_template.format(class_name=class_name)
                    response = self.query_llama(prompt)
                    concepts = self.extract_concepts_from_response(response)
                    raw_concepts.extend(concepts)
                    
                    if len(raw_concepts) >= 500:  # Limit as in paper
                        break
                
                if len(raw_concepts) >= 500:
                    break
            
            class_info['raw_concepts'] = raw_concepts[:500]
            
            # Step 2: Apply submodular concept selection
            selected_concepts = self._submodular_selection(raw_concepts, class_name, class_names)
            class_info['filtered_concepts'] = selected_concepts
            
            all_concepts[class_name] = selected_concepts
            generation_details['classes_processed'].append(class_info)
            
            if self.config.verbose:
                print(f"  Generated {len(selected_concepts)} concepts for {class_name}")
        
        return all_concepts, generation_details
    
    def _get_prompts(self) -> List[str]:
        """Get the 5 prompts from LaBo paper."""
        return [
            "Describe what the {class_name} looks like:",
            "Describe the appearance of the {class_name}:",
            "Describe the color of the {class_name}:",
            "Describe the pattern of the {class_name}:",
            "Describe the shape of the {class_name}:"
        ]
    
    def _submodular_selection(self, concepts: List[str], target_class: str, all_classes: List[str]) -> List[str]:
        """
        Simplified submodular concept selection.
        Full implementation would use discriminability and coverage scores.
        """
        # Remove duplicates and filter
        unique_concepts = list(set(concepts))
        filtered_concepts = [c for c in unique_concepts 
                           if len(c) <= self.config.length_threshold 
                           and target_class.lower() not in c.lower()]
        
        # Select top K concepts using simplified scoring
        target_count = min(self.config.concepts_per_class, len(filtered_concepts))
        
        if len(filtered_concepts) <= target_count:
            return filtered_concepts
        
        # Simplified discriminability scoring
        concept_embeddings = self.embedding_model.encode(filtered_concepts)
        class_embeddings = self.embedding_model.encode(all_classes)
        
        scores = []
        for i, concept in enumerate(filtered_concepts):
            # Discriminability: how well it distinguishes target class
            concept_emb = concept_embeddings[i:i+1]
            similarities = cosine_similarity(concept_emb, class_embeddings)[0]
            
            # Find target class index
            target_idx = all_classes.index(target_class) if target_class in all_classes else 0
            target_similarity = similarities[target_idx]
            other_similarities = [sim for j, sim in enumerate(similarities) if j != target_idx]
            
            # Score = similarity to target - max similarity to others
            discriminability = target_similarity - (max(other_similarities) if other_similarities else 0)
            scores.append(discriminability)
        
        # Select top concepts by score
        scored_concepts = list(zip(filtered_concepts, scores))
        scored_concepts.sort(key=lambda x: x[1], reverse=True)
        
        return [concept for concept, score in scored_concepts[:target_count]]


class UnifiedConceptGenerator:
    """Main class that orchestrates different concept generation methods."""
    
    def __init__(self, config: MethodConfig):
        self.config = config
        
        # Initialize the appropriate method
        if config.method == 'label_free_cbm':
            self.method = LabelFreeCBM(config)
        elif config.method == 'concise_descriptive':
            self.method = ConciseDescriptiveMethod(config)
        elif config.method == 'labo':
            self.method = LaBo(config)
        else:
            raise ValueError(f"Unknown method: {config.method}")
    
    def generate_concepts(self, class_names: List[str]) -> Tuple[Dict[str, List[str]], Dict]:
        """Generate concepts using the selected method."""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"CONCEPT GENERATION")
            print(f"{'='*60}")
            print(f"Method: {self.config.method}")
            print(f"Classes: {', '.join(class_names)}")
            print(f"{'='*60}")
        
        start_time = datetime.now()
        concepts, details = self.method.generate_concepts(class_names)
        end_time = datetime.now()
        
        details['generation_time'] = str(end_time - start_time)
        details['total_classes'] = len(class_names)
        
        if self.config.verbose:
            self._print_summary(concepts, details)
        
        return concepts, details
    
    def _print_summary(self, concepts: Dict[str, List[str]], details: Dict):
        """Print generation summary."""
        print(f"\n{'='*60}")
        print(f"GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Method: {details['method']}")
        print(f"Total classes: {details['total_classes']}")
        print(f"Generation time: {details['generation_time']}")
        
        total_concepts = sum(len(class_concepts) for class_concepts in concepts.values())
        print(f"Total concepts generated: {total_concepts}")
        
        print(f"\nPer-class breakdown:")
        for class_name, class_concepts in concepts.items():
            print(f"  {class_name}: {len(class_concepts)} concepts")
            if self.config.verbose and class_concepts:
                print(f"    Examples: {', '.join(class_concepts[:3])}")
    
    def save_results(self, concepts: Dict[str, List[str]], details: Dict, 
                    class_names: List[str], output_file: Optional[str] = None):
        """Save generated concepts in Label-free CBM repository format."""
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create dataset name from class names (e.g., cifar10, cifar100)
        if len(class_names) == 10:
            dataset_name = "cifar10"
        elif len(class_names) == 100:
            dataset_name = "cifar100"  
        else:
            dataset_name = f"dataset_{len(class_names)}classes"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Create filtered.txt file (like cifar10_filtered.txt)
        txt_filename = f"{dataset_name}_filtered_{timestamp}.txt"
        txt_path = os.path.join(self.config.output_dir, txt_filename)
        
        # Collect all unique concepts across all classes
        all_concepts = []
        for class_concepts in concepts.values():
            all_concepts.extend(class_concepts)
        
        # Remove duplicates while preserving order
        unique_concepts = []
        seen = set()
        for concept in all_concepts:
            if concept not in seen:
                unique_concepts.append(concept)
                seen.add(concept)
        
        # Write to txt file (one concept per line)
        with open(txt_path, 'w', encoding='utf-8') as f:
            for concept in unique_concepts:
                f.write(f"{concept}\n")
        
        print(f"💾 Filtered concepts saved to: {txt_path}")
        print(f"   Total unique concepts: {len(unique_concepts)}")
        
        # 2. Create three separate JSON files for each prompt type
        prompt_types = ['prompt1', 'prompt2', 'prompt3']
        prompt_names = {
            'prompt1': 'most_important_features',
            'prompt2': 'commonly_seen_around', 
            'prompt3': 'superclasses'
        }
        
        # Split concepts roughly equally among the three prompt types
        concepts_per_prompt = len(unique_concepts) // 3
        
        for i, (prompt_key, prompt_name) in enumerate(prompt_names.items()):
            json_filename = f"{dataset_name}_{prompt_name}_{timestamp}.json"
            json_path = os.path.join(self.config.output_dir, json_filename)
            
            # Assign concepts to this prompt type
            start_idx = i * concepts_per_prompt
            if i == 2:  # Last prompt gets remaining concepts
                end_idx = len(unique_concepts)
            else:
                end_idx = (i + 1) * concepts_per_prompt
            
            prompt_concepts = unique_concepts[start_idx:end_idx]
            
            # Create class-concept mapping for this prompt
            prompt_data = {}
            concepts_per_class = max(1, len(prompt_concepts) // len(class_names))
            
            for j, class_name in enumerate(class_names):
                class_start = j * concepts_per_class
                class_end = min(class_start + concepts_per_class, len(prompt_concepts))
                
                # If this class would get no concepts, give it some from the pool
                if class_start >= len(prompt_concepts):
                    class_concepts = prompt_concepts[:min(5, len(prompt_concepts))]
                else:
                    class_concepts = prompt_concepts[class_start:class_end]
                    # Ensure each class gets at least a few concepts
                    if len(class_concepts) < 3 and len(prompt_concepts) >= 3:
                        class_concepts = prompt_concepts[:3]
                
                prompt_data[class_name] = class_concepts
            
            # Save JSON file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(prompt_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 {prompt_name} concepts saved to: {json_path}")
            print(f"   Concepts in this prompt: {len(prompt_concepts)}")
        
        # 3. Create comprehensive metadata file
        metadata_filename = f"{dataset_name}_metadata_{timestamp}.json"
        metadata_path = os.path.join(self.config.output_dir, metadata_filename)
        
        metadata = {
            'dataset_info': {
                'dataset_name': dataset_name,
                'total_classes': len(class_names),
                'class_names': class_names
            },
            'generation_info': {
                'method': self.config.method,
                'generation_timestamp': datetime.now().isoformat(),
                'total_unique_concepts': len(unique_concepts),
                'concepts_per_class_avg': len(unique_concepts) // len(class_names)
            },
            'config': {
                'model_name': self.config.model_name,
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens,
                'length_threshold': self.config.length_threshold,
                'class_similarity_threshold': self.config.class_similarity_threshold,
                'concept_similarity_threshold': self.config.concept_similarity_threshold
            },
            'files_generated': {
                'filtered_concepts': txt_filename,
                'most_important_features': f"{dataset_name}_most_important_features_{timestamp}.json",
                'commonly_seen_around': f"{dataset_name}_commonly_seen_around_{timestamp}.json",
                'superclasses': f"{dataset_name}_superclasses_{timestamp}.json"
            },
            'generation_details': details
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Metadata saved to: {metadata_path}")
        
        # 4. Create summary report
        summary_filename = f"{dataset_name}_summary_{timestamp}.txt"
        summary_path = os.path.join(self.config.output_dir, summary_filename)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Label-free CBM Concept Generation Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Method: {self.config.method}\n") 
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Classes: {len(class_names)}\n")
            f.write(f"Total Unique Concepts: {len(unique_concepts)}\n")
            f.write(f"Average Concepts per Class: {len(unique_concepts) // len(class_names)}\n")
            f.write(f"\nFiles Generated:\n")
            f.write(f"- Filtered concepts: {txt_filename}\n")
            f.write(f"- Most important features: {dataset_name}_most_important_features_{timestamp}.json\n")
            f.write(f"- Commonly seen around: {dataset_name}_commonly_seen_around_{timestamp}.json\n")
            f.write(f"- Superclasses: {dataset_name}_superclasses_{timestamp}.json\n")
            f.write(f"- Metadata: {metadata_filename}\n")
            f.write(f"\nClass Names:\n")
            for i, class_name in enumerate(class_names, 1):
                f.write(f"{i:2d}. {class_name}\n")
            
            f.write(f"\nSample Concepts:\n")
            for i, concept in enumerate(unique_concepts[:20], 1):
                f.write(f"{i:2d}. {concept}\n")
            if len(unique_concepts) > 20:
                f.write(f"... and {len(unique_concepts) - 20} more concepts\n")
        
        print(f"💾 Summary report saved to: {summary_path}")
        print(f"\n🎉 Generated {len(unique_concepts)} concepts in Label-free CBM format!")
        print(f"📁 All files saved to: {self.config.output_dir}")
        
        return {
            'txt_file': txt_path,
            'json_files': [
                os.path.join(self.config.output_dir, f"{dataset_name}_most_important_features_{timestamp}.json"),
                os.path.join(self.config.output_dir, f"{dataset_name}_commonly_seen_around_{timestamp}.json"), 
                os.path.join(self.config.output_dir, f"{dataset_name}_superclasses_{timestamp}.json")
            ],
            'metadata_file': metadata_path,
            'summary_file': summary_path
        }

    @staticmethod
    def load_class_names(file_path: str) -> List[str]:
        """Load class names from a text file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Class names file not found: {file_path}")
        
        class_names = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    class_names.append(line)
        
        return class_names


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Multi-Method Concept Generation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Methods:
  label_free_cbm    - Label-free CBM (ICLR 2023) - Automated CBM without manual concepts
  concise_descriptive - Concise & Descriptive Attributes (2023) - Learning compact attribute sets  
  labo             - Language in a Bottle (2023) - LLM-guided concept bottlenecks

Examples:
  # Use Label-free CBM method
  python script.py --classes classes.txt --method label_free_cbm --verbose
  
  # Use Concise & Descriptive method with 16 target concepts
  python script.py --classes classes.txt --method concise_descriptive --target-concepts 16
  
  # Use LaBo method with 30 concepts per class
  python script.py --classes classes.txt --method labo --concepts-per-class 30
        """
    )
    
    # Required arguments
    parser.add_argument("--classes", "-c", required=True,
                       help="Path to file containing class names (one per line)")
    parser.add_argument("--method", "-m", required=True,
                       choices=['label_free_cbm', 'concise_descriptive', 'labo'],
                       help="Concept generation method to use")
    
    # Optional arguments
    parser.add_argument("--output", "-o", 
                       help="Output file name (default: auto-generated)")
    parser.add_argument("--output-dir", default="/DiskHDD/s112504502/local-llm/output_files/unify_concept_output",
                       help="Output directory (default: /DiskHDD/s112504502/local-llm/output_files/unify_concept_output)")
    parser.add_argument("--llama-endpoint", default="http://localhost:11434/api/generate",
                       help="Llama API endpoint")
    parser.add_argument("--model-name", default="llama3",
                       help="Llama model name")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for LLM generation")
    parser.add_argument("--max-tokens", type=int, default=500,
                       help="Maximum tokens for LLM generation")
    
    # Method-specific parameters
    parser.add_argument("--length-threshold", type=int, default=30,
                       help="Maximum concept length (Label-free CBM & Concise)")
    parser.add_argument("--class-similarity-threshold", type=float, default=0.85,
                       help="Class similarity threshold for filtering")
    parser.add_argument("--concept-similarity-threshold", type=float, default=0.9,
                       help="Concept similarity threshold for deduplication")
    parser.add_argument("--target-concepts", type=int, default=32,
                       help="Target number of concepts (Concise & Descriptive)")
    parser.add_argument("--concepts-per-class", type=int, default=50,
                       help="Concepts per class (LaBo)")
    
    # General options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Load class names
    try:
        class_names = UnifiedConceptGenerator.load_class_names(args.classes)
        print(f"✅ Loaded {len(class_names)} class names from {args.classes}")
        if args.verbose:
            print(f"Classes: {', '.join(class_names)}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Create configuration
    config = MethodConfig(
        method=args.method,
        llama_endpoint=args.llama_endpoint,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        length_threshold=args.length_threshold,
        class_similarity_threshold=args.class_similarity_threshold,
        concept_similarity_threshold=args.concept_similarity_threshold,
        target_concept_count=args.target_concepts,
        concepts_per_class=args.concepts_per_class,
        verbose=args.verbose,
        output_dir=args.output_dir
    )
    
    # Initialize generator
    try:
        generator = UnifiedConceptGenerator(config)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Generate concepts
    try:
        concepts, details = generator.generate_concepts(class_names)
        
        # 🔽 Write entire flattened concept list
        flat_concepts = set()
        for concept_list in concepts.values():
            flat_concepts.update(concept_list)

        flat_output_path = os.path.join(config.output_dir, "flattened_concepts.txt")
        with open(flat_output_path, "w", encoding="utf-8") as f:
            for concept in sorted(flat_concepts):
                f.write(concept + "\n")

        print(f"\n📝 Flattened concept list saved to: {flat_output_path}")
        print(f"   Total unique concepts: {len(flat_concepts)}")


        # Save results
        output_path = generator.save_results(concepts, details, class_names, args.output)
        
        print(f"\n🎉 Concept generation completed successfully!")
        print(f"📊 Generated concepts for {len(class_names)} classes using {args.method}")
        print(f"📁 Results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during concept generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
