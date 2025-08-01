#!/usr/bin/env python3
"""
Multi-Method Concept Generation Framework
Implements three specific research-backed methods for concept generation:

1. Label-free CBM 
2. Learning Concise and Descriptive Attributes 
3. Language in a Bottle (LaBo) 

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


# ---------- Configurations ----------
PROMPT_TEMPLATES = {
    "label_free_cbm": {
        "around": "List the things most commonly seen around a {class_name}:",
        "important": "List the most important features for recognizing something as a {class_name}:",
        "superclass": "Give superclasses for the word {class_name}:"
    },
    "labo": {
        "main": "Describe what the {class_name} looks like:",
        "appearance": "Describe the appearance of the {class_name}:",
        "color": "Describe the color of the {class_name}:",
        "pattern": "Describe the pattern of the {class_name}:",
        "shape": "Describe the shape of the {class_name}:"
    },
    "LM4CV": {
        "visual": "What are useful visual features to distinguish {class_name} in a photo?",
        "attributes": "List visual attributes (color, texture, shape) that describe {class_name}:",
        "functional": "List functional attributes (purpose, behavior, interaction) of {class_name}:",
        "contextual": "List contextual attributes (location, time, association) related to {class_name}:"
    }
}


@dataclass 
class MethodConfig:
    """Configuration for different concept generation methods."""
    method: str  # 'label_free_cbm', 'LM4CV', 'labo'
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
        self.prompts = PROMPT_TEMPLATES[config.method]
        
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

    def generate_all_concepts(self, class_names: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """Generate concepts using all prompts for the method."""
        results = {}
        for variant, template in self.prompts.items():
            variant_concepts = {}
            for cls in class_names:
                prompt = template.format(class_name=cls)
                if self.config.verbose:
                    print(f"  Prompting for {cls} ({variant}): {prompt}")
                
                response = self.query_llama(prompt)
                concepts = self.extract_concepts_from_response(response)
                variant_concepts[cls] = concepts
            results[variant] = variant_concepts
        return results


class LabelFreeCBM(BaseMethod):
    """
    Label-free CBM implementation (ICLR 2023)
    Paper: https://arxiv.org/pdf/2304.06129
    """
    
    def generate_concepts(self, class_names: List[str]) -> Tuple[Dict[str, Dict[str, List[str]]], Dict]:
        """Generate concepts using Label-free CBM method."""
        if self.config.verbose:
            print(" Using Label-free CBM method (ICLR 2023)")
        
        generation_details = {
            'method': 'Label-free CBM',
            'prompts_used': list(self.prompts.keys()),
            'classes_processed': class_names
        }
        
        # Generate concepts for all prompt variants
        raw_concepts = self.generate_all_concepts(class_names)
        
        # Apply Label-free CBM filtering to each variant
        filtered_concepts = {}
        for variant, variant_concepts in raw_concepts.items():
            filtered_variant = {}
            for class_name, concepts in variant_concepts.items():
                filtered_variant[class_name] = self._apply_label_free_filtering(
                    concepts, class_names
                )[:20]  # Limit to top 20 per class
            filtered_concepts[variant] = filtered_variant
        
        return filtered_concepts, generation_details
    
    def _apply_label_free_filtering(self, concepts: List[str], class_names: List[str]) -> List[str]:
        """Apply the 5-step filtering process from Label-free CBM."""
        if self.config.verbose:
            print(f"     Starting Label-free CBM filtering with {len(concepts)} concepts")
        
        # Step 1: Length filter (≤30 characters)
        concepts = [c for c in concepts if len(c) <= self.config.length_threshold]
        if self.config.verbose:
            print(f"      After length filter: {len(concepts)} concepts")
        
        # Step 2: Remove concepts too similar to classes
        if concepts:
            concept_embeddings = self.embedding_model.encode(concepts)
            class_embeddings = self.embedding_model.encode(class_names)
            similarities = cosine_similarity(concept_embeddings, class_embeddings)
            max_similarities = np.max(similarities, axis=1)
            
            concepts = [c for i, c in enumerate(concepts) 
                       if max_similarities[i] < self.config.class_similarity_threshold]
            
            if self.config.verbose:
                print(f"     After class similarity filter: {len(concepts)} concepts")
        
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
                print(f"     After concept similarity filter: {len(concepts)} concepts")
        
        # Steps 4 & 5: Remove concepts not present in training data & can't project accurately
        # (Simplified - in full implementation would use CLIP scores and CLIP-Dissect)
        filtered_concepts = [c for c in concepts if len(c.split()) <= 5]  # Keep shorter concepts
        
        if self.config.verbose:
            print(f"     Final concepts after all filtering: {len(filtered_concepts)} concepts")
        
        return filtered_concepts


class ConciseDescriptiveMethod(BaseMethod):
    """
    Learning Concise and Descriptive Attributes implementation (2023)
    Paper: https://arxiv.org/pdf/2308.03685
    """
    
    def generate_concepts(self, class_names: List[str]) -> Tuple[Dict[str, Dict[str, List[str]]], Dict]:
        """Generate concepts using Concise & Descriptive method."""
        if self.config.verbose:
            print(" Using Concise & Descriptive Attributes method (2023)")
        
        generation_details = {
            'method': 'Concise & Descriptive Attributes',
            'prompts_used': list(self.prompts.keys()),
            'target_concepts': self.config.target_concept_count,
            'classes_processed': class_names
        }
        
        # Generate concepts for all prompt variants
        raw_concepts = self.generate_all_concepts(class_names)
        
        # Apply task-guided attribute searching
        filtered_concepts = {}
        for variant, variant_concepts in raw_concepts.items():
            # Collect all concepts from this variant
            all_variant_concepts = []
            for concepts in variant_concepts.values():
                all_variant_concepts.extend(concepts)
            
            # Apply task-guided selection
            selected_concepts = self._task_guided_search(all_variant_concepts, class_names)
            
            # Distribute back to classes
            filtered_variant = {}
            concepts_per_class = max(1, len(selected_concepts) // len(class_names))
            for i, class_name in enumerate(class_names):
                start_idx = i * concepts_per_class
                end_idx = min(start_idx + concepts_per_class, len(selected_concepts))
                filtered_variant[class_name] = selected_concepts[start_idx:end_idx]
            
            filtered_concepts[variant] = filtered_variant
        
        return filtered_concepts, generation_details
    
    def _task_guided_search(self, concepts: List[str], class_names: List[str]) -> List[str]:
        """
        Simplified task-guided attribute searching.
        In full implementation, this would use learnable embeddings and Mahalanobis distance.
        """
        if self.config.verbose:
            print(f"     Task-guided search from {len(concepts)} concepts")
        
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
            print(f"     Selected {len(selected_concepts)} diverse concepts")
        
        return selected_concepts


class LaBo(BaseMethod):
    """
    Language in a Bottle (LaBo) implementation (2023)
    Paper: https://arxiv.org/pdf/2211.11158
    """
    
    def generate_concepts(self, class_names: List[str]) -> Tuple[Dict[str, Dict[str, List[str]]], Dict]:
        """Generate concepts using LaBo method."""
        if self.config.verbose:
            print(" Using Language in a Bottle (LaBo) method (2023)")
        
        generation_details = {
            'method': 'Language in a Bottle (LaBo)',
            'prompts_used': list(self.prompts.keys()),
            'concepts_per_class': self.config.concepts_per_class,
            'classes_processed': class_names
        }
        
        # For LaBo, we primarily use the main prompt but generate multiple responses
        filtered_concepts = {"main": {}}
        
        for class_name in class_names:
            # Generate candidate concepts for each class
            raw_concepts = []
            main_prompt = self.prompts["main"].format(class_name=class_name)
            
            # Generate multiple responses per class (as in paper)
            for _ in range(10):  # Generate 10 responses per class
                response = self.query_llama(main_prompt)
                concepts = self.extract_concepts_from_response(response)
                raw_concepts.extend(concepts)
                
                if len(raw_concepts) >= 100:  # Limit to prevent too many concepts
                    break
            
            # Apply submodular concept selection
            selected_concepts = self._submodular_selection(raw_concepts, class_name, class_names)
            filtered_concepts["main"][class_name] = selected_concepts
            
            if self.config.verbose:
                print(f"    Generated {len(selected_concepts)} concepts for {class_name}")
        
        return filtered_concepts, generation_details
    
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


# ---------- Output Writers ----------
class UnifiedConceptWriter:
    """Unified writer for all concept generation methods."""
    
    def __init__(self, config: MethodConfig):
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)

    def save_label_free_outputs(self, concepts_dict: Dict[str, Dict[str, List[str]]], dataset_name: str):
        """Save Label-free CBM outputs in the expected format."""
        for variant, concepts in concepts_dict.items():
            path = os.path.join(self.config.output_dir, f"gpt3_{dataset_name}_{variant}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(concepts, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved: {path}")

        # Save flat .txt from 'important' variant
        important_concepts = concepts_dict.get("important", {})
        txt_path = os.path.join(self.config.output_dir, f"{dataset_name}_filtered.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for items in important_concepts.values():
                for concept in items:
                    f.write(concept.strip() + "\n")
        print(f" Saved: {txt_path}")

    def save_labo_outputs(self, concepts_dict: Dict[str, Dict[str, List[str]]], dataset_name: str):
        """Save LaBo outputs in the expected format."""
        output_dir = os.path.join(self.config.output_dir, "asso_opt", dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        json_path = os.path.join(output_dir, "selected_concepts.json")
        log_path = os.path.join(output_dir, "log.txt")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(concepts_dict["main"], f, indent=2, ensure_ascii=False)
        
        with open(log_path, "w", encoding="utf-8") as f:
            for class_name, items in concepts_dict["main"].items():
                f.write(f"[{class_name}]\n")
                for item in items[:10]:  # Show top 10 in log
                    f.write(f"  - {item}\n")
                f.write("\n")
        
        print(f" LaBo output: {json_path}, {log_path}")

    def save_lm4cv_outputs(self, concepts_dict: Dict[str, Dict[str, List[str]]], dataset_name: str):
        """Save LM4CV outputs in the expected format."""
        # Combine all variants into a single output for LM4CV
        combined_concepts = {}
        
        # Get the first variant as the primary output, or combine all
        first_variant = list(concepts_dict.values())[0]
        for class_name in first_variant.keys():
            class_concepts = []
            for variant_concepts in concepts_dict.values():
                class_concepts.extend(variant_concepts.get(class_name, []))
            # Remove duplicates while preserving order
            seen = set()
            unique_concepts = []
            for concept in class_concepts:
                if concept not in seen:
                    unique_concepts.append(concept)
                    seen.add(concept)
            combined_concepts[class_name] = unique_concepts
        
        json_path = os.path.join(self.config.output_dir, f"{dataset_name}_attributes.json")
        txt_path = os.path.join(self.config.output_dir, f"{dataset_name}_attributes.txt")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(combined_concepts, f, indent=2, ensure_ascii=False)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            for v in combined_concepts.values():
                for c in v:
                    f.write(c.strip() + "\n")
        
        print(f" LM4CV output: {json_path}, {txt_path}")


class UnifiedConceptGenerator:
    """Main class that orchestrates different concept generation methods."""
    
    def __init__(self, config: MethodConfig):
        self.config = config
        
        # Initialize the appropriate method
        if config.method == 'label_free_cbm':
            self.method = LabelFreeCBM(config)
        elif config.method == 'LM4CV':
            self.method = ConciseDescriptiveMethod(config)
        elif config.method == 'labo':
            self.method = LaBo(config)
        else:
            raise ValueError(f"Unknown method: {config.method}")
        
        self.writer = UnifiedConceptWriter(config)
    
    def generate_concepts(self, class_names: List[str]) -> Tuple[Dict[str, Dict[str, List[str]]], Dict]:
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
    
    def _print_summary(self, concepts: Dict[str, Dict[str, List[str]]], details: Dict):
        """Print generation summary."""
        print(f"\n{'='*60}")
        print(f"GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Method: {details['method']}")
        print(f"Total classes: {details['total_classes']}")
        print(f"Generation time: {details['generation_time']}")
        
        for variant_name, variant_concepts in concepts.items():
            total_concepts = sum(len(class_concepts) for class_concepts in variant_concepts.values())
            print(f"Total concepts in {variant_name}: {total_concepts}")
            
            if self.config.verbose:
                print(f"\nPer-class breakdown for {variant_name}:")
                for class_name, class_concepts in variant_concepts.items():
                    print(f"  {class_name}: {len(class_concepts)} concepts")
                    if class_concepts:
                        print(f"    Examples: {', '.join(class_concepts[:3])}")
    
    def save_results(self, concepts: Dict[str, Dict[str, List[str]]], details: Dict, 
                    class_names: List[str], dataset_name: str = "cifar10"):
        """Save generated concepts in the appropriate format for each method."""
        if self.config.method == "label_free_cbm":
            self.writer.save_label_free_outputs(concepts, dataset_name)
        elif self.config.method == "labo":
            self.writer.save_labo_outputs(concepts, dataset_name)
        elif self.config.method == "LM4CV":
            self.writer.save_lm4cv_outputs(concepts, dataset_name)
        
        # Save metadata
        metadata_path = os.path.join(self.config.output_dir, f"{dataset_name}_{self.config.method}_metadata.json")
        metadata = {
            'dataset_info': {
                'dataset_name': dataset_name,
                'total_classes': len(class_names),
                'class_names': class_names
            },
            'generation_info': {
                'method': self.config.method,
                'generation_timestamp': datetime.now().isoformat(),
            },
            'config': {
                'model_name': self.config.model_name,
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens,
                'length_threshold': self.config.length_threshold,
                'class_similarity_threshold': self.config.class_similarity_threshold,
                'concept_similarity_threshold': self.config.concept_similarity_threshold
            },
            'generation_details': details
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f" Metadata saved to: {metadata_path}")

    @staticmethod
    def load_class_names(file_path: str) -> List[str]:
        """Load class names from a text file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Class names file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Multi-Method Concept Generation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Methods:
  label_free_cbm    - Label-free CBM (ICLR 2023) - Automated CBM without manual concepts
  LM4CV             - Concise & Descriptive Attributes (2023) - Learning compact attribute sets  
  labo              - Language in a Bottle (2023) - LLM-guided concept bottlenecks

Examples:
  # Use Label-free CBM method
  python script.py --classes classes.txt --method label_free_cbm --verbose
  
  # Use Concise & Descriptive method with 16 target concepts
  python script.py --classes classes.txt --method LM4CV --target-concepts 16
  
  # Use LaBo method with 30 concepts per class
  python script.py --classes classes.txt --method labo --concepts-per-class 30
        """
    )
    
    # Required arguments
    parser.add_argument("--classes", "-c", required=True,
                       help="Path to file containing class names (one per line)")
    parser.add_argument("--method", "-m", required=True,
                       choices=['label_free_cbm', 'LM4CV', 'labo'],
                       help="Concept generation method to use")
    
    # Optional arguments
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for generated concepts")
    parser.add_argument("--dataset", default="cifar10",
                       help="Dataset name for output files")
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
        print(f" Loaded {len(class_names)} class names from {args.classes}")
        if args.verbose:
            print(f"Classes: {', '.join(class_names)}")
    except FileNotFoundError as e:
        print(f" Error: {e}")