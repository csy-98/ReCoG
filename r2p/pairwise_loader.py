# src/utils/pairwise_loader.py
import os
import json
import re
from typing import Dict, List, Optional, Tuple

class PairwiseDistinctFeaturesLoader:
    """Loader for pairwise distinct features between concepts."""

    def __init__(self, pairwise_features_dir: str):
        """
        Initialize the loader with the directory containing pairwise feature files.

        Args:
            pairwise_features_dir: Path to directory with pairwise distinct feature JSON files
                                   e.g., "<local-pairwise-directory>"
        """
        self.pairwise_features_dir = pairwise_features_dir
        self._cache = {}

    def load_pairwise_features(self, concept1: str, concept2: str) -> Optional[Dict]:
        """
        Load pairwise distinct features for two concepts.

        Args:
            concept1: First concept name
            concept2: Second concept name

        Returns:
            Dict with keys:
                - 'concept1_distinct_features': List of distinct features for concept1
                - 'concept2_distinct_features': List of distinct features for concept2
            Returns None if file not found.
        """
        # Normalize concept order (alphabetical) to match file naming convention
        concepts_sorted = sorted([concept1, concept2])
        cache_key = f"{concepts_sorted[0]}_vs_{concepts_sorted[1]}"

        # Check cache first
        if cache_key in self._cache:
            return self._get_features_in_order(self._cache[cache_key], concept1, concept2)

        # Construct file path
        filename = f"{cache_key}_distinct.json"
        filepath = os.path.join(self.pairwise_features_dir, filename)

        # Load from file
        if not os.path.exists(filepath):
            print(f"Warning: Pairwise features file not found: {filepath}")
            return None

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            data = self._normalize_pairwise_data(data)

            # Cache the loaded data
            self._cache[cache_key] = data

            # Return features in the requested order
            return self._get_features_in_order(data, concept1, concept2)

        except Exception as e:
            print(f"Error loading pairwise features from {filepath}: {e}")
            return None

    def _normalize_pairwise_data(self, data: Dict) -> Dict:
        """Support both direct feature JSONs and raw_output-wrapped outputs."""
        if "raw_output" not in data:
            return data

        raw_output = data.get("raw_output")
        if not isinstance(raw_output, str):
            return data

        raw_output = raw_output.strip()
        raw_output = raw_output.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        parsed = self._parse_feature_lists_from_text(raw_output)
        return parsed if parsed else data

    def _parse_feature_lists_from_text(self, text: str) -> Dict[str, List[str]]:
        """Parse simple *_distinct_features lists from non-strict JSON text."""
        parsed = {}
        current_key = None

        for line in text.splitlines():
            key_match = re.match(r'\s*"([^"]+_distinct_features)"\s*:\s*\[', line)
            if key_match:
                current_key = key_match.group(1)
                parsed[current_key] = []
                continue

            if current_key is None:
                continue

            stripped = line.strip()
            if stripped.startswith("]"):
                current_key = None
                continue
            if not stripped:
                continue

            stripped = stripped.rstrip(",").strip()
            if stripped.startswith('"') and stripped.endswith('"'):
                stripped = stripped[1:-1]
            if stripped:
                parsed[current_key].append(stripped)

        return parsed

    def _get_features_in_order(self, data: Dict, concept1: str, concept2: str) -> Dict:
        """
        Extract features from loaded data in the requested concept order.

        Args:
            data: Loaded JSON data with keys like 'concept_a_distinct_features', 'concept_b_distinct_features'
            concept1: First concept name (requested order)
            concept2: Second concept name (requested order)

        Returns:
            Dict with 'concept1_distinct_features' and 'concept2_distinct_features'
        """
        # Find the keys in the data
        concept1_key = f"{concept1}_distinct_features"
        concept2_key = f"{concept2}_distinct_features"

        if concept1_key not in data or concept2_key not in data:
            print(f"Warning: Expected keys not found in data. Keys: {list(data.keys())}")
            return None

        return {
            'concept1_distinct_features': data[concept1_key],
            'concept2_distinct_features': data[concept2_key]
        }

    def load_all_pairwise_features_for_concept(self, concept: str, other_concepts: List[str]) -> Dict[str, Dict]:
        """
        Load all pairwise features for a given concept against a list of other concepts.

        Args:
            concept: The main concept
            other_concepts: List of concepts to compare against

        Returns:
            Dict mapping other_concept_name -> pairwise_features_dict
        """
        results = {}
        for other_concept in other_concepts:
            if other_concept == concept:
                continue
            features = self.load_pairwise_features(concept, other_concept)
            if features:
                results[other_concept] = features
        return results

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()


class SharedFeaturesLoader:
    """Loader for pairwise shared features between concepts."""

    def __init__(self, shared_features_dir: str):
        self.shared_features_dir = shared_features_dir
        self._cache = {}

    def load_shared_features(self, concept1: str, concept2: str) -> Optional[List[str]]:
        """
        Load shared features for a concept pair.

        Returns:
            List of shared feature strings, or None if not found.
        """
        concepts_sorted = sorted([concept1, concept2])
        cache_key = f"{concepts_sorted[0]}_vs_{concepts_sorted[1]}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        filename = f"{cache_key}_shared.json"
        filepath = os.path.join(self.shared_features_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: Shared features file not found: {filepath}")
            return None

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            shared = data.get("shared_features", [])
            self._cache[cache_key] = shared
            return shared
        except Exception as e:
            print(f"Error loading shared features from {filepath}: {e}")
            return None
