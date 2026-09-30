"""R2P MiniCPM components; see LICENSE for the upstream notice."""

import json

class ModelAdapter:
    """Base class for model-specific message format adapters."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def format_image2image_plus_text_comparison_msgs(self, test_image, ret_image, prompt):
        raise NotImplementedError("Subclasses must implement this method.")

    def format_text_options_msgs(self, test_image, prompt):
        raise NotImplementedError("Subclasses must implement this method.")

    def format_attribute_based_text_options_msgs(self, test_image, prompt):
        raise NotImplementedError("Subclasses must implement this method.")

    def format_personalized_caption_msgs(self, test_image, prompt):
        raise NotImplementedError("Subclasses must implement this method.")

class MiniCPMAdapter(ModelAdapter):
    """Adapter for MiniCPM model message format."""

    def format_image2image_plus_text_comparison_msgs(self, test_image, ret_image, prompt):
        return [{'role': 'user', 'content': [test_image, ret_image, prompt]}]

    def format_text_options_msgs(self, test_image, prompt):
        return [{'role': 'user', 'content': [test_image, prompt]}]

    def format_attribute_based_text_options_msgs(self, test_image, prompt):
        return [{'role': 'user', 'content': [test_image, prompt]}]

    def format_personalized_caption_msgs(self, test_image, prompt):
        return [{'role': 'user', 'content': [test_image, prompt]}]

    def format_pairwise_comparison_msgs(self, test_image, concept1_image, concept2_image, prompt):
        """Format messages for pairwise comparison with three images."""
        return [{'role': 'user', 'content': [test_image, concept1_image, concept2_image, prompt]}]
