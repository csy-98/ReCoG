# src/models/mini_cpm_reasoning.py
import torch
import numpy as np
import random
from transformers import AutoModel, AutoTokenizer
from r2p.compute_confidence import ConfidenceCalculator
from r2p.model_interface import ModelInterface
from r2p.model_adapters import MiniCPMAdapter
from r2p.prompt_generator import BasePromptGenerator

class MiniCPMModel(ModelInterface):
    """Implementation for MiniCPM model."""

    def chat(self, msgs):
        result = self.model.chat(msgs=msgs, tokenizer=self.tokenizer, **self.generation_kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        if isinstance(result, dict):
            answer = result.get("text") or result.get("answer") or result.get("response") or str(result)
            return None, answer
        return None, result

class MiniCPMReasoning:
    def __init__(self,
                 model_path: str = "openbmb/MiniCPM-o-2_6",
                 device: str = "cuda",
                 attn_implementation: str = "sdpa",
                 torch_dtype = torch.bfloat16,
                 seed: int = 42):
        print("Loading Description Generator")
        self.set_seed(seed)
        self.device = device
        self.model_interface = MiniCPMModel(model_path, device, attn_implementation, torch_dtype)
        self.adapter = MiniCPMAdapter(self.model_interface.tokenizer)
        self.conf_calculator = ConfidenceCalculator(self.model_interface.tokenizer)
        self.prompt_generator = BasePromptGenerator()

    def set_seed(self, seed: int):
        print(f"Setting seed: {seed}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _chat(self, msgs, task, target_tokens=None):
        """Helper method to call the model's chat API with the provided messages."""

        token_info = None
        outputs, answer = self.model_interface.chat(msgs)
        if outputs is None:
            labels = target_tokens or (["yes", "no"] if task == "recognition" else ["a", "b"])
            confidence = (
                {label + "_confidence": 0.0 for label in labels}
                | {label + "_score": 0.0 for label in labels}
                | {"margin": 0.0}
            )
        else:
            confidence = self.conf_calculator.calculate_confidence(outputs, task, target_tokens=target_tokens)
        return confidence, answer, token_info

    def generate_personalized_caption(self, test_image, concept_name, category_name, answer_format):
        answer_format = {
            "Caption": "<caption>"
        }
        prompt = self.prompt_generator.get_personalized_caption_prompt(test_image, concept_name, category_name, answer_format)
        msgs = self.adapter.format_personalized_caption_msgs(test_image, prompt)
        outputs, caption = self.model_interface.chat(msgs)
        if "sorry" in caption:
            caption = f"A photo of {concept_name}"
        return caption

    def reason_with_multiple_text(self, test_image, test_question, descriptions, task, answer_format, target_tokens):
        if 'A' in answer_format.keys():
            prompt = self.prompt_generator.get_attribute_based_text_options_prompt(test_question, descriptions, answer_format)
        else:
            prompt = self.prompt_generator.get_text_options_prompt(test_question, descriptions, answer_format)
        msgs = self.adapter.format_text_options_msgs(test_image, prompt)
        confidence, answer, token_info = self._chat(msgs, task, target_tokens=target_tokens)
        return confidence, answer, token_info

    def reason_with_only_text(self, test_image, test_question, descriptions, task, answer_format, target_tokens=None):
        prompt = self.prompt_generator.get_text_options_prompt(test_question, descriptions, answer_format)
        # print("\n===== PROMPT (baseline text-only) =====\n", prompt, "\n=======================================\n")
        msgs = self.adapter.format_text_options_msgs(test_image, prompt)
        confidence, answer, token_info = self._chat(msgs, task, target_tokens=target_tokens)
        return confidence, answer, token_info

    def reason_image2image_plus_text(self, test_image, ret_image, test_question, descriptions, task, answer_format):
        prompt = self.prompt_generator.get_image2image_plus_text_comparison_prompt(test_question, descriptions, answer_format)
        # print("\n===== PROMPT (baseline image2image) =====\n", prompt, "\n=========================================\n")
        msgs = self.adapter.format_image2image_plus_text_comparison_msgs(test_image, ret_image, prompt)
        confidence, answer, token_info = self._chat(msgs, task)
        return confidence, answer, token_info
