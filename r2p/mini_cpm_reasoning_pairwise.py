# src/models/mini_cpm_reasoning_pairwise.py
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

class PairwisePromptGenerator(BasePromptGenerator):
    """Extended prompt generator for pairwise concept comparison."""

    def get_pairwise_concept_comparison_prompt(self, test_question, concept1_info, concept2_info, answer_format, shared_features=None):
        """
        Generate prompt for comparing test image with two concepts in a pairwise manner.

        Args:
            test_question: Question about which concept the test image belongs to
            concept1_info: Dict with concept1 name and pairwise distinct features
            concept2_info: Dict with concept2 name and pairwise distinct features
            answer_format: Expected JSON response format
            shared_features: Optional list of shared features between the two concepts
        """
        shared_section = ""
        if shared_features:
            shared_section = f"""
The following features are shared by BOTH concepts. Do NOT use them as a basis for your decision:
{self._format_features(shared_features)}

You MUST base your decision ONLY on the distinctive features listed below, and strictly ignore the shared features above:
        """

        prompt = f"""You are a helpful AI agent specializing in image analysis and object recognition.
You are given three images:
- **Image 1**: Query image (the image to classify)
- **Image 2**: Reference image for concept "{concept1_info['name']}"
- **Image 3**: Reference image for concept "{concept2_info['name']}"

Additionally, you are provided with information about these two concepts:
{shared_section}
**Distinctive features of {concept1_info['name']}:**
{self._format_features(concept1_info['distinct_features'])}

**Distinctive features of {concept2_info['name']}:**
{self._format_features(concept2_info['distinct_features'])}

**Task:**
- Compare Image 1 (query) with Image 2 and Image 3.
- Determine which concept (Image 2 or Image 3) the query image belongs to.
- Focus on the distinctive features provided above to make your decision.
- **Ignore superficial details** such as pose, background, or lighting. Focus on permanent features.

**Question:** {test_question}

- Provide your reasoning based on the distinctive features.
- Generate your response in JSON format: {answer_format}
- Output only the JSON response. DO NOT output any additional text.
"""
        return prompt

    def get_pairwise_caption_selection_prompt(
        self,
        test_question,
        concept1_info,
        concept2_info,
        answer_format,
        shared_features=None,
    ):
        """
        Generate a ReCoG captioning/recall prompt for a single candidate pair.

        This keeps the R2P recall_hard task framing ("which description matches
        the image?") while changing the candidate set from top-K to A/B.
        """
        shared_section = ""
        if shared_features:
            shared_section = f"""
The following features are shared by BOTH options. Do NOT use them as a basis for your decision:
{self._format_features(shared_features)}

Base your decision on the distinctive evidence below:
        """

        prompt = f"""You are a helpful AI agent specializing in image analysis and personalized captioning.
You are given three images:
- **Image 1**: Query image.
- **Image 2**: Reference image for option A, concept "{concept1_info['name']}".
- **Image 3**: Reference image for option B, concept "{concept2_info['name']}".

Before generating a caption, select which personalized concept best matches Image 1.

{shared_section}
**Option A. {concept1_info['name']}**
Distinctive evidence:
{self._format_features(concept1_info['distinct_features'])}

**Option B. {concept2_info['name']}**
Distinctive evidence:
{self._format_features(concept2_info['distinct_features'])}

**Question:** {test_question}

- Compare Image 1 with options A and B.
- Choose the option whose description and pair-specific evidence best match Image 1.
- Ignore superficial details such as pose, background, or lighting. Focus on permanent features.
- Do not list separate A/B feature summaries in the response.
- You MUST include an "Answer" field, and its value MUST be exactly "A" or "B".
- Output exactly one JSON object in this format:
{{
  "Reasoning": "<one short sentence citing the decisive evidence>",
  "Answer": "A"
}}
- Replace "A" with "B" only when option B is the better match.
- Output only the JSON response. DO NOT output any additional text.
"""
        return prompt

    def _format_features(self, features):
        """Format list of features as bullet points."""
        if isinstance(features, list):
            return "\n".join([f"  - {feat}" for feat in features])
        return str(features)

class MiniCPMReasoningPairwise:
    """Extended MiniCPM reasoning class with pairwise league comparison support."""

    def __init__(self,
                 model_path: str = "openbmb/MiniCPM-o-2_6",
                 device: str = "cuda",
                 attn_implementation: str = "sdpa",
                 torch_dtype = torch.bfloat16,
                 seed: int = 42):
        print("Loading Pairwise Description Generator")
        self.set_seed(seed)
        self.device = device
        self.model_interface = MiniCPMModel(model_path, device, attn_implementation, torch_dtype)
        self.adapter = MiniCPMAdapter(self.model_interface.tokenizer)
        self.conf_calculator = ConfidenceCalculator(self.model_interface.tokenizer)
        self.prompt_generator = PairwisePromptGenerator()

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

    def reason_pairwise_concept_comparison(self, test_image, concept1_image, concept2_image,
                                          concept1_info, concept2_info, answer_format,
                                          shared_features=None):
        """
        Perform pairwise comparison between two concepts for a test image.

        Args:
            test_image: PIL Image of the query
            concept1_image: PIL Image of concept 1 reference
            concept2_image: PIL Image of concept 2 reference
            concept1_info: Dict with 'name' and 'distinct_features' for concept 1
            concept2_info: Dict with 'name' and 'distinct_features' for concept 2
            answer_format: Expected JSON response format
            shared_features: Optional list of shared features between the two concepts

        Returns:
            confidence: Confidence scores
            answer: Generated answer text
            token_info: Token usage information
        """
        test_question = f"Which concept does the query image (Image 1) belong to: {concept1_info['name']} or {concept2_info['name']}?"

        prompt = self.prompt_generator.get_pairwise_concept_comparison_prompt(
            test_question, concept1_info, concept2_info, answer_format,
            shared_features=shared_features
        )

        # print("\n===== PROMPT (pairwise) =====\n", prompt, "\n=============================\n")

        # Format messages with three images
        msgs = self.adapter.format_pairwise_comparison_msgs(
            test_image, concept1_image, concept2_image, prompt
        )

        confidence, answer, token_info = self._chat(msgs, 'recognition')
        return confidence, answer, token_info

    def reason_pairwise_caption_selection(self, test_image, concept1_image, concept2_image,
                                          concept1_info, concept2_info, test_question,
                                          answer_format, shared_features=None):
        """
        Select between two personalized concepts for recall_hard/captioning.

        Unlike recognition, the expected answer is an A/B option. The final
        caption is generated later from the league winner.
        """
        prompt = self.prompt_generator.get_pairwise_caption_selection_prompt(
            test_question, concept1_info, concept2_info, answer_format,
            shared_features=shared_features
        )

        msgs = self.adapter.format_pairwise_comparison_msgs(
            test_image, concept1_image, concept2_image, prompt
        )

        confidence, answer, token_info = self._chat(
            msgs, 'recall_hard', target_tokens=["a", "b"]
        )
        return confidence, answer, token_info

    def generate_personalized_caption(self, test_image, concept_name, category_name, answer_format):
        """
        Generate a personalized caption for the test image using the predicted concept.

        Args:
            test_image: PIL Image to caption
            concept_name: Name of the predicted concept
            category_name: Category of the concept
            answer_format: Expected JSON response format

        Returns:
            caption: Generated caption text
        """
        answer_format = {
            "Caption": "<caption>"
        }
        prompt = self.prompt_generator.get_personalized_caption_prompt(test_image, concept_name, category_name, answer_format)
        msgs = self.adapter.format_personalized_caption_msgs(test_image, prompt)
        outputs, caption = self.model_interface.chat(msgs)
        if "sorry" in caption.lower():
            caption = f"A photo of {concept_name}"
        return caption
