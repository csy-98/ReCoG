"""Concrete frozen MiniCPM adapters using the included R2P reasoning code."""

from contextlib import ExitStack
import json

from .schemas import PairEvidence


def parse_json_answer(raw):
    if not isinstance(raw, str):
        raise ValueError("Expected a text response from the VLM.")
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) > 2 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("The VLM must return a JSON object.")
    return value


class MiniCPMBackend:
    def __init__(self, model_path="openbmb/MiniCPM-o-2_6", device="cuda",
                 dtype="bfloat16", max_new_tokens=512, seed=42):
        import torch
        from r2p.mini_cpm_reasoning import MiniCPMReasoning
        from r2p.mini_cpm_reasoning_pairwise import PairwisePromptGenerator

        self.torch = torch
        self.reasoner = MiniCPMReasoning(model_path=model_path, device=device,
                                       torch_dtype=getattr(torch, dtype), seed=seed)
        self.reasoner.model_interface.generation_kwargs = {
            "sampling": False, "max_new_tokens": max_new_tokens,
        }
        self.pairwise_prompts = PairwisePromptGenerator()

    def chat(self, image_paths, prompt):
        from PIL import Image

        with ExitStack() as stack:
            images = [stack.enter_context(Image.open(path)).convert("RGB") for path in image_paths]
            with self.torch.inference_mode():
                _, text = self.reasoner.model_interface.chat([
                    {"role": "user", "content": [*images, prompt]}
                ])
        return parse_json_answer(text)

    def describe(self, image):
        result = self.chat([image], 'Describe the permanent physical attributes of the main '
                           'object. Ignore the background. Return only JSON with '
                           '"general" (a description) and "distinct features" (a list of cues).')
        if not isinstance(result.get("general"), str) or not result["general"].strip():
            raise ValueError("The VLM did not return a nonempty general description.")
        return result

    def extract(self, a, b, prompt):
        result = self.chat([a.support_image, b.support_image], prompt)
        left, right = result[f"{a.name}_distinct_features"], result[f"{b.name}_distinct_features"]
        if any(not isinstance(cues, list) or len(cues) != 3 for cues in (left, right)):
            raise ValueError("Extraction must return three cues in a JSON array per direction.")
        return PairEvidence(a.name, b.name, tuple(left), tuple(right))

    def judge(self, graph):
        def compare(query, a, b, _prompt):
            evidence = graph.evidence(a.name, b.name)
            prompt = self.pairwise_prompts.get_pairwise_concept_comparison_prompt(
                f"Which concept does the query image belong to: {a.name} or {b.name}?",
                {"name": a.name, "distinct_features": list(evidence.a_to_b)},
                {"name": b.name, "distinct_features": list(evidence.b_to_a)},
                {"Answer": f"{a.name} or {b.name}", "Reasoning": "visible distinctive cues"},
                shared_features=list(evidence.shared),
            )
            result = self.chat([query, a.support_image, b.support_image], prompt)
            return result["Answer"]
        return compare

    def select_r2p(self, query, candidates, descriptions):
        """R2P text-option reasoning over the retrieved unary descriptions."""
        options = {chr(65 + i): candidate.concept.name for i, candidate in enumerate(candidates)}
        if not options:
            raise ValueError("R2P selection requires at least one candidate.")
        for name in options.values():
            if not descriptions.get(name):
                raise ValueError(f"R2P requires a unary description for {name}.")
        details = {option: {"name": name, "description": descriptions[name]}
                   for option, name in options.items()}
        prompt = self.reasoner.prompt_generator.get_text_options_prompt(
            "Which option matches the query image? Answer with exactly one option letter.",
            details, {"Answer": " / ".join(options), "Reasoning": "brief visual evidence"},
        )
        result = self.chat([query], prompt)
        answer = str(result.get("Answer", "")).strip().upper()
        if answer not in options:
            raise ValueError("R2P must return exactly one valid option letter.")
        return options[answer], result
