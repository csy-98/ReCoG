"""MiniCPM-o model loading shared by the R2P and ReCoG reasoners."""

import torch
from transformers import AutoModel, AutoTokenizer


class ModelInterface:
    def __init__(self, model_path, device="cuda", attn_implementation="sdpa",
                 torch_dtype=torch.bfloat16):
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            init_vision=True,
            init_audio=False,
            init_tts=False,
        ).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.generation_kwargs = {"sampling": False, "max_new_tokens": 512}

    def chat(self, msgs):
        raise NotImplementedError
