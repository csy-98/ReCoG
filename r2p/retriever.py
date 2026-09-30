"""R2P CLIP image encoding, with local paths and lazy runtime dependencies."""

from contextlib import ExitStack

CLIP_MODEL = "openai/clip-vit-large-patch14-336"


class ClipEncoder:
    def __init__(self, model_path=CLIP_MODEL, device="cuda"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model_path).eval().to(device)
        self.processor = CLIPProcessor.from_pretrained(model_path)

    def encode(self, paths, batch_size=8):
        from PIL import Image

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        embeddings = []
        for start in range(0, len(paths), batch_size):
            with ExitStack() as stack:
                images = [stack.enter_context(Image.open(path)).convert("RGB")
                          for path in paths[start:start + batch_size]]
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                with self.torch.inference_mode():
                    features = self.model.get_image_features(**inputs)
                    if not isinstance(features, self.torch.Tensor):
                        features = features.image_embeds
                    features = features / features.norm(p=2, dim=-1, keepdim=True)
                embeddings.extend(tuple(row) for row in features.float().cpu().tolist())
        return embeddings
