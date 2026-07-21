from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image


class LocalRecognizer:
    """TrOCR carregado somente de um diretório local; nunca consulta a rede."""

    def __init__(self, model_dir: Path, allow_stub: bool = False):
        self.model_dir = model_dir
        self.allow_stub = allow_stub
        self.processor = self.model = self.torch = None

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.model_dir.exists():
            if self.allow_stub:
                return
            raise RuntimeError(f"Checkpoint local ausente: {self.model_dir}. Execute scripts/cache_model.py em uma máquina autorizada.")
        # Reforça o modo offline mesmo que o ambiente tenha acesso à internet.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.torch = torch
        self.processor = TrOCRProcessor.from_pretrained(str(self.model_dir), local_files_only=True)
        self.model = VisionEncoderDecoderModel.from_pretrained(str(self.model_dir), local_files_only=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def predict(self, bgr_image: np.ndarray) -> tuple[str, float]:
        return self.predict_many([bgr_image])[0]

    def predict_many(self, bgr_images: list[np.ndarray], batch_size: int = 8) -> list[tuple[str, float]]:
        self.load()
        if self.model is None:
            return [("", 0.0) for _ in bgr_images]
        results = []
        for start in range(0, len(bgr_images), batch_size):
            images = [Image.fromarray(item[:, :, ::-1]) for item in bgr_images[start:start + batch_size]]
            pixels = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
            with self.torch.inference_mode():
                output = self.model.generate(pixels, return_dict_in_generate=True, output_scores=True, max_new_tokens=64)
            texts = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
            # Cada score possui uma linha por imagem; calcula média geométrica por sequência.
            token_confidences = [[] for _ in images]
            for score in output.scores:
                maximum = self.torch.softmax(score, dim=-1).max(dim=-1).values.detach().cpu().numpy()
                for index, value in enumerate(maximum): token_confidences[index].append(float(value))
            for text, probabilities in zip(texts, token_confidences):
                confidence = float(np.exp(np.mean(np.log(np.clip(probabilities, 1e-8, 1.0))))) if probabilities else 0.0
                results.append((text.strip(), confidence))
        return results
