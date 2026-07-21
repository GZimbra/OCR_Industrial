"""Único script que usa internet: baixa os pesos para transporte e uso offline."""
from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

TARGET = Path(__file__).resolve().parent.parent / "data" / "model_checkpoints" / "base"
MODEL = "microsoft/trocr-base-handwritten"
TARGET.mkdir(parents=True, exist_ok=True)
TrOCRProcessor.from_pretrained(MODEL).save_pretrained(TARGET)
VisionEncoderDecoderModel.from_pretrained(MODEL).save_pretrained(TARGET)
print(f"Checkpoint salvo em {TARGET}")
