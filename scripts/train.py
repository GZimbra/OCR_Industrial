from __future__ import annotations

import argparse
import csv
import sqlite3
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import cer, wer

from htr_local.config import DB_PATH, MODEL_DIR, DATA_DIR
from htr_local.database import now


class CorrectedCells(Dataset):
    def __init__(self, rows, processor): self.rows, self.processor = rows, processor
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        path, text = self.rows[index]
        image = Image.open(path).convert("RGB")
        pixels = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(text, padding="max_length", max_length=64, truncation=True).input_ids
        labels = [token if token != self.processor.tokenizer.pad_token_id else -100 for token in labels]
        return {"pixel_values": pixels, "labels": torch.tensor(labels)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int, default=3); args = parser.parse_args()
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute("SELECT image_path,corrected_text FROM cells WHERE reviewed=1 AND corrected_text<>''").fetchall()
    if len(rows) < 20: raise SystemExit("São necessárias pelo menos 20 células revisadas.")
    source = MODEL_DIR
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    output = DATA_DIR / "model_checkpoints" / version
    processor = TrOCRProcessor.from_pretrained(source, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(source, local_files_only=True)
    split = max(1, int(len(rows) * .9)); train, valid = rows[:split], rows[split:]
    def metrics(prediction):
        predicted = processor.batch_decode(prediction.predictions, skip_special_tokens=True)
        labels = prediction.label_ids.copy(); labels[labels == -100] = processor.tokenizer.pad_token_id
        references = processor.batch_decode(labels, skip_special_tokens=True)
        return {"cer": cer(references, predicted), "wer": wer(references, predicted)}
    settings = Seq2SeqTrainingArguments(output_dir=str(output), num_train_epochs=args.epochs, per_device_train_batch_size=2, per_device_eval_batch_size=2, eval_strategy="epoch", save_strategy="epoch", predict_with_generate=True, fp16=torch.cuda.is_available(), report_to="none")
    trainer = Seq2SeqTrainer(model=model, args=settings, train_dataset=CorrectedCells(train, processor), eval_dataset=CorrectedCells(valid, processor), compute_metrics=metrics)
    trainer.train(); final = trainer.evaluate(); trainer.save_model(output); processor.save_pretrained(output)
    metric_path = DATA_DIR / "metrics_log.csv"; new_file = not metric_path.exists()
    with metric_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file: writer.writerow(["versao", "data_treino", "cer", "wer", "n_exemplos_treino"])
        writer.writerow([version, now(), final.get("eval_cer"), final.get("eval_wer"), len(train)])
    with sqlite3.connect(DB_PATH) as db:
        db.execute("INSERT INTO model_versions(version,checkpoint_path,cer,wer,train_examples,created_at,active) VALUES(?,?,?,?,?,?,0)", (version, str(output), final.get("eval_cer"), final.get("eval_wer"), len(train), now()))
    print(f"Checkpoint {version}: CER={final.get('eval_cer'):.4f}, WER={final.get('eval_wer'):.4f}. Valide antes de promover para base.")

if __name__ == "__main__": main()
