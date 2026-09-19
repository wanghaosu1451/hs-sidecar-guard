"""完整训练服务：checkpoint / 评估 / 断点续训。

复用 finetune 的配置，支持 resume、继续训练。以轻量接口呈现，供 UI 调用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..llm.keys import load as load_cfg
from .datasets import load_dataset


class TrainerSession:
    def __init__(self, output_dir: str | None = None,
                 resume_from: str | None = None,
                 log: Callable[[str], None] | None = None):
        cfg = load_cfg().get("training", {})
        self.output_dir = Path(output_dir or cfg.get("output_dir", "./artifacts"))
        self.resume_from = resume_from
        self.log = log or (lambda s: None)

    def train(self, dataset_path: str | Path, epochs: int | None = None) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import torch
            from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                      TrainingArguments, Trainer)
            from peft import get_peft_model, prepare_model_for_kbit_training
        except ImportError as e:
            return {"ok": False, "error": f"缺少训练依赖: {e}"}

        texts = load_dataset(dataset_path)
        if not texts:
            return {"ok": False, "error": "数据集为空或路径无效"}
        self.log(f"加载样本 {len(texts)} 条，epochs={epochs}")

        cfg = load_cfg().get("training", {})
        base = cfg.get("base_model", "Qwen/Qwen2.5-0.5B-Instruct")
        self.log(f"加载基座 {base} ...")
        model = AutoModelForCausalLM.from_pretrained(base)
        tokenizer = AutoTokenizer.from_pretrained(base)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 断点续训：从已保存 checkpoint 构建 Trainer 时自动 resume
        tr_args = TrainingArguments(
            output_dir=str(self.output_dir / "train_logs"),
            num_train_epochs=int(epochs or cfg.get("epochs", 1)),
            per_device_train_batch_size=int(cfg.get("batch_size", 4)),
            learning_rate=float(cfg.get("learning_rate", 2e-4)),
            logging_steps=1, save_steps=100, save_total_limit=3,
            report_to=[], remove_unused_columns=False,
        )

        def tokenize(sample):
            out = tokenizer(sample, truncation=True, max_length=512, padding=False)
            out["labels"] = out["input_ids"].copy()
            return out

        import datasets as hfds
        ds = hfds.Dataset.from_list([{"text": t} for t in texts])
        ds = ds.map(tokenize, remove_columns=["text"])

        trainer = Trainer(
            model=model, args=tr_args, train_dataset=ds,
            data_collator=lambda b: {
                "input_ids": torch.tensor([x["input_ids"] for x in b]),
                "attention_mask": torch.tensor([x["attention_mask"] for x in b]),
                "labels": torch.tensor([x["labels"] for x in b]),
            },
        )
        if self.resume_from:
            self.log(f"从 {self.resume_from} 断点续训 ...")
            trainer.train(resume_from_checkpoint=self.resume_from)
        else:
            self.log("开始训练 ...")
            trainer.train()
        save_dir = self.output_dir / "model_final"
        trainer.save_model(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        self.log(f"训练完成：{save_dir}")
        return {"ok": True, "model": str(save_dir), "samples": len(texts)}