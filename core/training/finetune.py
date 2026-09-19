"""微调服务（PEFT/LoRA，适配 8GB GPU）——幻觉抑制版。

相对原实现的改造（Factuality-aware）：
1. prompt 掩码：把 <human>/<system> 部分的 labels 置 -100，loss 只对
   <assistant> 输出 token 计算，避免模型只背 prompt。
2. 验算步加权：output 中命中"验证/校验/验算/代回/核对/检查"等关键词的
   token 赋予更高 loss 权重，迫使模型优先学对推理与验证路径，而非只求流畅。
3. 切出独立验证集并用 EarlyStoppingCallback 做早停，训练盯着验证 loss 而非死背。
4. epochs 默认 3（可由配置覆盖）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from ..llm.keys import load as load_cfg
from .datasets import load_dataset
from .optimize import build_lora_config, detect_gpu, recommended_lora_r

from transformers import Trainer, EarlyStoppingCallback

# 验算步关键词：命中即以更高权重学习
_VERIFY_KEYWORDS = ("验证", "校验", "验算", "代回", "核对", "检查", "反向验证")
_VERIFY_WEIGHT = 1.5          # 事实/验算陈述 token 权重（原 2.0 改为 1.5）
_NORM_WEIGHT = 1.0            # 普通 output token 权重
_VERIFY_RATIO = 0.90          # 训练/验证划分比例

_ASSISTANT_MARK = "<|im_start|>assistant"


class WeightedTrainer(Trainer):
    """Factuality-aware Trainer：继承 transformers.Trainer，重写 compute_loss
    做 per-token 加权（验算/事实步 1.5x、prompt 掩码），复用其训练循环/早停/保存。

    不传 compute_loss/compute_loss_func 参数（新版两者签名均不匹配），
    而是覆写稳定的 compute_loss 方法钩子，跨 transformers 版本兼容。"""

    def __init__(self, model, args, train_dataset, eval_dataset,
                 train_weights, eval_weights, data_collator):
        import torch
        from transformers import Trainer, EarlyStoppingCallback
        self.torch = torch
        self._tweights = train_weights
        self._eweights = eval_weights
        super().__init__(
            model=model, args=args,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        torch = self.torch
        labels = inputs.pop("labels")
        weights = inputs.pop("weights", None)
        outputs = model(**inputs)
        logits = outputs.logits
        ce = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)).contiguous(),
            labels.view(-1), reduction="none", ignore_index=-100)
        if weights is None:
            loss = ce.mean()
        else:
            # -100 处权重置 0，其余用归一化权重（验算/事实步更高）
            w = torch.where(labels.view(-1) == -100,
                            torch.zeros_like(ce), weights.view(-1).float())
            valid = (w > 0).sum()
            loss = (ce * w).sum() / valid.clamp(min=1)
        return (loss, outputs) if return_outputs else loss


def _mask_and_weights(tokenizer, text: str):
    """返回 (ids, attn, labels, weights)；labels= -100 于 prompt 部分，
    输出命中验算关键词的 token 权重更高。"""
    tok = tokenizer(text, truncation=True, max_length=512,
                    return_offsets_mapping=True)
    ids = tok["input_ids"]
    offsets = tok["offset_mapping"]
    labels = ids[:]
    weights = [_NORM_WEIGHT] * len(ids)
    # 定位 assistant 输出起点（prompt 结束处）
    marker = text.rfind(_ASSISTANT_MARK)
    prompt_end = marker if marker >= 0 else None
    # 找出输出区验算关键词出现的 [start,end) 字符区间集合
    verify_spans = []
    lo = marker if marker >= 0 else 0
    if marker >= 0:
        for kw in _VERIFY_KEYWORDS:
            i = lo
            while True:
                j = text.find(kw, i)
                if j < 0:
                    break
                verify_spans.append((j, j + len(kw)))
                i = j + len(kw)
    for ti, (s, e) in enumerate(offsets):
        if prompt_end is not None and e is not None and e <= prompt_end:
            labels[ti] = -100          # prompt 不参与 loss
            weights[ti] = 0.0
            continue
        if s is None or e is None:
            weights[ti] = 0.0
            continue
        # 命中验算关键词区间则加权
        for vs, ve in verify_spans:
            if s < ve and e > vs:
                weights[ti] = _VERIFY_WEIGHT
                break
    return ids, tok["attention_mask"], labels, weights


class FinetuneSession:
    def __init__(self, base_model: str | None = None, output_dir: str | None = None,
                 quantization: str | None = None, lora_r: int | None = None,
                 learning_rate: float | None = None, epochs: int | None = None,
                 batch_size: int | None = None, use_gpu: bool | None = None,
                 log: Callable[[str], None] | None = None):
        cfg = load_cfg().get("training", {})
        self.base_model = base_model or cfg.get("base_model")
        self.output_dir = Path(output_dir or cfg.get("output_dir", "./artifacts"))
        self.quantization = quantization or cfg.get("quantization", "4bit")
        self.lora_r = lora_r or cfg.get("lora_r", 16)
        self.learning_rate = float(learning_rate or cfg.get("learning_rate", 2e-4))
        self.epochs = int(epochs or cfg.get("epochs", 3))
        self.batch_size = int(batch_size or cfg.get("batch_size", 4))
        self.use_gpu = bool(use_gpu if use_gpu is not None else cfg.get("use_gpu", True))
        self.log = log or (lambda s: None)

    def tune(self, dataset_path: str | Path) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import torch
            from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                      TrainingArguments)
            from peft import get_peft_model, prepare_model_for_kbit_training
        except ImportError as e:
            return {"ok": False, "error": f"缺少训练依赖: {e}"}

        gpu = detect_gpu()
        if not self.use_gpu and gpu["available"]:
            self.log(f"检测到 GPU {gpu['name']}，但已禁用；将用 CPU 慢速演示。")
        self.log(f"设备: {gpu}，量化: {self.quantization}，LoRA r={self.lora_r}，epochs={self.epochs}")

        texts = load_dataset(dataset_path)
        if not texts:
            return {"ok": False, "error": "数据集为空或路径无效"}
        self.log(f"加载样本 {len(texts)} 条，按 {1-_VERIFY_RATIO:.0%} 作验证集")

        cfg = build_lora_config(self.base_model, self.lora_r, self.quantization)
        self.log(f"加载基座模型 {self.base_model} ...")
        model = AutoModelForCausalLM.from_pretrained(self.base_model, **cfg["load_kwargs"])
        if cfg["load_kwargs"]:
            model = prepare_model_for_kbit_training(model)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = get_peft_model(model, cfg["lora_config"])
        self.log("PEFT 注入完成")
        model.print_trainable_parameters()

        tr_args = TrainingArguments(
            output_dir=str(self.output_dir / "logs"),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=2,
            learning_rate=self.learning_rate,
            fp16=self.use_gpu and gpu["available"],
            logging_steps=1,
            save_steps=1000,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            eval_strategy="steps",
            eval_steps=200,
            report_to=[],
            remove_unused_columns=False,
        )

        n = len(texts)
        split = int(n * _VERIFY_RATIO)
        train_texts, eval_texts = texts[:split], texts[split:]

        def build(part):
            ids, attns, labs, wts = [], [], [], []
            for t in part:
                i, a, l, w = _mask_and_weights(tokenizer, t)
                ids.append(i); attns.append(a); labs.append(l); wts.append(w)
            return ids, attns, labs, wts

        ti, ta, tl, tw = build(train_texts)
        ei, ea, el, ew = build(eval_texts)

        import datasets as hfds
        def mk(ids, attns, labs, wts):
            return hfds.Dataset.from_dict(
                {"input_ids": ids, "attention_mask": attns, "labels": labs,
                 "weights": wts})

        tds = mk(ti, ta, tl, tw)
        eds = mk(ei, ea, el, ew)

        def collator(b):
            pad = tokenizer.pad_token_id or tokenizer.eos_token_id
            L = max(len(x["input_ids"]) for x in b)
            ids, am, la, wt = [], [], [], []
            for x in b:
                n = len(x["input_ids"])
                pn = L - n
                ids.append(x["input_ids"] + [pad] * pn)
                am.append(x["attention_mask"] + [0] * pn)
                la.append(x["labels"] + [-100] * pn)
                wt.append(x["weights"] + [0.0] * pn)
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(am),
                "labels": torch.tensor(la, dtype=torch.long),
                "weights": torch.tensor(wt, dtype=torch.float),
            }

        trainer = WeightedTrainer(model, tr_args, tds, eds, tw, ew, collator)
        self.log("开始训练（loss 已做 prompt 掩码 + 验算步加权，验证集早停）...")
        trainer.train()
        save_dir = self.output_dir / "adapter"
        trainer.save_model(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        self.log(f"微调完成，适配器已保存: {save_dir}")
        return {"ok": True, "adapter": str(save_dir),
                "vram_gb": gpu["vram_gb"], "samples": len(texts),
                "epochs": self.epochs}


def quick_finetune(dataset_path: str | Path, **kw) -> dict:
    return FinetuneSession(**kw).tune(dataset_path)