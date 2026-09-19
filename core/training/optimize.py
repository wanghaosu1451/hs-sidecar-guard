"""低显存优化：建议参数与运行时检测（8GB GPU 友好）。

真正生效依赖 torch/peft，缺失时给出降级建议。
"""
from __future__ import annotations


def detect_gpu() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False, "name": "CPU", "vram_gb": 0}
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        }
    except Exception:  # noqa: BLE001
        return {"available": False, "name": "CPU", "vram_gb": 0}


def recommended_quantization(vram_gb: float) -> str:
    """按显存推荐量化等级。"""
    if vram_gb <= 0:
        return "none"      # 纯 CPU
    if vram_gb < 6:
        return "4bit"
    if vram_gb < 12:
        return "8bit"
    return "none"


def recommended_lora_r(vram_gb: float) -> int:
    if vram_gb < 6:
        return 8
    if vram_gb < 12:
        return 16
    return 32


def build_lora_config(base_model: str, lora_r: int = 16,
                      quantization: str = "4bit"):
    """构建 PEFT/Transformers 配置字典（懒加载依赖，缺库时抛可读错误）。"""
    try:
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig
    except ImportError as e:
        raise RuntimeError(
            f"缺少训练依赖，请安装 torch/transformers/peft: {e}"
        ) from e

    load_kwargs = {}
    if quantization in ("4bit", "8bit"):
        load_in = 4 if quantization == "4bit" else 8
        bnb = BitsAndBytesConfig(
            load_in_4bit=(load_in == 4), load_in_8bit=(load_in == 8),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=__import__("torch").bfloat16,
        )
        load_kwargs = {
            "quantization_config": bnb,
            "use_cache": False,
            "device_map": "auto",
        }

    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return {"load_kwargs": load_kwargs, "lora_config": lora}