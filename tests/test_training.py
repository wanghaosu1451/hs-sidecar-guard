"""三大研究创新训练模块测试（离线，不依赖 torch/网络）。"""
from __future__ import annotations

from core.training.ced_moe import CedMoeSession
from core.training.opd_distill import OpdDistillSession, build_teacher_prompt
from core.training.fp4_qat import Fp4QatSession, swiglu_clamped, quantize_fp4, anticipatory_balance


# ---------- ① CED-MoE 异构显存调度 ----------
def test_ced_plan_schedule(tmp_path):
    s = CedMoeSession(vram_gb=8.0, total_layers=40, encoder_layers=20)
    r = s.run(str(tmp_path))
    assert r["ok"] is True
    assert 0 <= r["gpu_usage_pct"] <= 100
    # 计划落盘
    import json
    plan = json.loads((tmp_path / "ced_moe_plan.json").read_text(encoding="utf-8"))
    assert len(plan["schedule"]) == 40


def test_ced_plan_gpu_first_then_cpu():
    s = CedMoeSession(vram_gb=1.0, total_layers=40, encoder_layers=20, hidden=4096)
    schedule, _ = s.plan()
    # 前若干层驻留 GPU，之后层卸载到 CPU（异构调度的核心断言）
    assert schedule[0]["device"] == "gpu"
    assert any(l["device"] == "cpu" for l in schedule)


# ---------- ③ FP4 QAT + SwiGLU 钳制（纯计算） ----------
def test_swiglu_clamped_bounds():
    for a in (2.0, -2.0, 20.0):
        y = swiglu_clamped(a, 1.0, clamp=8.0)
        assert -8.0 - 1e-9 <= y <= 8.0 + 1e-9


def test_quantize_fp4_errors_small():
    err = abs(0.5 - quantize_fp4(0.5, 4))
    assert err < 0.1


def test_anticipatory_balance_corrects_overload():
    loads = [100, 10, 5, 2, 1, 1]
    bias = anticipatory_balance(loads, 0.05)
    assert len(bias) == len(loads)
    # 最热专家应得到负偏置(被抑制)，最冷专家正偏置(被提升)
    assert bias[loads.index(max(loads))] <= bias[loads.index(min(loads))]


def test_fp4_qat_run(tmp_path):
    r = Fp4QatSession(bits=4, clamp=8.0).run(str(tmp_path))
    assert r["ok"] is True and r["swiglu_clamp_active"] is True
    assert (tmp_path / "fp4_qat_plan.json").exists()


# ---------- ② OPD 策略蒸馏 ----------
def test_opd_build_prompt_contains_fields():
    p = build_teacher_prompt("写一个函数")
    assert "reasoning" in p and "tool_calls" in p and "output" in p


def test_opd_generate_graceful_when_teacher_down(monkeypatch, tmp_path):
    import core.llm.gateway as gw

    def boom(messages, provider_model=None):
        raise OSError("no network / no teacher key")

    monkeypatch.setattr(gw, "chat", boom)
    s = OpdDistillSession(tasks=["一个任务"], output_dir=str(tmp_path), max_steps=1)
    r = s.run()
    assert r["ok"] is False
    assert "生成任何样本" in r.get("error", "")
    assert (tmp_path / "opd_dataset.jsonl").exists()


def test_opd_generate_ok_when_teacher_valid_json(monkeypatch, tmp_path):
    import core.llm.gateway as gw

    def fake(messages, provider_model=None):
        return {"content": '{"reasoning": "逐步思考", "tool_calls": [], "output": "回答"}'}

    monkeypatch.setattr(gw, "chat", fake)
    s = OpdDistillSession(tasks=["任务A", "任务B"], output_dir=str(tmp_path), max_steps=2)
    r = s.run()
    assert r["ok"] is True and r["generated"] == 2
    lines = (tmp_path / "opd_dataset.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    import json
    assert json.loads(lines[0])["instruction"] == "任务A"