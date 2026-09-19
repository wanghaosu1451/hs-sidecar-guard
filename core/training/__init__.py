"""微调与训练服务包（支柱C）。"""

# 研究向训练模块（三大创新方向）注册表
from .ced_moe import CedMoeSession, run_ced
from .opd_distill import OpdDistillSession, run_opd
from .fp4_qat import Fp4QatSession, run_fp4

RESEARCH_RUNNERS = {
    "CED-MoE 内存压缩": run_ced,
    "OPD 策略蒸馏": run_opd,
    "FP4 QAT": run_fp4,
}