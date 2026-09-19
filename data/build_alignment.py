# -*- coding: utf-8 -*-
"""从 ModelScope 下载的对齐数据源构建"通用反幻觉对齐"训练集。

- 正样本：common_zh_70k(通用问答) + Open-cn-Platypus(知识/数理) + ruozhi(心智)。
  转成 Alpaca 三字段 {"instruction","input":"","output"}，抽单轮 human/gpt，
  过滤过短/非中文为主的样本，并按 instruction 去重(保留更长 output)。
- 负边界样本：内置手写边界问题(虚构概念/极冷门/信息残缺)，配统一拒答话术变体，
  保证 20% 左右比例。
输出：data/train_alignment_clean.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

RAW = Path(__file__).parent / "raw_modelscope"
OUT = Path(__file__).parent / "train_alignment_clean.jsonl"

# 抽样预算（正样本）
# 注：Open-cn-Platypus 以英文为主，经中文过滤后近空，故移除该源。
BUDGET = {
    "common_zh_70k.jsonl": 6000,
    "ruozhiout_qa_cn.jsonl": -1,  # 全量
}
MAX_Q = 200   # question 最大字符
MAX_A = 900   # answer 最大字符
MIN_A = 12    # answer 最小字符


def _is_chineseish(s: str) -> bool:
    han = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    return han > 0 and han / max(len(s.strip()), 1) >= 0.4


def norm(s: str) -> str:
    import re
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_pos(path: Path, budget: int):
    out = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        convs = d.get("conversations") or []
        # 只要第一条 human -> gpt（单轮）
        if len(convs) < 2:
            continue
        q = norm(str(convs[0].get("value", "")))
        a = norm(str(convs[1].get("value", "")))
        if not (0 < len(q) <= MAX_Q and MIN_A <= len(a) <= MAX_A):
            continue
        if not (_is_chineseish(q) and _is_chineseish(a)):
            continue
        key = q
        if key in seen:
            continue
        seen.add(key)
        out.append({"instruction": q, "input": "", "output": a})
        if budget > 0 and len(out) >= budget:
            break
    return out


# ---------------- 手写边界负样本 ----------------
# 结构: (question, 拒答话术_id相对于 _NA_ 变体下标)
_NAs = [
    "该信息我暂未掌握，无法准确回答。",
    "关于这个问题，我目前没有可靠的依据，不想凭空编造，建议你核实权威来源。",
    "这一情况我暂无确切信息，为避免误导，无法给出确定答复。",
    "我不掌握与这个问题相关的事实，只能如实说不知道。",
]

# 虚构/不存在概念
_FICT = [
    "红月亮为什么会在夜里发出绿色的光？",
    "请问什么是量子永动机？",
    "巴黎铁塔的隐身涂料是哪个公司生产的？",
    "地球南北两端的地磁鼹会在什么时候集体迁回赤道？",
    "听说有种叫太阳风暴蝶的动物会预报地震，是真的吗？",
    "时间胶囊协会每年会把谁的时间封存成黄金溶液？",
    "月球背面的磁场灯塔建于哪一年？",
    "食疗界所谓的紫魂草的药用价值有多大？",
    "请介绍下马里亚纳海沟里栖居的发光鲶鱼群的生活习性。",
    "什么是第九大行星轨道上的反物质云？",
    "火星上的玻璃矿坑是谁最先开采的？",
    "人脑的第三松果体的功能是什么？",
    "中国古代有没有一种叫《千里账》的官修历书？",
    "银河系中心叫什么海？",
    "请讲解一下超声波在真空中的折射定律应用。",
    "为什么食指上没有指甲根部的月牙就说明贫血？",
    "差点被选为空间站专用燃料的菠萝汁是哪一年被淘汰的？",
    "什么是神经元冻结病？",
    "北极鱼群的集体跳崖现象在生物学上叫什么？",
    "请问如来佛的官方身高是多少厘米？",
    "世界上哪一种石头会自我再生？",
    "手掌上的生命线长短和寿命有关系吗，长的人更长寿吗？",
    "冥王星的智慧生物留下了什么遗迹？",
    "谁发明了可以在梦境中通话的头环？",
    "请问卡塔菲勒那行星公转一圈是几年？",
]
# 极冷门
_COLD = [
    "1987年安哥拉某个偏远邮局发行的邮票里，第三版图案的设计师是谁？",
    "维也纳某条小巷18世纪的地砖拼接方式是怎样的？",
    "里斯本1943年一间钟表铺的账簿上，记录的第一笔交易是多少？",
    "请问苏俾你岛上居民的传统婚嫁歌谣大致有几段？",
    "1952年一位不知名水手在汉堡写下的航海日记里提到了哪个港口？",
    "塔尔多市的青铜喷泉底部刻的那行拉丁文写的是什么？",
    "在1930年前后，萨马拉一间面粉厂的磨盘直径是多少？",
    "那位在1961年在亚喀巴教学生编渔网的村子的教师叫什么？",
    "开罗一座废弃电报局档案室里的某张手写纸条上，注明的发报时间是什么？",
    "请问19世纪末某支球队客场更衣室的门锁品牌是什么？",
    "布宜诺斯艾利斯一个老剧场二楼座席的扶手是用什么木做的？",
    "那位在1920年拍摄初岛灯塔的摄影师的相机型号是什么？",
    "某位19世纪记账员的左手习惯与他的记账错误率有无关联？",
    "西贡旧市集一家香料的标签上写着产自哪个村庄？",
    "那条1920年代横穿某国边境的铁路支线的实际长度是多少里？",
    "一位退休铁路工在1940年代写下的检修手记里，记录了哪座桥的编号？",
    "早年间某座修道院经卷夹层里的那张便签是用什么墨写的？",
    "那位1900年在某某港口卸货的商人留下的账本，封皮是什么颜色？",
    "请问黑白默片时代某跑龙套演员在片场常喝什么茶？",
    "一位乡村医生1923年记的药方贴上，备注了哪种植物在庭院里种植？",
]
# 信息残缺/模糊
_FUZZY = [
    "那家店现在开门了吗？",
    "这个东西最近怎么变成那样了？",
    "你能告诉我在那之后又发生了什么吗？",
    "那个据说的幸运数字到底是多少，好像挺准的？",
    "你能大概判断一下那边下一步会怎么走吗？",
    "听说有个老旧咖啡店老是在下午关张，是真的假的？",
    "那个政要说要推的新制度具体内容你知道吗？",
    "上次那个新闻里提到的具体时间点是几点？",
    "你能说说那个神秘文档里到底写了什么吗？",
    "我听说那个地方的房价变了，具体变了多少？",
    "那个匿名人士最后说的那句话指的是谁？",
    "你能猜一下今年的某件大事会在哪个月定下来吗？",
    "那个传了很久的谣言的源头到底是谁？",
    "你能讲讲那个内部会议讨论的具体细节吗？",
    "那人昨天提到的一个朋友说的一句话，你怎么看？",
    "如果我是那个位置的人，你觉得我下一步该不该动？",
    "那个只出现了一面的角色最后的结局到底怎样？",
    "你能告诉我那个隐藏任务的官方满分标准吗？",
    "那个所谓的内幕消息卖多少钱一条？",
    "听说有个机会特别赚钱，你觉得我该不该抓紧？",
]


def build_neg():
    negs = []
    for q in _FICT + _COLD + _FUZZY:
        idx = len(negs) % len(_NAs)
        negs.append({"instruction": q, "input": "", "output": _NAs[idx]})
    return negs


def main():
    pos = []
    for fn, bud in BUDGET.items():
        p = RAW / fn
        if not p.exists():
            print("MISS", p)
            continue
        n = load_pos(p, bud)
        pos.extend(n)
        print(f"{fn}: +{len(n)}  (累计 {len(pos)})")

    negs = build_neg()
    print(f"负边界样本(内置手写): +{len(negs)}")
    all_ = pos + negs

    # 去重(instruction)
    seen = {}
    for it in all_:
        k = it["instruction"]
        if k not in seen or len(it["output"]) > len(seen[k]["output"]):
            seen[k] = it
    final = list(seen.values())

    OUT.write_text(
        "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in final),
        encoding="utf-8",
    )
    npos = sum(1 for it in final if it["output"].startswith(("该信息我", "关于这个问题", "这一情况我", "我不掌握")))
    print(f"写入 {OUT}  共 {len(final)} 条 | 负边界 {npos} ({npos/max(len(final),1):.0%})")


if __name__ == "__main__":
    main()