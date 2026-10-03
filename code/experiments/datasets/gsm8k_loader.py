"""
GSM8K 50 题固定子集 Loader (P1.4)

策略：
- 用固定 50 题（gsm8k_50_subset.jsonl），保证可复现
- 包装成 NCNLP 任务格式（domain=math）
- 提取最终数字答案作为 expected_output.final_state.answer
- 兼容 HF datasets 库（如果未来要扩到全量）

数据来源：GSM8K test 集前 50 题（Cobbe et al. 2021, MIT License）
"""
import json
import re
import os
from pathlib import Path
from typing import List, Dict, Optional

# 默认子集文件路径 (优先使用改写版防数据污染)
DEFAULT_SUBSET_PATH = Path(__file__).parent / "gsm8k_50_rewritten.jsonl"
# 如果改写版不存在，回退到原始版本
if not DEFAULT_SUBSET_PATH.exists():
    DEFAULT_SUBSET_PATH = Path(__file__).parent / "gsm8k_50_subset.jsonl"


def parse_gsm8k_answer(answer_text: str) -> Optional[float]:
    """
    从 GSM8K 的 answer 文本中提取最终数字。
    GSM8K 格式：推理步骤...#### <数字>
    """
    # 优先匹配 #### 后面的数字
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if m:
        return float(m.group(1).replace(",", ""))
    # 退化匹配：最后一个数字
    nums = re.findall(r"-?[\d,]+\.?\d*", answer_text)
    if nums:
        return float(nums[-1].replace(",", ""))
    return None


def gsm8k_to_ncnlp_task(item: Dict, idx: int) -> Dict:
    """把单条 GSM8K 题包装成 NCNLP 任务格式。"""
    question = item["question"].strip()
    answer_text = item["answer"].strip()
    final_answer = parse_gsm8k_answer(answer_text)

    if final_answer is None:
        raise ValueError(f"无法从 GSM8K 答案中提取数字: {answer_text[:100]}")

    return {
        "id": f"gsm8k_{idx:03d}",
        "domain": "math",
        "difficulty": "medium",
        "raw_prompt": question,
        "input_data": {"problem_text": question},
        "expected_output": {
            "actions": [{"type": "compute", "op": "arithmetic"}],
            "thresholds": {},
            "final_state": {"answer": final_answer}
        },
        "edge_cases": [],
        "source": "gsm8k",
        "notes": f"GSM8K 原题，答案 {final_answer}。用途：PAL baseline 主战场 + 验证 RQ2 正确率"
    }


def load_gsm8k_subset(
    subset_path: Optional[str] = None,
    n: int = 50
) -> List[Dict]:
    """
    加载 GSM8K 固定子集，包装成 NCNLP 任务格式。

    Args:
        subset_path: 子集文件路径（JSONL，每行 {question, answer}），None 用默认
        n: 加载前 N 题，默认 50

    Returns:
        List[Dict] NCNLP 任务列表
    """
    if subset_path is None:
        subset_path = str(DEFAULT_SUBSET_PATH)

    subset_path = Path(subset_path)
    if not subset_path.exists():
        raise FileNotFoundError(
            f"GSM8K 子集文件不存在: {subset_path}\n"
            f"请运行 build_gsm8k_subset.py 生成，或手动放置 50 题 JSONL"
        )

    tasks = []
    skipped = 0
    with open(subset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                tasks.append(gsm8k_to_ncnlp_task(item, i))
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"[SKIP] GSM8K 第 {i} 题: {e}", flush=True)
                skipped += 1

    print(f"[GSM8K LOADER] 加载 {len(tasks)} 题，跳过 {skipped} 题", flush=True)
    return tasks


def build_gsm8k_subset_from_hf(
    output_path: Optional[str] = None,
    n: int = 50,
    split: str = "test"
) -> List[Dict]:
    """
    从 HuggingFace datasets 库下载 GSM8K 并写入 JSONL。
    首次运行需要: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("需要安装 datasets: pip install datasets")

    if output_path is None:
        output_path = str(DEFAULT_SUBSET_PATH)

    print(f"[GSM8K LOADER] 从 HF 下载 GSM8K {split} 集...", flush=True)
    ds = load_dataset("gsm8k", "main", split=split)

    items = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        items.append({"question": row["question"], "answer": row["answer"]})
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{n}", flush=True)

    # 写入 JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[GSM8K LOADER] 已写入 {n} 题到 {output_path}", flush=True)
    return items


# ============================================================================
# 命令行入口
# ============================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        # python gsm8k_loader.py build [n]
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        build_gsm8k_subset_from_hf(n=n)
    else:
        # 默认：加载 50 题
        tasks = load_gsm8k_subset()
        print(f"\n===== GSM8K 子集统计 =====")
        print(f"总数: {len(tasks)}")
        print(f"样例 1: id={tasks[0]['id']}, answer={tasks[0]['expected_output']['final_state']['answer']}")
        print(f"样例 raw_prompt: {tasks[0]['raw_prompt'][:80]}...")
