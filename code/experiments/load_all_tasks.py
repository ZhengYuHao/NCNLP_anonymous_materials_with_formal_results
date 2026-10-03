"""
统一任务加载入口：自构 + GSM8K 一起加载
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "task_set"))
sys.path.insert(0, str(Path(__file__).parent / "datasets"))

from task_set.loader import load_all_tasks, get_stats
from datasets.gsm8k_loader import load_gsm8k_subset


def load_all() -> list:
    """加载自构任务集 + GSM8K 子集"""
    self_constructed = load_all_tasks()
    gsm8k = load_gsm8k_subset()
    all_tasks = self_constructed + gsm8k
    print(f"\n===== 全部任务统计 =====")
    print(f"自构: {len(self_constructed)}")
    print(f"GSM8K: {len(gsm8k)}")
    print(f"合计: {len(all_tasks)}")
    print(f"按领域: {get_stats(all_tasks)['by_domain']}")
    return all_tasks


if __name__ == "__main__":
    tasks = load_all()
