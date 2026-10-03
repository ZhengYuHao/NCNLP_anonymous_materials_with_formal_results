"""
NCNLP 任务集加载器（P1.1）
- 递归加载 task_set/ 下所有 JSON
- 用 _schema.json 校验
- 提供按领域/难度筛选接口
- 提供统计接口

设计原则：
- 任何 baseline（B0/B1/B2/PAL/DSPy）和 NCNLP 都用同一个 loader，避免评测口径不一致
- 校验失败要给出具体出错位置（不要只说 'invalid'）
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

# 兼容无 jsonschema 库的情况（环境受限时优雅降级）
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    print("[WARN] jsonschema 未安装，仅做字段必填检查（强烈建议 pip install jsonschema）", file=sys.stderr)


SCHEMA_PATH = Path(__file__).parent / "_schema.json"

# 必填字段（即使没有 jsonschema 也要校验）
REQUIRED_FIELDS = ["id", "domain", "difficulty", "raw_prompt", "input_data", "expected_output", "edge_cases"]
VALID_DOMAINS = {"governance", "finance", "medical", "industrial", "customer_service", "general", "math"}
VALID_DIFFICULTY = {"simple", "medium", "complex"}
ID_PATTERN = lambda x: (
    x.startswith(("gov_", "fin_", "med_", "ind_", "cus_", "gen_", "gsm8k_")) and
    x.split("_", 1)[1].isdigit()
)


def _load_schema() -> Optional[Dict]:
    if not SCHEMA_PATH.exists():
        return None
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_schema(task: Dict) -> Tuple[bool, str]:
    """
    校验单条任务是否符合 Schema。
    返回 (is_valid, error_msg)。
    """
    # 1) 必填字段检查
    missing = [f for f in REQUIRED_FIELDS if f not in task]
    if missing:
        return False, f"缺少必填字段: {missing}"

    # 2) 字段值枚举检查
    if task["domain"] not in VALID_DOMAINS:
        return False, f"domain 非法: {task['domain']}，应属于 {VALID_DOMAINS}"
    if task["difficulty"] not in VALID_DIFFICULTY:
        return False, f"difficulty 非法: {task['difficulty']}，应属于 {VALID_DIFFICULTY}"
    if not ID_PATTERN(task["id"]):
        return False, f"id 格式非法: {task['id']}，应匹配 ^xxx_NNN$"
    if not isinstance(task["input_data"], dict):
        return False, f"input_data 必须是 dict，实际是 {type(task['input_data']).__name__}"
    if not isinstance(task["expected_output"], dict) or "actions" not in task["expected_output"]:
        return False, "expected_output 必须是 dict 且含 actions 字段"
    if not isinstance(task["edge_cases"], list) or len(task["edge_cases"]) < 1:
        return False, "edge_cases 必须是非空 list"
    if not isinstance(task["raw_prompt"], str) or len(task["raw_prompt"]) < 20:
        return False, f"raw_prompt 太短（{len(task.get('raw_prompt',''))} 字符），至少 20"

    # 3) 若有 jsonschema，用它做完整校验
    if HAS_JSONSCHEMA:
        schema = _load_schema()
        if schema:
            try:
                jsonschema.validate(instance=task, schema=schema)
            except jsonschema.ValidationError as e:
                return False, f"jsonschema 校验失败: {e.message} (path: {list(e.path)})"

    return True, ""


def load_all_tasks(task_set_dir: Optional[str] = None) -> List[Dict]:
    """
    递归加载 task_set 下所有 JSON 文件。
    校验失败的任务会被跳过并打印警告。
    """
    if task_set_dir is None:
        task_set_dir = str(Path(__file__).parent)
    task_set_dir = Path(task_set_dir)

    tasks = []
    skipped = 0
    for json_path in sorted(task_set_dir.rglob("*.json")):
        # 跳过 schema 文件本身和隐藏文件
        if json_path.name.startswith("_") or json_path.name.startswith("."):
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                task = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[SKIP] {json_path.name}: JSON 解析失败 - {e}", file=sys.stderr)
            skipped += 1
            continue

        ok, err = verify_schema(task)
        if not ok:
            print(f"[SKIP] {json_path.name}: {err}", file=sys.stderr)
            skipped += 1
            continue

        tasks.append(task)

    print(f"[LOADER] 加载 {len(tasks)} 条任务，跳过 {skipped} 条不合格", file=sys.stderr)
    return tasks


def filter_by_domain(tasks: List[Dict], domain: str) -> List[Dict]:
    return [t for t in tasks if t["domain"] == domain]


def filter_by_difficulty(tasks: List[Dict], level: str) -> List[Dict]:
    return [t for t in tasks if t["difficulty"] == level]


def filter_by_source(tasks: List[Dict], source: str) -> List[Dict]:
    return [t for t in tasks if t.get("source", "self-constructed") == source]


def get_stats(tasks: List[Dict]) -> Dict:
    """统计各 domain/difficulty 分布，便于 sanity check。"""
    return {
        "total": len(tasks),
        "by_domain": dict(Counter(t["domain"] for t in tasks)),
        "by_difficulty": dict(Counter(t["difficulty"] for t in tasks)),
        "by_source": dict(Counter(t.get("source", "self-constructed") for t in tasks)),
    }


# ============================================================================
# 命令行快速验证
# ============================================================================
if __name__ == "__main__":
    tasks = load_all_tasks()
    stats = get_stats(tasks)
    print("\n===== 任务集统计 =====")
    print(f"总数: {stats['total']}")
    print(f"按领域: {stats['by_domain']}")
    print(f"按难度: {stats['by_difficulty']}")
    print(f"按来源: {stats['by_source']}")
