"""
B0 - 直接执行 Baseline
=================
论文中路线一"概率引擎内加厚"的朴素代表。
整段提示词给 LLM, temperature=0, 1 次调用, 要求输出 JSON 格式动作列表。

代表论文: Brown et al. GPT-3 (NeurIPS 2020) - 整段 few-shot 范式
"""
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

# 接入系统统一 LLM 客户端
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_client import create_llm_client


SYSTEM_PROMPT = """你是一个严格的规则执行助手。给定一段业务规则描述和输入数据,你需要严格按照规则执行,并以 JSON 格式输出动作列表。

## 输出格式

```json
{
  "actions": [
    {"type": "<动作类型>", "<其他字段>": "<值>"}
  ],
  "final_state": {"<字段>": "<值>"}
}
```

## 要求
1. 严格按规则条件判断,不要自行添加规则
2. 数值计算要准确
3. 输出必须是合法 JSON,不要包含任何额外解释
4. 当规则有多个分支时,选择当前输入对应的那一条
"""


def build_user_prompt(task: Dict) -> str:
    """构造 user prompt, 把 raw_prompt + input_data 合并."""
    raw = task["raw_prompt"]
    data = task.get("input_data", {})
    data_str = json.dumps(data, ensure_ascii=False, indent=2)
    return f"## 业务规则\n{raw}\n\n## 输入数据\n{data_str}\n\n请按规则执行,以 JSON 格式输出动作列表。"


def parse_json_output(text: str) -> Optional[Dict]:
    """从 LLM 输出中提取 JSON. 容忍 markdown 包裹, 兼容数组和对象格式."""
    text = (text or "").strip()

    # 尝试直接解析
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return {"actions": result}
        return result
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 块 (兼容数组 [] 和对象 {})
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            result = json.loads(m.group(1))
            if isinstance(result, list):
                return {"actions": result}
            return result
        except json.JSONDecodeError:
            pass

    # 尝试提取 ``` ... ``` 块 (无语言标注)
    m = re.search(r"```\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            result = json.loads(m.group(1))
            if isinstance(result, list):
                return {"actions": result}
            return result
        except json.JSONDecodeError:
            pass

    # 尝试提取第一个完整的 JSON 对象 (非贪婪匹配, 避免抓取推理文字)
    # 使用括号平衡来找到最外层的 { ... }
    result = _extract_balanced_json(text, '{', '}')
    if result is not None:
        return result

    # 尝试提取第一个完整的 JSON 数组
    result = _extract_balanced_json(text, '[', ']')
    if result is not None:
        return result

    return None


def _extract_balanced_json(text: str, open_char: str, close_char: str) -> Optional[Dict]:
    """用括号平衡法从文本中提取最外层完整的 JSON 块 (避免贪婪匹配抓取推理文字)."""
    start = text.find(open_char)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, list):
                        return {"actions": result}
                    return result
                except json.JSONDecodeError:
                    # 这个平衡块不是有效 JSON, 继续找下一个 open_char
                    return _extract_balanced_json(text[i+1:], open_char, close_char)
    return None


def extract_actions_with_fallback(parsed: Optional[Dict]) -> List[Dict]:
    """从 LLM 解析结果中提取 actions, 含 final_state 兜底 (数学题答案提取).

    逻辑:
    1. 先取 parsed["actions"]
    2. 如果没有 compute 类 action, 尝试从 final_state 提取答案
    3. 这是 B0/B2/B4 共用的数学题兜底逻辑
    """
    if parsed is None:
        return []
    actions = parsed.get("actions", []) or []
    fs = parsed.get("final_state", {}) or {}
    # final_state 兜底: 只在 actions 为空时才提取 (避免业务题误加 compute)
    if not actions and isinstance(fs, dict):
        priority_keys = ("answer", "Answer", "Total", "total", "result", "Result",
                         "天数", "结果", "答案", "总数", "sum", "days_needed", "days")
        chosen = None
        for k in priority_keys:
            if k in fs and isinstance(fs[k], (int, float)) and not isinstance(fs[k], bool):
                chosen = (k, fs[k])
                break
        if chosen is None:
            nums = [(k, v) for k, v in fs.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if nums:
                chosen = max(nums, key=lambda x: x[1])
        if chosen is not None:
            actions = list(actions) + [{
                "type": "compute",
                "op": "arithmetic",
                "answer": chosen[1],
                "_source": f"final_state.{chosen[0]}",
            }]
    return actions


class B0Direct:
    """B0 直接执行 baseline. 1 次 LLM 调用, temperature=0."""

    def __init__(self, llm_client=None):
        self.client = llm_client or create_llm_client()
        self.name = "B0"

    def run(self, task: Dict, n_runs: int = 1) -> Dict:
        """
        运行 baseline, 返回统一格式结果.

        Args:
            task: NCNLP 任务 dict
            n_runs: 重复运行次数 (用于 RQ1 一致性测试)

        Returns:
            {
              "baseline": "B0",
              "task_id": str,
              "n_runs": int,
              "actions_per_run": List[List[Dict]],  # 每次的 actions
              "tokens_per_run": List[Dict],  # 每次的 token 用量
              "raw_outputs": List[str],  # 每次的原始输出
              "final_actions": List[Dict]  # 最终采用的动作 (第 1 次)
            }
        """
        user_prompt = build_user_prompt(task)
        actions_per_run = []
        tokens_per_run = []
        raw_outputs = []

        for i in range(n_runs):
            try:
                resp = self.client.call(
                    system_prompt=SYSTEM_PROMPT,
                    user_content=user_prompt,
                    temperature=0,  # B0 关键: 温度为 0
                )
                parsed = parse_json_output(resp.content)
                if parsed is None:
                    actions = []
                else:
                    actions = extract_actions_with_fallback(parsed)
                raw_outputs.append(resp.content)
                actions_per_run.append(actions)
                tokens_per_run.append(resp.usage or {})
            except Exception as e:
                raw_outputs.append(f"[ERROR] {e}")
                actions_per_run.append([])
                tokens_per_run.append({})

        return {
            "baseline": self.name,
            "task_id": task["id"],
            "n_runs": n_runs,
            "actions_per_run": actions_per_run,
            "tokens_per_run": tokens_per_run,
            "raw_outputs": raw_outputs,
            "final_actions": actions_per_run[0] if actions_per_run else [],
        }


if __name__ == "__main__":
    # 简单测试: 跑 gov_001
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from load_all_tasks import load_all

    tasks = load_all()
    gov = next(t for t in tasks if t["id"] == "gov_001")

    print("=" * 60)
    print(f"测试 B0 on {gov['id']}: {gov['raw_prompt'][:50]}...")
    print("=" * 60)

    b0 = B0Direct()
    result = b0.run(gov, n_runs=3)

    print(f"\n最终动作: {json.dumps(result['final_actions'], ensure_ascii=False, indent=2)}")
    print(f"3 次运行的 actions 集合: {[set(map(lambda a: a.get('type'), run)) for run in result['actions_per_run']]}")
    print(f"3 次 tokens: {[t.get('total_tokens', 0) for t in result['tokens_per_run']]}")
