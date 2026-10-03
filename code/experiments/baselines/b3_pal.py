"""
B3 - PAL (Program-aided Language Models) Baseline
=================
Gao et al. ICML 2023 标准做法.
让 LLM 生成 Python 代码, 用 Python 解释器执行, 捕获 print 输出作为最终答案.

代表论文: Gao et al. "PAL: Program-aided Language Models" ICML 2023

忠实于原 paper 的实现细节:
- 原 paper prompt 模板: "Q: <问题>\nA: Let's think step by step.\n```python\n<code>\n```"
- 关键设计: LLM 在写代码前先 "Let's think step by step" 引导推理
- 代码末尾 `print(answer)`, 解释器 stdout 即为最终答案
- 安全沙箱: 限制危险 import, 5s 超时

对业务规则任务的扩展 (本 paper 创新):
- 原 paper 仅针对数学/逻辑推理
- 本实现保留原 prompt 风格, 允许 LLM 写 Python 代码表达业务规则
- 代码可 print JSON 包含 actions 列表, 供 judge 评估
"""
import sys
import re
import json
import multiprocessing
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # 项目根目录
sys.path.insert(0, str(Path(__file__).parent))  # baselines 目录
from llm_client import create_llm_client


SYSTEM_PROMPT = """You are a helpful assistant that solves problems by writing Python code. Follow the PAL format: state reasoning then provide executable code.

Format:
Q: <question>
A: Let's think step by step.
```python
<code that prints the final answer>
```

Rules:
1. Use only standard library modules (math, decimal, json, re, datetime, time, collections)
2. The last line of your code must be a `print(...)` that outputs the final answer
3. For math problems: print the numeric answer directly (e.g., `print(answer)`)
4. For business rules: print a JSON object where `actions` is a list of dicts with a `type` field

Example for math problem:
Q: Janet's ducks lay 16 eggs per day. She eats 3 and bakes 4. She sells the rest at $2 each. How much does she make?
A: Let's think step by step.
```python
total_eggs = 16
eaten_eggs = 3
baked_eggs = 4
sold_eggs = total_eggs - eaten_eggs - baked_eggs
dollars_per_egg = 2
answer = sold_eggs * dollars_per_egg
print(answer)
```

Example for business rule:
Q: If monthly budget exceeds 50000, transfer the excess to reserve account and email the finance manager.
A: Let's think step by step.
```python
import json
monthly_budget = 58000
threshold = 50000
actions = []
if monthly_budget > threshold:
    excess = monthly_budget - threshold
    actions.append({"type": "transfer", "amount": excess, "to_account": "reserve"})
    actions.append({"type": "email", "to": "finance_manager", "subject": "Budget exceeded"})
print(json.dumps({"actions": actions}, ensure_ascii=False))
```

Important:
- Code must be complete and runnable, no placeholders
- Don't import os, sys, subprocess, or any network/file modules
"""


def build_user_prompt(task: dict) -> str:
    """
    构造 user prompt, 忠实于 PAL 原 paper 的 Q/A 格式.
    原 paper 模板: "Q: <问题>\\nA: ..."
    """
    raw = task["raw_prompt"]
    data = task.get("input_data", {})
    data_str = json.dumps(data, ensure_ascii=False, indent=2)
    return (
        f"Q: {raw}\n"
        f"Input data: {data_str}\n\n"
        f"A: Let's think step by step."
    )


def extract_python_code(text: str) -> Optional[str]:
    """从 LLM 输出中提取 Python 代码块."""
    m = re.search(r"```python\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text  # 兜底: 当成代码直接执行


# Python 沙箱 - 限制危险模块
ALLOWED_BUILTINS = {
    'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'divmod',
    'enumerate', 'filter', 'float', 'format', 'hex', 'int', 'isinstance',
    'issubclass', 'iter', 'len', 'list', 'map', 'max', 'min', 'next',
    'object', 'oct', 'ord', 'pow', 'print', 'range', 'repr', 'reversed',
    'round', 'set', 'slice', 'sorted', 'str', 'sum', 'tuple', 'type',
    'zip', 'True', 'False', 'None',
}

# 允许 import 的白名单模块
ALLOWED_MODULES = {'json', 'math', 'decimal', 're', 'datetime', 'time', 'collections'}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """受限 import: 只允许白名单模块."""
    if name in ALLOWED_MODULES:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"模块 {name!r} 禁止 import (PAL 沙箱白名单)")


def _safe_exec_worker(code: str, result_queue) -> None:
    """Child-process worker used by ``safe_exec`` on every supported OS."""
    import io
    import contextlib

    # 受限 globals: 只暴露白名单 builtins, 重写 __import__
    builtins_dict = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
    safe_builtins = {k: builtins_dict[k] for k in ALLOWED_BUILTINS if k in builtins_dict}
    # 额外移除危险函数 (防止 Python exec 自动补回)
    for dangerous in ('exec', 'eval', 'compile', 'open', 'globals', 'locals',
                      'vars', 'dir', 'getattr', 'setattr', 'delattr', 'breakpoint'):
        safe_builtins.pop(dangerous, None)
    # 关键: 把 _safe_import 放进 builtins, 这样 import 语句能找到白名单模块
    safe_builtins['__import__'] = _safe_import
    safe_globals = {
        "__builtins__": safe_builtins,
    }
    safe_locals = {}

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, safe_globals, safe_locals)

        output = stdout.getvalue().strip()
        if not output:
            result_queue.put({"success": True, "result": None, "stdout": output})
            return
        # 尝试从 stdout 逐行倒序查找有效 JSON
        lines = output.split("\n")
        result_obj = None
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                result_obj = json.loads(line)
                break
            except json.JSONDecodeError:
                # 可能是纯数字或纯字符串
                try:
                    result_obj = float(line) if '.' in line else int(line)
                    break
                except (ValueError, TypeError):
                    continue
        if result_obj is not None:
            result_queue.put({"success": True, "result": result_obj, "stdout": output})
            return
        # 所有行都不行, 整体试一下
        try:
            result_obj = json.loads(output)
            result_queue.put({"success": True, "result": result_obj, "stdout": output})
        except json.JSONDecodeError:
            result_queue.put({"success": True, "result": None, "stdout": output})
    except Exception as e:
        result_queue.put({"success": False, "error": str(e), "stdout": stdout.getvalue()})


def safe_exec(code: str, timeout_sec: int = 5) -> Dict[str, Any]:
    """Execute generated code in a restricted, time-bounded child process."""
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_safe_exec_worker, args=(code, result_queue))
    process.start()
    process.join(timeout_sec)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"success": False, "error": "代码执行超时", "stdout": ""}
    try:
        return result_queue.get(timeout=1)
    except queue.Empty:
        return {
            "success": False,
            "error": f"PAL 沙箱子进程异常退出 (exitcode={process.exitcode})",
            "stdout": "",
        }


class B3PAL:
    """B3 PAL: LLM 产 Python 代码, 沙箱执行."""

    def __init__(self, llm_client=None, timeout_sec: int = 5):
        self.client = llm_client or create_llm_client()
        self.name = "B3"
        self.timeout = timeout_sec

    def run(self, task: dict, n_runs: int = 1) -> dict:
        user_prompt = build_user_prompt(task)
        actions_per_run = []
        tokens_per_run = []
        raw_outputs = []

        for i in range(n_runs):
            try:
                resp = self.client.call(
                    system_prompt=SYSTEM_PROMPT,
                    user_content=user_prompt,
                    temperature=0,
                )
                code = extract_python_code(resp.content)
                exec_result = safe_exec(code, timeout_sec=self.timeout)

                if exec_result["success"] and exec_result["result"] is not None:
                    result_obj = exec_result["result"]
                    # 兼容多种格式:
                    #   1) {"actions": [...]}                 业务规则
                    #   2) 260 (int) / 3.14 (float)            PAL 原版数学题: print(answer)
                    #   3) "18" (str)                          PAL 原版可能 print 字符串
                    #   4) {"answer": N} / {"Total": N}         兼容旧 prompt
                    #   5) {"days_needed": N}                  计算结果名
                    #   6) {"final_state": {"answer": N}}      嵌套
                    if isinstance(result_obj, (int, float)) and not isinstance(result_obj, bool):
                        # PAL 原版数学题: print(answer) 直接出数字
                        actions = [{"type": "compute", "op": "arithmetic", "answer": result_obj}]
                    elif isinstance(result_obj, str):
                        # print 字符串: 尝试解析为数字
                        try:
                            ans = float(result_obj) if '.' in result_obj else int(result_obj)
                            actions = [{"type": "compute", "op": "arithmetic", "answer": ans}]
                        except (ValueError, TypeError):
                            actions = []
                    elif isinstance(result_obj, dict):
                        if "actions" in result_obj:
                            actions = result_obj["actions"]
                        else:
                            # 找最可能是"答案"的字段
                            ans = None
                            answer_keys = (
                                "answer", "Answer", "ANSWER",
                                "Total", "total", "TOTAL",
                                "结果", "答案",
                                "days_needed", "days", "hours_needed", "minutes_needed",
                                "sum", "Sum", "SUM", "result", "Result", "RESULT",
                                "final_answer", "final_result",
                            )
                            for k in answer_keys:
                                if k in result_obj and isinstance(result_obj[k], (int, float)):
                                    ans = result_obj[k]
                                    break
                            # 兜底: 找 dict 中最大的数字
                            if ans is None:
                                nums = [
                                    (k, v) for k, v in result_obj.items()
                                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                                ]
                                if nums:
                                    priority_kw = ("answer", "total", "need", "sum", "result", "天数", "总数", "结果", "答案", "count")
                                    picked = None
                                    for kw in priority_kw:
                                        for k, v in nums:
                                            if kw.lower() in k.lower():
                                                picked = v
                                                break
                                        if picked is not None:
                                            break
                                    if picked is None:
                                        picked = max(v for _, v in nums)
                                    ans = picked
                            if ans is not None:
                                actions = [{"type": "compute", "op": "arithmetic", "answer": ans}]
                            elif "final_state" in result_obj and isinstance(result_obj["final_state"], dict):
                                fs = result_obj["final_state"]
                                for k, v in fs.items():
                                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                                        actions = [{"type": "compute", "op": "arithmetic", "answer": v}]
                                        break
                                else:
                                    actions = []
                            else:
                                actions = []
                    elif isinstance(result_obj, list):
                        # 直接是 actions 列表
                        actions = result_obj
                    else:
                        actions = []

                raw_outputs.append({
                    "llm_output": resp.content[:500],
                    "code": code[:500],
                    "exec_result": exec_result,
                })
                actions_per_run.append(actions)
                tokens_per_run.append(resp.usage or {})
            except Exception as e:
                raw_outputs.append({"error": str(e)})
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
    sys.path.insert(0, str(Path(__file__).parent.parent))  # 实验目录
    from load_all_tasks import load_all

    tasks = load_all()
    # 先测 gov_001 (自构), 再测一个 gsm8k
    gov = next(t for t in tasks if t["id"] == "gov_001")
    gsm = next(t for t in tasks if t["id"] == "gsm8k_000")

    print("=" * 60)
    print(f"测试 B3 (PAL) on {gov['id']}")
    print("=" * 60)

    b3 = B3PAL()
    r1 = b3.run(gov, n_runs=1)
    print(f"\n[gov_001] 最终动作: {r1['final_actions']}")
    print(f"  tokens: {r1['tokens_per_run'][0]}")
    print(f"  exec 成功: {r1['raw_outputs'][0].get('exec_result', {}).get('success', '?')}")
    print(f"  exec error: {r1['raw_outputs'][0].get('exec_result', {}).get('error', '?')}")
    print(f"  llm_output[:300]: {r1['raw_outputs'][0].get('llm_output', '?')[:300]}")

    print("\n" + "=" * 60)
    print(f"测试 B3 (PAL) on {gsm['id']}: {gsm['raw_prompt'][:50]}...")
    print("=" * 60)
    r2 = b3.run(gsm, n_runs=1)
    print(f"\n[gsm8k_000] 最终动作: {r2['final_actions']}")
    print(f"  期望答案: {gsm['expected_output']['final_state']}")
    print(f"\n  [debug] raw_output: {r2['raw_outputs'][0]}")
