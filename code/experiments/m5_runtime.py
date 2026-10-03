"""
M5 运行时执行引擎 (P3.1)
=================
对应论文 M5 模块：编译产物的运行时执行, 配合 SemanticSensor 语义传感点 + M0 校验回环 + 惰性回调 + 异常策略。

设计要点 (论文§3 M5 节):
1. 编译产物 = 固化的代码骨架 (骨架来自 M4, 零 LLM)
2. SemanticSensor = 窄化 LLM 调用: 单点、单意图、强类型输出
3. M0 校验 = 每次 sensor 调用后, 验证输出是否在 schema 允许集合内
4. 惰性回调 = 只有代码骨架执行到 sensor 占位符时才调用 LLM
5. 异常策略 = M0 失败 → raise SensorValidationError, 不静默回退

与现有模块关系:
- M1/M2/M3/M4 输出代码骨架 (compiled_skeleton: Callable)
- M5 负责执行, 不做编译
- 不修改 dsl_v2/executor.py (那是方案 C 的轻量解释器)
"""
from __future__ import annotations

import json
import time
import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# 异常与状态
# ============================================================================

class SensorError(Exception):
    """Sensor 基类异常."""


class SensorValidationError(SensorError):
    """M0 校验失败 — 输出不在 schema 允许集合内."""


class SensorParseError(SensorError):
    """LLM 输出无法解析为预期类型."""


class SensorBudgetExceeded(SensorError):
    """超出 sensor 调用预算 (用于防止死循环/失控)."""


class SensorKind(Enum):
    """传感点类型."""
    CLASSIFICATION = "classification"  # 分类 (输出 enum)
    EXTRACTION = "extraction"          # 信息抽取 (输出 dict)
    GENERATION = "generation"          # 受限生成 (输出 str, 受 schema 约束)
    SCORING = "scoring"                # 评分 (输出 float 0-1)
    COMPUTE = "compute"                # 计算 (输出 number)
    BOOLEAN = "boolean"                # 是/否 (输出 bool)


# ============================================================================
# M0 校验器 (出论文 M0 = 校验回环)
# ============================================================================

class M0Validator:
    """
    M0 校验器: 对 sensor 输出做强约束.
    实现形式: 类型 + 枚举 + 范围 + 必填字段的纯函数校验.
    """

    @staticmethod
    def enum_validator(allowed: List[Any]) -> Callable[[Any], Tuple[bool, str]]:
        def _v(x: Any) -> Tuple[bool, str]:
            if x in allowed:
                return True, "ok"
            return False, f"value {x!r} not in allowed enum {allowed}"
        return _v

    @staticmethod
    def type_validator(expected_type: type) -> Callable[[Any], Tuple[bool, str]]:
        def _v(x: Any) -> Tuple[bool, str]:
            if isinstance(x, expected_type):
                # bool 是 int 的子类, 这里特判
                if expected_type is int and isinstance(x, bool):
                    return False, f"expected int, got bool"
                return True, "ok"
            return False, f"expected {expected_type.__name__}, got {type(x).__name__}"
        return _v

    @staticmethod
    def range_validator(lo: float = None, hi: float = None) -> Callable[[Any], Tuple[bool, str]]:
        def _v(x: Any) -> Tuple[bool, str]:
            try:
                v = float(x)
            except (TypeError, ValueError):
                return False, f"not a number: {x!r}"
            if lo is not None and v < lo:
                return False, f"{v} < {lo}"
            if hi is not None and v > hi:
                return False, f"{v} > {hi}"
            return True, "ok"
        return _v

    @staticmethod
    def regex_validator(pattern: str) -> Callable[[Any], Tuple[bool, str]]:
        rgx = re.compile(pattern)
        def _v(x: Any) -> Tuple[bool, str]:
            if rgx.search(str(x)):
                return True, "ok"
            return False, f"does not match pattern {pattern!r}"
        return _v

    @staticmethod
    def schema_validator(required_keys: List[str], value_types: Dict[str, type] = None) -> Callable[[Any], Tuple[bool, str]]:
        """校验 dict 包含全部必填 key, 且值类型符合."""
        value_types = value_types or {}
        def _v(x: Any) -> Tuple[bool, str]:
            if not isinstance(x, dict):
                return False, f"expected dict, got {type(x).__name__}"
            for k in required_keys:
                if k not in x:
                    return False, f"missing key {k!r}"
                if k in value_types and not isinstance(x[k], value_types[k]):
                    return False, f"key {k!r}: expected {value_types[k].__name__}, got {type(x[k]).__name__}"
            return True, "ok"
        return _v

    @staticmethod
    def chain(*validators: Callable[[Any], Tuple[bool, str]]) -> Callable[[Any], Tuple[bool, str]]:
        def _v(x: Any) -> Tuple[bool, str]:
            for fn in validators:
                ok, msg = fn(x)
                if not ok:
                    return False, msg
            return True, "ok"
        return _v


# ============================================================================
# SemanticSensor: 窄化 LLM 语义传感点
# ============================================================================

@dataclass
class SensorResult:
    """单次 sensor 调用的结果."""
    value: Any
    raw_output: str
    usage: Optional[Dict[str, int]] = None
    latency_ms: int = 0
    sensor_name: str = ""
    validation_msg: str = "ok"


@dataclass
class SensorSpec:
    """传感点规范: 描述一次窄化调用的完整契约."""
    name: str
    kind: SensorKind
    description: str
    input_vars: List[str] = field(default_factory=list)
    output_type: type = str
    m0_validator: Optional[Callable[[Any], Tuple[bool, str]]] = None
    system_prompt: str = ""
    user_prompt_template: str = ""
    temperature: float = 0.0
    max_retries: int = 2  # 校验失败时重试次数
    allowed_values: Optional[List[Any]] = None  # 仅对 classification


class SemanticSensor:
    """
    语义传感点: 把 LLM 调用窄化为带 schema + M0 校验的函数.

    关键性质:
    1. 单点: 一次调用 = 单一意图
    2. 强类型: 输出类型 + 枚举/范围 schema
    3. M0 必过: 不通过 raise SensorValidationError
    4. 重试: M0 失败时用同一 prompt 重试, 仍失败则 raise
    """

    def __init__(self, spec: SensorSpec, llm_client=None):
        self.spec = spec
        # 注入统一 LLM 客户端
        if llm_client is None:
            from llm_client import create_llm_client
            llm_client = create_llm_client()
        self.client = llm_client

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        """填充 prompt 模板的 {{var}} 占位符."""
        tpl = self.spec.user_prompt_template
        for k, v in context.items():
            tpl = tpl.replace("{{" + k + "}}", str(v))
        return tpl

    def _parse_output(self, text: str) -> Any:
        """根据 sensor 类型解析 LLM 文本输出."""
        kind = self.spec.kind
        text = (text or "").strip()

        if kind == SensorKind.BOOLEAN:
            low = text.lower().strip()
            # 容错: 是/否/true/false/yes/no/1/0
            if low in ("true", "yes", "是", "y", "1", "✓"):
                return True
            if low in ("false", "no", "否", "n", "0", "✗", "×"):
                return False
            # 尝试解析 JSON
            try:
                obj = json.loads(text)
                if isinstance(obj, bool):
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass
            raise SensorParseError(f"无法解析为 bool: {text[:80]!r}")

        if kind == SensorKind.CLASSIFICATION:
            # 优先尝试 JSON
            try:
                obj = json.loads(text)
                if isinstance(obj, dict) and "value" in obj:
                    return obj["value"]
                if isinstance(obj, (str, int)):
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass
            # 兜底: 直接返回文本 (假设已经是 enum 值)
            return text

        if kind == SensorKind.COMPUTE:
            # 提取数字
            m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
            if m:
                return float(m.group(0))
            try:
                return float(text)
            except ValueError:
                raise SensorParseError(f"无法解析为数字: {text[:80]!r}")

        if kind == SensorKind.SCORING:
            # 提取 0-1 的浮点数
            m = re.search(r"(\d+(?:\.\d+)?)", text)
            if m:
                v = float(m.group(1))
                if v > 1.0 and v <= 100.0:
                    v = v / 100.0
                return max(0.0, min(1.0, v))
            raise SensorParseError(f"无法解析为分数: {text[:80]!r}")

        if kind == SensorKind.EXTRACTION:
            obj = json.loads(text)  # 必须 JSON
            if not isinstance(obj, dict):
                raise SensorParseError(f"extraction 期望 dict, 实际 {type(obj).__name__}")
            return obj

        if kind == SensorKind.GENERATION:
            return text

        return text

    def sense(self, context: Dict[str, Any]) -> SensorResult:
        """
        执行一次 sensor 调用: 构造 prompt → LLM → 解析 → M0 校验 → 返回.

        Raises:
            SensorParseError: 解析失败
            SensorValidationError: M0 校验失败
        """
        user_prompt = self._build_user_prompt(context)
        last_err: Optional[Exception] = None
        t_start = time.time()
        for attempt in range(self.spec.max_retries + 1):
            try:
                resp = self.client.call(
                    system_prompt=self.spec.system_prompt,
                    user_content=user_prompt,
                    temperature=self.spec.temperature,
                )
                value = self._parse_output(resp.content)
                # M0 校验
                if self.spec.m0_validator is not None:
                    ok, msg = self.spec.m0_validator(value)
                    if not ok:
                        if attempt < self.spec.max_retries:
                            last_err = SensorValidationError(
                                f"[{self.spec.name}] M0 校验失败 (attempt {attempt+1}): {msg}"
                            )
                            continue
                        raise SensorValidationError(
                            f"[{self.spec.name}] M0 校验失败 (重试 {self.spec.max_retries} 次后): {msg}"
                        )
                return SensorResult(
                    value=value,
                    raw_output=resp.content,
                    usage=resp.usage or {},
                    latency_ms=int((time.time() - t_start) * 1000),
                    sensor_name=self.spec.name,
                    validation_msg="ok",
                )
            except SensorParseError as e:
                if attempt < self.spec.max_retries:
                    last_err = e
                    continue
                raise
            except SensorValidationError:
                raise  # 校验错直接抛, 不重试 (M0 守住)
        # 走到这里说明重试用尽
        raise last_err or SensorError("sensor 调用未成功")


# ============================================================================
# CompiledExecutor: 编译产物的执行器
# ============================================================================

@dataclass
class ExecutionTrace:
    """执行轨迹: 每次 sensor 调用的记录."""
    sensor_name: str
    context: Dict[str, Any]
    value: Any
    raw_output: str
    latency_ms: int
    usage: Optional[Dict[str, int]]
    validation_msg: str
    timestamp_ms: int


@dataclass
class ExecutionResult:
    """一次执行的完整结果."""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    final_state: Dict[str, Any] = field(default_factory=dict)
    traces: List[ExecutionTrace] = field(default_factory=list)
    total_tokens: int = 0
    sensor_call_count: int = 0
    total_latency_ms: int = 0
    ok: bool = True
    error: str = ""


class CompiledExecutor:
    """
    编译产物执行器.

    用法:
        executor = CompiledExecutor(llm_client)
        executor.register_sensor("classify_intent", sensor1)
        result = executor.run(skeleton_fn, input_data)

    skeleton_fn 签名: (sensors: Dict[str, SemanticSensor], input_data: Dict) -> Tuple[List[Dict], Dict]
    """

    def __init__(self, llm_client=None, sensor_budget: int = 50):
        self.sensors: Dict[str, SemanticSensor] = {}
        self.llm_client = llm_client
        self.sensor_budget = sensor_budget

    def register_sensor(self, name: str, sensor: SemanticSensor):
        self.sensors[name] = sensor

    def _sense(self, name: str, context: Dict[str, Any], trace: List[ExecutionTrace],
               tokens_box: List[int]) -> Any:
        """
        惰性回调: 通过这里调 sensor, 记 trace + 统计 token.
        """
        if name not in self.sensors:
            raise SensorError(f"未注册的 sensor: {name}")
        sensor = self.sensors[name]
        t0 = time.time()
        res = sensor.sense(context)
        # 累计 token
        if res.usage:
            tokens_box.append(res.usage.get("total_tokens", 0))
        # 记 trace
        trace.append(ExecutionTrace(
            sensor_name=name,
            context=dict(context),
            value=res.value,
            raw_output=res.raw_output,
            latency_ms=res.latency_ms,
            usage=res.usage,
            validation_msg=res.validation_msg,
            timestamp_ms=int(t0 * 1000),
        ))
        return res.value

    def run(
        self,
        skeleton: Callable[[Dict[str, SemanticSensor], Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]],
        input_data: Dict[str, Any],
    ) -> ExecutionResult:
        """
        执行编译产物 skeleton.

        Args:
            skeleton: 编译后的函数, 接收 (sensors_dict, input_data) → (actions, final_state)
            input_data: 运行时输入

        Returns:
            ExecutionResult
        """
        t0 = time.time()
        traces: List[ExecutionTrace] = []
        tokens_box: List[int] = []

        # 注入一个 _sense 包装, 让 skeleton 可以惰性调用 sensor
        # 设计: skeleton 通过 sensors_dict[name].sense() 调用, 我们改为用 executor 的 _sense
        # 但为了简洁, 直接把 executor 自身当 sensors_dict 暴露 sense 方法
        sensor_view = _SensorView(self, traces, tokens_box, self.sensor_budget)

        try:
            actions, final_state = skeleton(sensor_view, dict(input_data))
        except SensorError as e:
            return ExecutionResult(
                actions=[],
                final_state={},
                traces=traces,
                ok=False,
                error=f"[SensorError] {e}",
                total_latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:
            return ExecutionResult(
                actions=[],
                final_state={},
                traces=traces,
                ok=False,
                error=f"[{type(e).__name__}] {e}",
                total_latency_ms=int((time.time() - t0) * 1000),
            )

        return ExecutionResult(
            actions=actions or [],
            final_state=final_state or {},
            traces=traces,
            total_tokens=sum(tokens_box),
            sensor_call_count=len(traces),
            total_latency_ms=int((time.time() - t0) * 1000),
            ok=True,
        )


class _SensorView:
    """
    skeleton 视角的 sensor 字典: sensor_view[name] → 返回一个可调用的 thunk.
    每次 skeleton 取值时才真正调 LLM (惰性).
    """

    def __init__(self, executor: CompiledExecutor, traces: List[ExecutionTrace],
                 tokens_box: List[int], budget: int):
        self._exec = executor
        self._traces = traces
        self._tokens = tokens_box
        self._budget = budget
        self._calls = 0

    def __getitem__(self, name: str) -> "_SensorCallable":
        return _SensorCallable(name, self)

    def _consume(self, name: str, context: Dict[str, Any]) -> Any:
        self._calls += 1
        if self._calls > self._budget:
            raise SensorBudgetExceeded(f"sensor 调用次数 {self._calls} 超过预算 {self._budget}")
        return self._exec._sense(name, context, self._traces, self._tokens)


class _SensorCallable:
    """sensor 调用壳: skeleton 写成 sensor['name'](ctx) 形式."""

    def __init__(self, name: str, view: _SensorView):
        self.name = name
        self._view = view

    def __call__(self, context: Dict[str, Any]) -> Any:
        return self._view._consume(self.name, context)


# ============================================================================
# 辅助: 内置的常用 Sensor 工厂
# ============================================================================

def make_classification_sensor(
    name: str,
    description: str,
    allowed_values: List[Any],
    question: str,
    temperature: float = 0.0,
    llm_client=None,
) -> SemanticSensor:
    """快速构造一个 classification 类型的 sensor."""
    spec = SensorSpec(
        name=name,
        kind=SensorKind.CLASSIFICATION,
        description=description,
        input_vars=["context"],
        output_type=str,
        m0_validator=M0Validator.enum_validator(allowed_values),
        system_prompt=(
            "你是一个意图分类器。根据用户提供的业务规则上下文, "
            f"从 {allowed_values} 中选一个最匹配的类别作为答案。\n"
            "严格只输出 JSON: {\"value\": <one_of_allowed_values>}"
        ),
        user_prompt_template=(
            f"## 业务上下文\n{{context}}\n\n"
            f"## 问题\n{question}\n\n"
            "请只输出 JSON, 不要其他文字。"
        ),
        temperature=temperature,
        allowed_values=allowed_values,
    )
    return SemanticSensor(spec, llm_client=llm_client)


def make_extraction_sensor(
    name: str,
    description: str,
    required_keys: List[str],
    value_types: Dict[str, type] = None,
    question: str = "从上下文抽取关键字段:",
    llm_client=None,
) -> SemanticSensor:
    """快速构造一个 extraction 类型的 sensor."""
    spec = SensorSpec(
        name=name,
        kind=SensorKind.EXTRACTION,
        description=description,
        input_vars=["context"],
        output_type=dict,
        m0_validator=M0Validator.schema_validator(required_keys, value_types),
        system_prompt=(
            "你是一个信息抽取器。从提供的业务上下文抽取结构化字段, "
            "以严格 JSON 格式输出。\n"
            "格式: {\"key1\": <value1>, \"key2\": <value2>, ...}\n"
            "只输出 JSON, 不输出其他文字。"
        ),
        user_prompt_template=(
            f"## 业务上下文\n{{context}}\n\n"
            f"## 任务\n{question}\n"
            f"必填字段: {required_keys}\n"
            "请输出 JSON。"
        ),
        temperature=0.0,
    )
    return SemanticSensor(spec, llm_client=llm_client)


def make_boolean_sensor(
    name: str,
    description: str,
    question: str,
    llm_client=None,
) -> SemanticSensor:
    """快速构造一个 boolean 类型的 sensor."""
    spec = SensorSpec(
        name=name,
        kind=SensorKind.BOOLEAN,
        description=description,
        output_type=bool,
        system_prompt=(
            "你是一个判定器。根据业务上下文, 对问题给出 是/否 的判断。\n"
            "严格只输出 JSON: {\"value\": true|false}"
        ),
        user_prompt_template=(
            f"## 业务上下文\n{{context}}\n\n"
            f"## 问题\n{question}\n\n"
            "只输出 JSON。"
        ),
        temperature=0.0,
    )
    return SemanticSensor(spec, llm_client=llm_client)


# ============================================================================
# 便捷: 编译 + 跑一个 NCNLP 任务 (用于 pipeline_runner.py)
# ============================================================================

def execute_task(
    task: Dict[str, Any],
    skeleton: Callable,
    sensor_specs: List[SensorSpec] = None,
    llm_client=None,
) -> ExecutionResult:
    """
    一站式: 构造 executor, 注册 sensors (如未提供), 跑 task.

    Args:
        task: NCNLP 任务 dict (id, raw_prompt, input_data, expected_output, ...)
        skeleton: 编译后的代码骨架函数
        sensor_specs: 该 task 用到的 sensor 列表
        llm_client: 复用的 LLM 客户端
    """
    executor = CompiledExecutor(llm_client=llm_client)
    if sensor_specs:
        for spec in sensor_specs:
            executor.register_sensor(spec.name, SemanticSensor(spec, llm_client=llm_client))
    return executor.run(skeleton, task.get("input_data", {}))


# ============================================================================
# 简单自测
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # 演示: 一个简单的 gov_001 风格任务
    def skeleton(sensors, input_data):
        """
        编译产物: 代码骨架 (对应论文 running example 政务经费版)
        本骨架零 LLM, 仅在条件判断需要 LLM 介入时调 sensor
        """
        actions = []
        budget = input_data.get("本月经费", 0)
        threshold = input_data.get("threshold", 50000)

        # 条件判断: 用固化代码, 0 LLM
        if budget > threshold:
            overflow = budget - threshold
            # 受控的语义传感点: 收件人/主题/正文(可由 sensor 决定, 也可由 input 给)
            # 这里用 input 已给的固定字段, 不需调 sensor
            actions.append({
                "type": "transfer",
                "amount": overflow,
                "to_account": "应急储备账户",
            })
            actions.append({
                "type": "email",
                "to": "财务负责人",
                "subject": "应急储备通报",
                "body_contains": [str(overflow), "应急储备"],
            })
        else:
            actions.append({"type": "default_plan"})

        final_state = {
            "overflow": (budget - threshold) if budget > threshold else 0,
            "plan_executed": "default" if budget <= threshold else "transfer+email",
        }
        return actions, final_state

    # 跑一个测试
    test_input = {"本月经费": 58000, "threshold": 50000}
    executor = CompiledExecutor()
    result = executor.run(skeleton, test_input)
    print(f"\n=== M5 自测 (零 sensor 情况) ===")
    print(f"actions: {json.dumps(result.actions, ensure_ascii=False, indent=2)}")
    print(f"final_state: {result.final_state}")
    print(f"sensor_calls: {result.sensor_call_count}")
    print(f"tokens: {result.total_tokens}")
    print(f"ok: {result.ok}")

    # 再跑一个真用 sensor 的场景
    spec = SensorSpec(
        name="classify_refund",
        kind=SensorKind.CLASSIFICATION,
        description="判定是否符合退款条件",
        input_vars=["context"],
        output_type=str,
        m0_validator=M0Validator.enum_validator(["APPROVE", "REJECT", "ESCALATE"]),
        system_prompt="你是退款审核助手。根据业务上下文, 给出审批结果 (APPROVE/REJECT/ESCALATE)。",
        user_prompt_template="## 上下文\n{{context}}\n\n请只输出 JSON: {\"value\": <APPROVE|REJECT|ESCALATE>}",
        temperature=0.0,
    )
    sensor = SemanticSensor(spec)

    def skeleton2(sensors, input_data):
        decision = sensors["classify_refund"]({"context": json.dumps(input_data, ensure_ascii=False)})
        if decision == "APPROVE":
            actions = [{"type": "refund", "amount": input_data.get("amount", 0)}]
        elif decision == "REJECT":
            actions = [{"type": "notify", "channel": "email", "reason": "不符合退款条件"}]
        else:
            actions = [{"type": "escalate", "to": "主管"}]
        return actions, {"decision": decision}

    executor2 = CompiledExecutor()
    executor2.register_sensor("classify_refund", sensor)
    test_input2 = {"amount": 350, "order_age_days": 2, "vip_level": "gold"}
    result2 = executor2.run(skeleton2, test_input2)
    print(f"\n=== M5 自测 (含 1 个 classification sensor) ===")
    print(f"actions: {json.dumps(result2.actions, ensure_ascii=False, indent=2)}")
    print(f"final_state: {result2.final_state}")
    print(f"sensor_calls: {result2.sensor_call_count}")
    print(f"tokens: {result2.total_tokens}")
    print(f"traces:")
    for t in result2.traces:
        print(f"  - {t.sensor_name}: {t.value} (latency={t.latency_ms}ms, validation={t.validation_msg})")
