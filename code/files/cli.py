"""命令行入口。

用法:
    python -m scene_dsl_converter.cli \\
        --input 场景检测智能体.txt \\
        --output 场景检测智能体.dsl

若想保留抽取的中间 JSON(用于调试/人工微调):
    python -m scene_dsl_converter.cli \\
        --input xxx.txt --output xxx.dsl --trace-json xxx.json

离线模式(跳过 LLM 调用,直接从 JSON 渲染):
    python -m scene_dsl_converter.cli --from-json spec.json --output xxx.dsl
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .converter import SceneDSLConverter
from .extractor import ClaudeExtractor, MockExtractor
from .schema import AgentSpec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="场景检测提示词 → DSL v1.2 转换器",
    )
    p.add_argument("--input", help="输入的中文提示词文档路径")
    p.add_argument("--output", required=True, help="输出 DSL 路径")
    p.add_argument(
        "--trace-json",
        help="同时保存 LLM 抽取得到的中间 JSON,便于人工审查",
    )
    p.add_argument(
        "--from-json",
        help="跳过 LLM,直接从预存的 spec JSON 渲染(用于复测)",
    )
    p.add_argument(
        "--model",
        default="claude-opus-4-7",
        help="使用的 Claude model(默认 claude-opus-4-7)",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API Key(默认读环境变量 ANTHROPIC_API_KEY)",
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="跳过输出合法性校验(调试用,不建议)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # 模式 A: 离线,从已抽取好的 JSON 直接渲染
    if args.from_json:
        spec_path = Path(args.from_json)
        with spec_path.open("r", encoding="utf-8") as f:
            spec = AgentSpec.model_validate_json(f.read()).sort_scenes()
        conv = SceneDSLConverter(
            extractor=MockExtractor(spec_path),  # 占位,不会被调用
            strict=not args.no_validate,
        )
        dsl = conv.convert_from_spec(spec)
        Path(args.output).write_text(dsl, encoding="utf-8")
        print(f"[OK] 渲染完成: {args.output}", file=sys.stderr)
        return 0

    # 模式 B: 在线,完整 LLM 抽取 + 渲染
    if not args.input:
        print("错误: 需要 --input 或 --from-json", file=sys.stderr)
        return 2

    extractor = ClaudeExtractor(api_key=args.api_key, model=args.model)
    conv = SceneDSLConverter(extractor=extractor, strict=not args.no_validate)

    if args.trace_json:
        conv.convert_file_with_trace(args.input, args.output, args.trace_json)
        print(f"[OK] DSL: {args.output}", file=sys.stderr)
        print(f"[OK] 中间 JSON: {args.trace_json}", file=sys.stderr)
    else:
        conv.convert_file(args.input, args.output)
        print(f"[OK] DSL: {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
