"""
DSL v2 命令行入口
使用统一的 llm_client.py 接口

用法:
    python -m dsl_v2 input.txt -o output.dsl
    python -m dsl_v2 input.txt -o output.dsl --max-retries 5
    python -m dsl_v2 input.txt -o output.dsl --mock
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsl_v2 import PromptCompiler


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DSL v2 编译器 - 使用统一 LLM 接口",
    )
    p.add_argument("input", help="输入的自然语言需求文件路径")
    p.add_argument("-o", "--output", required=True, help="输出 DSL 文件路径")
    p.add_argument("--max-retries", type=int, default=3, help="最大重试次数 (默认: 3)")
    p.add_argument("--auto-fix-threshold", type=int, default=3, help="自动修复阈值 (默认: 3)")
    p.add_argument("--mock", action="store_true", help="使用 mock 模式 (跳过 LLM 调用)")
    p.add_argument("--model", default="gpt-4o-mini", help="LLM 模型 (默认: gpt-4o-mini)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}", file=sys.stderr)
        return 1

    input_text = open(args.input, encoding="utf-8").read()

    print(f"[INFO] 开始编译...")
    print(f"[INFO] 输入: {args.input}")
    print(f"[INFO] 输出: {args.output}")
    print(f"[INFO] 最大重试: {args.max_retries}")
    print(f"[INFO] Mock 模式: {args.mock}")

    compiler = PromptCompiler(
        max_retries=args.max_retries,
        auto_fix_threshold=args.auto_fix_threshold,
        use_mock=args.mock
    )

    result = compiler.compile(input_text)

    if result.success:
        print(f"\n✅ 编译成功!")
        print(f"Tokens 使用: {result.total_tokens}")

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result.dsl_code)

        print(f"已保存到: {args.output}")

        if result.validation_result:
            print("\n验证报告:")
            print(f"  状态: {'通过' if result.validation_result.ok else '失败'}")
            print(f"  错误数: {len(result.validation_result.errors)}")

        return 0
    else:
        print(f"\n❌ 编译失败: {result.error_message}")
        print(f"Tokens 使用: {result.total_tokens}")

        if result.validation_result:
            print("\n验证报告:")
            print(f"  状态: {'通过' if result.validation_result.ok else '失败'}")
            print(f"  错误数: {len(result.validation_result.errors)}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
