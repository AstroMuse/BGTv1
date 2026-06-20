"""
DeepSeek-V4-Pro 思考模式（max）API 测试脚本
=================================================

目的：测试 deepseek-v4-pro API 在 max 思考模式下，流式返回的
      reasoning_content（深度思考）与 content（最终回答），
      用于和 DeepSeek 网页版「专家模式 - 深度思考」的输出做对照。

特性：
  - 流式传输（stream=True）
  - reasoning_content 与 content 实时分段打印
  - 运行后在终端输入一个问题，完成一轮对话后即结束
  - 同时把完整记录写入 logs/ 下的带时间戳日志文件
  - 每次运行都是全新对话（进程级隔离，无上一轮上下文记忆污染）

依赖：
    pip install openai

API Key（任选其一）：
    1) 环境变量：  export DEEPSEEK_API_KEY=sk-xxx
    2) 运行时按提示手动输入

思考强度（reasoning_effort，默认 high）：
    python test_deepseek_thinking.py                       # 默认 high
    python test_deepseek_thinking.py --reasoning-effort max
    python test_deepseek_thinking.py -r low

参考文档：https://api-docs.deepseek.com/zh-cn/guides/thinking_mode
"""

import os
import sys
import argparse
import datetime

try:
    from openai import OpenAI
except ImportError:
    print("缺少依赖：请先运行  pip install openai")
    sys.exit(1)

# ---- 固定配置 ----
MODEL = "deepseek-v4-pro"
BASE_URL = "https://api.deepseek.com"
DEFAULT_REASONING_EFFORT = "high"   # 默认思考强度
REASONING_EFFORT_CHOICES = ["low", "medium", "high", "max"]
LOG_DIR = "logs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepSeek-V4-Pro 思考模式 API 测试（流式）"
    )
    parser.add_argument(
        "-r", "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=DEFAULT_REASONING_EFFORT,
        help=f"思考强度，默认 {DEFAULT_REASONING_EFFORT}（可选：{', '.join(REASONING_EFFORT_CHOICES)}）",
    )
    return parser.parse_args()


def get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        try:
            key = input("未检测到环境变量 DEEPSEEK_API_KEY，请粘贴 API Key: ").strip()
        except (EOFError, KeyboardInterrupt):
            key = ""
    if not key:
        print("未提供 API Key，退出。")
        sys.exit(1)
    return key


def new_log_file() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOG_DIR, f"deepseek_thinking_{ts}.log")


def main():
    args = parse_args()
    reasoning_effort = args.reasoning_effort

    client = OpenAI(api_key=get_api_key(), base_url=BASE_URL)

    # ---- 单轮对话：终端输入问题 ----
    try:
        question = input("\n请输入你的问题：\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        return
    if not question:
        print("问题为空，退出。")
        return

    log_path = new_log_file()
    log_lines = []  # 收集完整内容，最后统一写盘

    def log(line: str = ""):
        log_lines.append(line)

    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    log(f"时间: {started_at}")
    log(f"模型: {MODEL}")
    log(f"思考模式: reasoning_effort={reasoning_effort}, thinking=enabled")
    log("=" * 70)
    log("【提问】")
    log(question)
    log("=" * 70)

    # ---- 发起流式请求 ----
    # 注意：思考模式下不支持 temperature / top_p 等采样参数，故不传
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": question}],
            stream=True,
            reasoning_effort=reasoning_effort,
            extra_body={"thinking": {"type": "enabled"}},
        )
    except Exception as e:
        print(f"\n请求失败：{e}")
        return

    reasoning_buf = []
    content_buf = []
    printed_reasoning_header = False
    printed_content_header = False

    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 思考过程（chain-of-thought）
            reasoning_piece = getattr(delta, "reasoning_content", None)
            if reasoning_piece:
                if not printed_reasoning_header:
                    print("\n========== 深度思考 (reasoning_content) ==========\n")
                    printed_reasoning_header = True
                print(reasoning_piece, end="", flush=True)
                reasoning_buf.append(reasoning_piece)

            # 最终回答
            content_piece = getattr(delta, "content", None)
            if content_piece:
                if not printed_content_header:
                    print("\n\n========== 最终回答 (content) ==========\n")
                    printed_content_header = True
                print(content_piece, end="", flush=True)
                content_buf.append(content_piece)
    except Exception as e:
        print(f"\n流式接收中断：{e}")
    finally:
        print()  # 收尾换行

    # ---- 写入日志 ----
    log("")
    log("【深度思考 reasoning_content】")
    log("".join(reasoning_buf) if reasoning_buf else "(无 reasoning_content 返回)")
    log("")
    log("【最终回答 content】")
    log("".join(content_buf) if content_buf else "(无 content 返回)")
    log("")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\n本轮对话已保存至: {log_path}")


if __name__ == "__main__":
    main()
