#!/usr/bin/env python3
"""
医学影像课程智能助教 - CLI 入口

功能：
- 启动检查和初始化
- 交互式对话循环
- 特殊命令处理（/help, /clear, /exit 等）

用法：
  python main.py           # 正常启动
  python main.py --reset   # 重置学生画像
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import settings, Settings
from config.prompts import build_system_prompt
from agent.chat_engine import ChatEngine
from agent.student_profile import ProfileManager
from agent.response_formatter import ResponseFormatter, print_welcome, print_help

console = Console()


def check_prerequisites() -> bool:
    """
    检查启动前置条件

    1. API 配置是否正确
    2. 知识库索引是否存在（警告但不强制）
    """
    missing = settings.validate()
    if missing:
        console.print(Panel.fit(
            f"[red]配置不完整，缺少: {', '.join(missing)}[/red]\n\n"
            "请按以下步骤配置:\n"
            "1. 复制 .env.example 为 .env: cp .env.example .env\n"
            "2. 编辑 .env 文件，填入你的 API Key",
            title="配置错误",
            border_style="red",
        ))
        return False
    return True


async def interactive_chat(engine: ChatEngine, formatter: ResponseFormatter):
    """
    交互式对话循环

    处理用户输入，识别特殊命令，调用对话引擎
    """
    print_welcome(
        name=getattr(engine, '_profile_name', ''),
        background=engine.student_background
    )

    while True:
        try:
            # 获取用户输入
            user_input = console.input("[bold blue]你[/bold blue]: ").strip()

            if not user_input:
                continue

            # 处理特殊命令
            if user_input.startswith("/"):
                handled = await handle_command(user_input, engine, formatter)
                if handled == "exit":
                    break
                continue

            # 调用对话引擎
            console.print("[bold green]助手[/bold green]: ", end="")

            response = await engine.chat(user_input, show_tool_status=True)

            # 显示回答
            formatter.format_response(response.content)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]再见！[/yellow]")
            break
        except EOFError:
            break
        except Exception as e:
            formatter.print_error(f"出错了: {e}")
            logging.exception("对话出错")


async def handle_command(
    cmd: str,
    engine: ChatEngine,
    formatter: ResponseFormatter
) -> str | None:
    """
    处理特殊命令

    参数:
        cmd: 命令字符串（如 "/help"）

    返回:
        "exit" 表示退出程序
        None 表示继续循环
    """
    cmd = cmd.lower().strip()

    if cmd in ["/exit", "/quit", "/q"]:
        console.print("[yellow]再见！[/yellow]")
        return "exit"

    elif cmd == "/help":
        print_help()

    elif cmd == "/clear":
        engine.clear_history()
        formatter.print_success("对话历史已清空")

    elif cmd == "/profile":
        profile_manager = ProfileManager()
        profile = profile_manager.interactive_setup()
        # 更新引擎的背景与 system prompt（下一条用户消息起使用）
        engine.student_background = profile.background
        engine.system_prompt = build_system_prompt(profile.background)
        engine._profile_name = profile.name
        formatter.print_success("画像已更新，下一条消息起将按新设置回答")

    elif cmd == "/status":
        console.print(Panel(
            f"[bold]当前状态[/bold]\n\n"
            f"会话 ID: {engine.session_id}\n"
            f"学生背景: {engine.student_background or '未设置'}\n"
            f"历史轮数: {len([m for m in engine.history if m.role == 'user'])}\n"
            f"可用工具: {', '.join(engine.tool_registry.list_tools())}",
            border_style="blue",
        ))

    elif cmd.startswith("/"):
        formatter.print_error(f"未知命令: {cmd}，输入 /help 查看帮助")

    return None


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="医学影像课程智能助教",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
特殊命令:
  /help      显示帮助信息
  /clear     清空对话历史
  /profile   修改学生画像
  /status    查看当前状态
  /exit      退出程序

示例:
  python main.py              # 正常启动
  python main.py --reset      # 重置画像并启动
        """
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="重置学生画像（重新配置背景）",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="跳过画像配置（使用默认设置）",
    )

    args = parser.parse_args()

    # 确保目录存在
    settings.ensure_directories()

    # 检查配置
    if not check_prerequisites():
        sys.exit(1)

    # 管理学生画像
    profile_manager = ProfileManager()

    if args.reset and profile_manager.exists():
        profile_manager.profile_path.unlink()
        console.print("[yellow]已重置学生画像[/yellow]")

    if args.no_profile:
        from agent.student_profile import StudentProfile
        profile = StudentProfile()
    else:
        profile = profile_manager.get_or_create()

    # 初始化对话引擎
    try:
        engine = ChatEngine(
            student_background=profile.background,
            max_tool_calls=10,
            max_history=20,
        )
        # 保存名字用于显示
        engine._profile_name = profile.name
    except Exception as e:
        console.print(Panel.fit(
            f"[red]初始化失败: {e}[/red]",
            title="错误",
            border_style="red",
        ))
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)

    # 检查知识库
    if not settings.INDEX_PATH.exists():
        console.print(Panel(
            "[yellow]⚠️ 知识库索引不存在[/yellow]\n\n"
            f"某些功能可能受限。如需使用知识库，请先运行:\n"
            f"  python setup.py",
            border_style="yellow",
        ))
        console.input("按回车键继续...")

    # 启动交互式对话
    formatter = ResponseFormatter()

    try:
        asyncio.run(interactive_chat(engine, formatter))
    except KeyboardInterrupt:
        console.print("\n[yellow]再见！[/yellow]")


if __name__ == "__main__":
    main()
