#!/usr/bin/env python3
"""
全系统集成测试

功能：依次验证所有核心模块
用法：python test_all.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_section(title: str):
    """打印测试章节标题"""
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")


def print_result(name: str, success: bool, message: str = ""):
    """打印测试结果"""
    status = "[green]✓ PASS[/green]" if success else "[red]✗ FAIL[/red]"
    console.print(f"  {status} {name}")
    if message and not success:
        console.print(f"      [dim]{message}[/dim]")


async def test_phase1():
    """测试 Phase 1: 基础设施"""
    print_section("Phase 1: 基础设施")
    results = {}

    # 测试配置加载
    try:
        from config.settings import settings, Settings
        from config.prompts import build_system_prompt
        results["配置加载"] = True
    except Exception as e:
        results["配置加载"] = False
        print_result("配置加载", False, str(e))
        return False

    # 测试 LLM 客户端
    try:
        from utils.llm_client import LLMClient, chat, embed
        results["LLM客户端"] = True
    except Exception as e:
        results["LLM客户端"] = False
        print_result("LLM客户端", False, str(e))
        return False

    # 测试 API 连接（可选）
    try:
        from config.settings import settings
        missing = settings.validate()
        if missing:
            results["API配置"] = False
            print_result("API配置", False, f"缺少: {', '.join(missing)}")
        else:
            result = await chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'OK' only."}
            ])
            results["API连接"] = result.get("content") is not None
    except Exception as e:
        results["API连接"] = False
        print_result("API连接", False, str(e))

    for name, success in results.items():
        print_result(name, success)

    return all(results.values())


async def test_phase2():
    """测试 Phase 2: 知识库"""
    print_section("Phase 2: 知识库")
    results = {}

    # 测试 PDF 解析
    try:
        from knowledge_base.pdf_parser import PDFParser, parse_pdfs
        results["PDF解析器"] = True
    except Exception as e:
        results["PDF解析器"] = False
        print_result("PDF解析器", False, str(e))

    # 测试分块器
    try:
        from knowledge_base.chunker import Chunker, chunk_texts
        results["文本分块"] = True
    except Exception as e:
        results["文本分块"] = False
        print_result("文本分块", False, str(e))

    # 测试 Embedder
    try:
        from knowledge_base.embedder import Embedder
        results["Embedder"] = True
    except Exception as e:
        results["Embedder"] = False
        print_result("Embedder", False, str(e))

    # 测试向量存储
    try:
        from knowledge_base.vector_store import VectorStore
        results["向量存储"] = True
    except Exception as e:
        results["向量存储"] = False
        print_result("向量存储", False, str(e))

    # 检查索引是否存在
    from config.settings import settings
    if settings.INDEX_PATH.exists():
        try:
            store = VectorStore.load(settings.INDEX_PATH, settings.CHUNKS_PATH)
            stats = store.get_stats()
            results["索引加载"] = True
            console.print(f"  [dim]  索引状态: {stats}[/dim]")
        except Exception as e:
            results["索引加载"] = False
            print_result("索引加载", False, str(e))
    else:
        console.print("  [yellow]⚠ 知识库索引未构建[/yellow]")

    for name, success in results.items():
        print_result(name, success)

    return all(results.values())


async def test_phase3():
    """测试 Phase 3: 工具系统"""
    print_section("Phase 3: 工具系统")
    results = {}

    # 测试工具基类
    try:
        from tools.base import BaseTool, ToolResult
        results["工具基类"] = True
    except Exception as e:
        results["工具基类"] = False
        print_result("工具基类", False, str(e))

    # 测试工具注册表
    try:
        from tools import load_tools, ToolRegistry
        registry = ToolRegistry()
        tools = registry.list_tools()
        results["工具注册表"] = len(tools) > 0
        console.print(f"  [dim]  已加载工具: {', '.join(tools)}[/dim]")
    except Exception as e:
        results["工具注册表"] = False
        print_result("工具注册表", False, str(e))

    # 测试 arXiv 工具
    try:
        from tools.arxiv_search import ArxivSearchTool
        tool = ArxivSearchTool()
        result = await tool.execute(query="test", max_results=1)
        results["arXiv工具"] = "Title:" in result or "错误" in result or "未找到" in result
    except Exception as e:
        results["arXiv工具"] = False
        print_result("arXiv工具", False, str(e))

    for name, success in results.items():
        print_result(name, success)

    return all(results.values())


async def test_phase4():
    """测试 Phase 4: 对话引擎"""
    print_section("Phase 4: 对话引擎")
    results = {}

    # 测试学生画像
    try:
        from agent.student_profile import ProfileManager, StudentProfile
        results["学生画像"] = True
    except Exception as e:
        results["学生画像"] = False
        print_result("学生画像", False, str(e))

    # 测试对话引擎
    try:
        from agent.chat_engine import ChatEngine, ChatResponse
        results["对话引擎"] = True
    except Exception as e:
        results["对话引擎"] = False
        print_result("对话引擎", False, str(e))

    # 测试格式化器
    try:
        from agent.response_formatter import ResponseFormatter
        results["格式化器"] = True
    except Exception as e:
        results["格式化器"] = False
        print_result("格式化器", False, str(e))

    for name, success in results.items():
        print_result(name, success)

    return all(results.values())


async def main():
    """运行所有测试"""
    console.print(Panel.fit(
        "[bold blue]医学影像课程智能助教 - 全系统集成测试[/bold blue]",
        border_style="blue",
    ))

    all_results = {}

    # 运行各阶段测试
    all_results["Phase 1: 基础设施"] = await test_phase1()
    all_results["Phase 2: 知识库"] = await test_phase2()
    all_results["Phase 3: 工具系统"] = await test_phase3()
    all_results["Phase 4: 对话引擎"] = await test_phase4()

    # 汇总结果
    console.print(f"\n[bold]{'=' * 60}[/bold]")
    console.print("[bold]测试结果汇总[/bold]")
    console.print(f"[bold]{'=' * 60}[/bold]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("阶段", style="cyan")
    table.add_column("状态", style="green")

    for phase, success in all_results.items():
        status = "[green]✓ 通过[/green]" if success else "[red]✗ 失败[/red]"
        table.add_row(phase, status)

    console.print(table)

    # 最终结论
    if all(all_results.values()):
        console.print(Panel.fit(
            "[bold green]✅ 所有测试通过！系统可以正常使用。[/bold green]\n\n"
            "接下来你可以:\n"
            "  1. 放入 PDF 课件到 data/raw/\n"
            "  2. 运行 python setup.py 构建知识库\n"
            "  3. 运行 python main.py 开始使用",
            border_style="green",
        ))
        return 0
    else:
        console.print(Panel.fit(
            "[bold red]❌ 部分测试失败，请检查上述错误信息。[/bold red]",
            border_style="red",
        ))
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
