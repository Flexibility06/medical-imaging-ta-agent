"""
工具系统测试脚本

功能：验证各工具是否能正常工作
用法：python test_tools.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import load_tools, ToolRegistry
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


async def test_tools():
    """测试所有工具"""
    print("=" * 60)
    print("工具系统测试")
    print("=" * 60)

    # 1. 加载工具
    print("\n[1/4] 加载工具...")
    try:
        registry = ToolRegistry()
        tools = registry.list_tools()
        print(f"✓ 成功加载 {len(tools)} 个工具:")
        for name in tools:
            print(f"  • {name}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

    # 2. 显示工具 OpenAI Schema
    print("\n[2/4] 工具 JSON Schema：")
    try:
        openai_tools = registry.get_openai_definitions()
        for tool_def in openai_tools:
            func = tool_def["function"]
            print(f"\n  📦 {func['name']}")
            print(f"     描述: {func['description'][:60]}...")
            print(f"     参数: {list(func['parameters'].get('properties', {}).keys())}")
    except Exception as e:
        print(f"❌ 获取 Schema 失败: {e}")
        return False

    # 3. 测试 arXiv 搜索
    print("\n[3/4] 测试 arXiv 搜索...")
    try:
        arxiv_tool = registry.get("search_arxiv")
        if arxiv_tool:
            result = await arxiv_tool.execute(
                query="U-Net medical image segmentation",
                max_results=2
            )
            print("✓ arXiv 搜索成功")
            print(f"  结果预览（前200字符）：")
            print(f"  {result[:200]}...")
        else:
            print("⚠️ 未找到 arXiv 工具")
    except Exception as e:
        print(f"⚠️ arXiv 测试跳过或失败: {e}")

    # 4. 测试知识库搜索（如果索引存在）
    print("\n[4/4] 测试知识库搜索...")
    try:
        kb_tool = registry.get("search_course_knowledge_base")
        if kb_tool:
            result = await kb_tool.execute(
                query="医学影像处理",
                top_k=2
            )
            if "未找到" in result or "未构建" in result:
                print("⚠️ 知识库索引未构建，跳过")
            else:
                print("✓ 知识库搜索成功")
                print(f"  结果预览（前200字符）：")
                print(f"  {result[:200]}...")
        else:
            print("⚠️ 未找到知识库工具")
    except Exception as e:
        print(f"⚠️ 知识库测试跳过或失败: {e}")

    print("\n" + "=" * 60)
    print("✅ 工具系统测试完成！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_tools())
    sys.exit(0 if success else 1)
