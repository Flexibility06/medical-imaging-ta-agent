#!/usr/bin/env python3
"""
验证修复：测试 Agent 是否能根据问题性质自主决策是否调用工具

用法：python test_fix.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.chat_engine import ChatEngine


async def test_scenarios():
    """测试 4 个典型场景"""

    test_cases = [
        {
            "name": "日常问候",
            "input": "你好",
            "expected_tools": [],
        },
        {
            "name": "通用编程问题",
            "input": "请帮我写一段 CNN 的 Python 代码",
            "expected_tools": [],  # 期望不调用任何工具，直接生成代码
        },
        {
            "name": "课件内容查询",
            "input": "课件里 U-Net 的网络结构是怎么讲的？",
            "expected_tools": ["search_course_knowledge_base"],
        },
        {
            "name": "最新论文查询",
            "input": "最近有什么医学影像分割的新论文？",
            "expected_tools": ["search_arxiv"],  # 可能调用 arXiv 或 web_search
        },
    ]

    print("=" * 70)
    print("修复验证测试：检查 Agent 是否能自主决策工具调用")
    print("=" * 70)

    # 初始化对话引擎
    engine = ChatEngine(student_background=None, max_tool_calls=5)

    for i, case in enumerate(test_cases, 1):
        print(f"\n测试 {i}/4: {case['name']}")
        print(f"输入: \"{case['input']}\"")
        print(f"期望调用: {case['expected_tools'] if case['expected_tools'] else '不调用任何工具'}")

        try:
            response = await engine.chat(case["input"], show_tool_status=True)

            actual_tools = [tc.tool for tc in response.tool_calls]
            print(f"实际调用: {actual_tools if actual_tools else '无'}")

            # 判断是否符合预期
            if case["expected_tools"]:
                # 期望调用特定工具
                if any(t in actual_tools for t in case["expected_tools"]):
                    print("✅ 符合预期")
                else:
                    print("⚠️ 未触发期望的工具，但可能直接回答了")
            else:
                # 期望不调用工具
                if not actual_tools:
                    print("✅ 符合预期：未调用工具，直接回答")
                else:
                    print("❌ 不符合预期：不应调用工具却调用了")

            # 显示回答前 100 字符
            preview = response.content[:100].replace('\n', ' ')
            print(f"回答预览: {preview}...")

        except Exception as e:
            print(f"❌ 测试出错: {e}")

        print("-" * 70)

    print("\n测试完成！")
    print("如果通用编程问题和日常问候没有调用工具，修复成功。")


if __name__ == "__main__":
    asyncio.run(test_scenarios())
