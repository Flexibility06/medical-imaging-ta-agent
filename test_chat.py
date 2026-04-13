"""
对话引擎测试脚本

功能：验证 Agent 能否正常对话和调用工具
用法：python test_chat.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.chat_engine import ChatEngine
from agent.student_profile import ProfileManager


async def test_chat():
    """测试对话引擎"""
    print("=" * 60)
    print("对话引擎测试")
    print("=" * 60)

    # 1. 加载或创建学生画像
    print("\n[1/3] 加载学生画像...")
    profile_manager = ProfileManager()

    if profile_manager.exists():
        profile = profile_manager.load()
        print(f"✓ 已加载现有画像")
        if profile.background:
            print(f"  背景: {profile.background}")
    else:
        print("⚠️ 未找到画像，使用默认设置")
        from agent.student_profile import StudentProfile
        profile = StudentProfile()

    # 2. 初始化对话引擎
    print("\n[2/3] 初始化对话引擎...")
    try:
        engine = ChatEngine(
            student_background=profile.background,
            max_tool_calls=5,
        )
        tools = engine.tool_registry.list_tools()
        print(f"✓ 引擎初始化成功")
        print(f"  可用工具: {', '.join(tools)}")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

    # 3. 测试简单对话
    print("\n[3/3] 测试对话...")
    test_questions = [
        "你好，请介绍一下你自己",
        "什么是医学影像处理？",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n  测试 {i}/{len(test_questions)}:")
        print(f"  用户: {question}")
        print(f"  助手: ", end="", flush=True)

        try:
            response = await engine.chat(question, show_tool_status=True)

            # 显示回答（截断）
            content = response.content
            if len(content) > 200:
                content = content[:200] + "..."
            print(content)

            # 显示工具调用信息
            if response.tool_calls:
                print(f"  [调用了 {len(response.tool_calls)} 个工具]")

        except Exception as e:
            print(f"错误: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    print("\n" + "=" * 60)
    print("✅ 对话引擎测试完成！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_chat())
    sys.exit(0 if success else 1)
