"""
LLM 连接测试脚本

功能：验证 LLM API 配置是否正确，能否正常生成回答
用法：python test_llm.py
"""

import asyncio
import sys
from pathlib import Path

# 确保可以导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import settings, Settings
from utils.llm_client import chat, LLMError


async def test_llm_connection():
    """测试 LLM API 连接"""
    print("=" * 50)
    print("LLM API 连接测试")
    print("=" * 50)

    # 1. 检查配置
    print("\n[1/3] 检查配置...")
    missing = settings.validate()
    if missing:
        print(f"❌ 配置不完整，缺少: {', '.join(missing)}")
        print(f"\n请按照以下步骤配置:")
        print(f"  1. 复制 .env.example 为 .env")
        print(f"     cp .env.example .env")
        print(f"  2. 编辑 .env 文件，填入你的 API Key 和配置")
        return False

    print(f"✓ API Key: {'*' * 8}{settings.LLM_API_KEY[-4:]}")
    print(f"✓ Base URL: {settings.LLM_BASE_URL}")
    print(f"✓ Model: {settings.LLM_MODEL_NAME}")

    # 2. 测试简单对话
    print("\n[2/3] 测试简单对话...")
    try:
        result = await chat(
            messages=[
                {"role": "system", "content": "你是一个友好的助手。"},
                {"role": "user", "content": "请用一句话介绍你自己。"},
            ],
            temperature=0.7,
        )

        print(f"✓ 调用成功")
        print(f"  耗时: {result['duration_ms']}ms")
        print(f"  Token 使用: {result['usage']}")
        print(f"\n🤖 LLM 回答:")
        print(f"  {result['content']}")

    except LLMError as e:
        print(f"❌ 调用失败: {e}")
        if "timeout" in e.message.lower():
            print("提示: 连接超时，请检查网络或增加 LLM_TIMEOUT 配置")
        elif "401" in str(e.status_code) or "auth" in e.message.lower():
            print("提示: 认证失败，请检查 API Key 是否正确")
        return False

    # 3. 测试 embedding
    print("\n[3/3] 测试 Embedding API...")
    from utils.llm_client import embed

    try:
        result = await embed("医学影像处理")
        print(f"✓ 调用成功")
        print(f"  耗时: {result['duration_ms']}ms")
        print(f"  Token 使用: {result['usage']}")
        print(f"  向量维度: {len(result['embeddings'])}")
    except LLMError as e:
        print(f"❌ 调用失败: {e}")
        print("提示: Embedding 可能使用不同的配置，请检查 .env 中的 EMBEDDING_* 配置")
        return False

    print("\n" + "=" * 50)
    print("✅ 所有测试通过！LLM API 配置正确。")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_llm_connection())
    sys.exit(0 if success else 1)
