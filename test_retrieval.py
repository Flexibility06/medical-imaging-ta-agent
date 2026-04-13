"""
知识库检索测试脚本

功能：验证知识库构建后能否正常检索
用法：
  1. 先运行 python setup.py 构建知识库
  2. 然后运行 python test_retrieval.py "你的查询"
"""

import asyncio
import sys
from pathlib import Path

# 确保可以导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import settings
from knowledge_base.vector_store import VectorStore
from utils.llm_client import embed, LLMError


async def test_retrieval(query: str, top_k: int = 3):
    """测试检索功能"""
    print("=" * 60)
    print("知识库检索测试")
    print("=" * 60)

    # 1. 检查索引是否存在
    print("\n[1/3] 检查索引文件...")
    if not settings.INDEX_PATH.exists():
        print(f"❌ 索引文件不存在: {settings.INDEX_PATH}")
        print("\n请先运行: python setup.py")
        return False

    if not settings.CHUNKS_PATH.exists():
        print(f"❌ 元数据文件不存在: {settings.CHUNKS_PATH}")
        return False

    print(f"✓ 索引文件存在: {settings.INDEX_PATH}")

    # 2. 加载索引
    print("\n[2/3] 加载向量索引...")
    try:
        store = VectorStore.load(settings.INDEX_PATH, settings.CHUNKS_PATH)
        stats = store.get_stats()
        print(f"✓ 加载成功")
        print(f"  • 向量数量: {stats['total_vectors']}")
        print(f"  • 向量维度: {stats['dimension']}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False

    # 3. 执行检索
    print(f"\n[3/3] 执行检索...")
    print(f"  查询: {query}")
    print(f"  top_k: {top_k}")
    print()

    try:
        # 获取查询向量
        result = await embed(query)
        query_embedding = result["embeddings"]

        # 检索
        results = store.search(query_embedding, top_k=top_k)

        if not results:
            print("⚠️ 未找到相关结果")
            return True

        print(f"✓ 找到 {len(results)} 个相关结果:\n")

        for i, (chunk, score) in enumerate(results, 1):
            print(f"[{i}] 相似度: {score:.4f} | 来源: {chunk.source_info}")
            print("-" * 60)
            # 截断显示，避免过长
            text = chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text
            print(text)
            print()

        print("=" * 60)
        print("✅ 检索测试通过！")
        print("=" * 60)
        return True

    except LLMError as e:
        print(f"❌ Embedding 失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 检索失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    # 获取查询参数
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "医学影像处理"

    success = asyncio.run(test_retrieval(query))
    sys.exit(0 if success else 1)
