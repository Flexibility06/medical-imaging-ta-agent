"""
知识库检索工具

功能：将 RAG 检索封装为工具，供 Agent 自主决定何时使用
设计要点：
- 懒加载向量索引（只在首次使用时加载）
- 自动处理索引不存在的情况
- 返回格式化的检索结果
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.llm_client import embed
from knowledge_base.vector_store import VectorStore
from config.settings import settings

from .base import BaseTool

logger = logging.getLogger(__name__)


class KnowledgeBaseSearchTool(BaseTool):
    """
    课程知识库检索工具

    在课程课件、教材内容中搜索相关信息
    """

    def __init__(self, default_top_k: int = 5):
        self.default_top_k = default_top_k
        self.logger = logging.getLogger(__name__)
        self._store: VectorStore | None = None

    def _load_index(self) -> VectorStore | None:
        """懒加载向量索引"""
        if self._store is not None:
            return self._store

        if not settings.INDEX_PATH.exists() or not settings.CHUNKS_PATH.exists():
            self.logger.warning("知识库索引不存在，请先运行 setup.py")
            return None

        try:
            self._store = VectorStore.load(settings.INDEX_PATH, settings.CHUNKS_PATH)
            self.logger.info(f"已加载知识库索引: {self._store.get_stats()}")
            return self._store
        except Exception as e:
            self.logger.error(f"加载索引失败: {e}")
            return None

    @property
    def name(self) -> str:
        return "search_course_knowledge_base"

    @property
    def description(self) -> str:
        return (
            "在课程知识库中搜索相关内容。"
            "知识库包含课程PPT课件、教材内容等。"
            "仅当用户明确询问课件中的知识点、课程讲义内容、或需要引用课程材料时才使用。"
            "不要用于通用编程问题、基础概念解释（如Python语法、CNN基础原理等你已掌握的知识）、或日常对话。"
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询，建议将学生问题改写为更适合语义检索的形式",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量（默认5）",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def _execute(self, **kwargs) -> str:
        """
        执行知识库检索

        参数:
            query: 搜索查询
            top_k: 返回结果数量

        返回:
            格式化的检索结果字符串
        """
        query = kwargs.get("query", "")
        top_k = kwargs.get("top_k", self.default_top_k)

        if not query:
            return "错误：搜索查询不能为空"

        # 加载索引
        store = self._load_index()
        if store is None:
            return "知识库索引未找到或未构建。请先运行 `python setup.py` 构建知识库。"

        if store.is_empty():
            return "知识库索引为空。"

        try:
            self.logger.info(f"搜索知识库: {query}")

            # 获取查询向量（添加超时控制）
            result = await asyncio.wait_for(embed(query), timeout=30.0)
            query_embedding = result["embeddings"]

            # 检索
            results = store.search(query_embedding, top_k=top_k)

            if not results:
                return f"未在知识库中找到与 '{query}' 相关的内容"

            # 格式化结果
            formatted_results = []
            for i, (chunk, score) in enumerate(results, 1):
                formatted = (
                    f"[{i}] 来源: {chunk.source_info} (相似度: {score:.4f})\n"
                    f"{chunk.text}"
                )
                formatted_results.append(formatted)

            return "\n\n".join(formatted_results)

        except asyncio.TimeoutError:
            return "知识库检索超时，请稍后重试"
        except Exception as e:
            self.logger.error(f"知识库检索失败: {e}")
            return f"检索出错: {str(e)}"


# 用于 Agent 直接检索的便捷函数（不通过工具调用）
async def search_knowledge_base(
    query: str,
    top_k: int = 5,
) -> list[tuple[Any, float]]:
    """
    直接搜索知识库

    使用示例:
        results = await search_knowledge_base("U-Net 结构")
        for chunk, score in results:
            print(f"{chunk.source_info}: {chunk.text[:100]}...")
    """
    import asyncio
    loop = asyncio.get_event_loop()
    # 在线程池中执行同步操作，避免阻塞事件循环
    store = await loop.run_in_executor(
        None, VectorStore.load, settings.INDEX_PATH, settings.CHUNKS_PATH
    )
    result = await embed(query)
    query_embedding = result["embeddings"]
    return await loop.run_in_executor(
        None, store.search, query_embedding, top_k
    )
