"""
Embedding 模块

功能：批量获取文本块的向量嵌入
设计要点：
- 使用 LLM API 获取 embeddings
- 支持批量处理，自动分批
- 内置速率控制，避免触发 API 限制
- 显示进度条和预估成本
"""

import asyncio
import logging
import time
from typing import Callable

from tqdm import tqdm
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.llm_client import get_llm_client, LLMError
from config.settings import settings
from .chunker import TextChunk, estimate_tokens

logger = logging.getLogger(__name__)


class Embedder:
    """
    嵌入生成器类

    负责将文本块转换为向量表示
    """

    def __init__(
        self,
        batch_size: int | None = None,
        delay_between_batches: float = 0.5,
    ):
        """
        初始化嵌入生成器

        参数:
            batch_size: 每批处理的文本数量；默认读取 settings.EMBEDDING_BATCH_SIZE
            delay_between_batches: 批次间的延迟（秒），用于速率控制
        """
        self.client = get_llm_client()
        self.batch_size = (
            batch_size if batch_size is not None else settings.EMBEDDING_BATCH_SIZE
        )
        self.delay = delay_between_batches
        self.logger = logging.getLogger(__name__)

    def estimate_cost(self, chunks: list[TextChunk]) -> dict:
        """
        估算 embedding 的 token 数量和成本

        参数:
            chunks: 文本块列表

        返回:
            包含 token 数和估算成本的字典
        """
        total_tokens = sum(estimate_tokens(c.text) for c in chunks)

        # 估算成本（基于 OpenAI text-embedding-3-small: $0.02 / 1M tokens）
        # 不同服务商价格不同，这里给出参考值
        estimated_cost_usd = total_tokens * 0.02 / 1_000_000

        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "estimated_cost_cny": estimated_cost_usd * 7.2,  # 粗略汇率
        }

    async def embed_chunks(
        self,
        chunks: list[TextChunk],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[list[list[float]], list[TextChunk]]:
        """
        批量获取文本块的嵌入向量

        参数:
            chunks: 文本块列表
            progress_callback: 进度回调函数 (current, total) -> None

        返回:
            (embeddings, valid_chunks) 元组
            embeddings: 向量列表
            valid_chunks: 成功获取嵌入的文本块（过滤掉失败的）
        """
        embeddings = []
        valid_chunks = []
        total = len(chunks)

        self.logger.info(f"开始生成 {total} 个块的 embeddings...")

        for i in range(0, total, self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_texts = [c.text for c in batch]

            try:
                # 调用 API 获取 embeddings
                result = await self.client.embedding(batch_texts)
                batch_embeddings = result["embeddings"]

                # 确保是列表的列表
                if not isinstance(batch_embeddings[0], list):
                    batch_embeddings = [batch_embeddings]

                embeddings.extend(batch_embeddings)
                valid_chunks.extend(batch)

                # 进度回调
                if progress_callback:
                    progress_callback(min(i + self.batch_size, total), total)

                # 速率控制
                if i + self.batch_size < total:
                    await asyncio.sleep(self.delay)

            except LLMError as e:
                self.logger.error(f"Embedding 批次失败 ({i}-{i+len(batch)}): {e}")
                # 继续处理下一批，不中断整体流程

        self.logger.info(f"✓ 成功生成 {len(embeddings)}/{total} 个 embeddings")
        return embeddings, valid_chunks

    async def embed_chunks_with_progress(
        self,
        chunks: list[TextChunk],
        show_progress: bool = True,
    ) -> tuple[list[list[float]], list[TextChunk]]:
        """
        带进度条的 embedding 生成

        参数:
            chunks: 文本块列表
            show_progress: 是否显示进度条

        返回:
            (embeddings, valid_chunks) 元组
        """
        if not show_progress:
            return await self.embed_chunks(chunks)

        # 使用 tqdm 显示进度条
        pbar = tqdm(total=len(chunks), desc="生成 Embeddings", unit="chunks")

        def update_progress(current: int, total: int):
            pbar.n = current
            pbar.refresh()

        try:
            result = await self.embed_chunks(chunks, update_progress)
        finally:
            pbar.close()

        return result

    def embed_chunks_sync(
        self,
        chunks: list[TextChunk],
        show_progress: bool = True,
    ) -> tuple[list[list[float]], list[TextChunk]]:
        """
        同步版本的 embedding 生成（便捷函数）

        使用示例:
            embedder = Embedder()
            embeddings, valid_chunks = embedder.embed_chunks_sync(chunks)
        """
        return asyncio.run(
            self.embed_chunks_with_progress(chunks, show_progress)
        )


# 便捷函数

async def generate_embeddings(
    chunks: list[TextChunk],
    batch_size: int | None = None,
) -> tuple[list[list[float]], list[TextChunk]]:
    """
    便捷函数：异步生成 embeddings

    使用示例:
        from knowledge_base.chunker import chunk_texts
        from knowledge_base.embedder import generate_embeddings

        chunks = chunk_texts(pages)
        embeddings, valid_chunks = await generate_embeddings(chunks)
    """
    embedder = Embedder(
        batch_size=batch_size if batch_size is not None else settings.EMBEDDING_BATCH_SIZE
    )
    return await embedder.embed_chunks_with_progress(chunks)


def generate_embeddings_sync(
    chunks: list[TextChunk],
    batch_size: int | None = None,
) -> tuple[list[list[float]], list[TextChunk]]:
    """
    便捷函数：同步生成 embeddings
    """
    embedder = Embedder(
        batch_size=batch_size if batch_size is not None else settings.EMBEDDING_BATCH_SIZE
    )
    return embedder.embed_chunks_sync(chunks)
