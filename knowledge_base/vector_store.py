"""
向量存储模块

功能：基于 FAISS 的向量索引管理
设计要点：
- 支持索引的构建、保存、加载
- 支持相似度检索
- 索引持久化到本地文件
- 支持增量更新（预留接口）
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .chunker import TextChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    向量存储类

    使用 FAISS 管理文本块的向量索引
    """

    def __init__(self, dimension: int | None = None):
        """
        初始化向量存储

        参数:
            dimension: 向量维度，如果为 None 则在添加数据时自动确定
        """
        self.dimension = dimension
        self.index: faiss.Index | None = None
        self.chunks: list[TextChunk] = []
        self.logger = logging.getLogger(__name__)

    def _create_index(self, dimension: int) -> faiss.Index:
        """
        创建 FAISS 索引

        使用 IndexFlatIP（内积）配合归一化向量实现余弦相似度
        """
        # IndexFlatIP 计算内积，如果向量已归一化，内积 = 余弦相似度
        index = faiss.IndexFlatIP(dimension)
        return index

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        L2 归一化向量

        使内积等价于余弦相似度
        """
        # 转换为 float32（FAISS 要求）
        vectors = vectors.astype(np.float32)

        # L2 归一化
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除以零
        normalized = vectors / norms

        return normalized

    def add(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        添加文本块和对应的嵌入向量到索引

        参数:
            chunks: 文本块列表
            embeddings: 对应的嵌入向量列表
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 和 embeddings 数量不匹配")

        if not chunks:
            self.logger.warning("没有数据需要添加")
            return

        # 转换为 numpy 数组
        vectors = np.array(embeddings)

        # 确定维度
        if self.dimension is None:
            self.dimension = vectors.shape[1]
            self.index = self._create_index(self.dimension)
            self.logger.info(f"创建索引，维度: {self.dimension}")
        elif vectors.shape[1] != self.dimension:
            raise ValueError(
                f"向量维度不匹配: 期望 {self.dimension}, 实际 {vectors.shape[1]}. "
                f"可能是 Embedding 模型变更导致，请删除索引文件重新运行 setup.py"
            )

        # 归一化向量
        normalized_vectors = self._normalize_vectors(vectors)

        # 添加到索引
        self.index.add(normalized_vectors)
        self.chunks.extend(chunks)

        self.logger.info(
            f"添加 {len(chunks)} 个向量，索引总计 {self.index.ntotal} 个"
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[TextChunk, float]]:
        """
        检索最相似的文本块

        参数:
            query_embedding: 查询向量
            top_k: 返回结果数量

        返回:
            (TextChunk, score) 列表，按相似度降序排列
            score 为余弦相似度（0-1，越接近1越相似）
        """
        if self.index is None or self.index.ntotal == 0:
            self.logger.warning("索引为空，无法检索")
            return []

        # 转换查询向量
        query_vector = np.array([query_embedding]).astype(np.float32)
        query_vector = self._normalize_vectors(query_vector)

        # 检索
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        # 构建结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append((chunk, float(score)))

        return results

    def save(self, index_path: Path, chunks_path: Path) -> None:
        """
        保存索引到文件

        参数:
            index_path: FAISS 索引文件路径
            chunks_path: 文本块元数据文件路径
        """
        if self.index is None:
            raise ValueError("索引为空，无法保存")

        # 确保目录存在
        index_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存 FAISS 索引
        faiss.write_index(self.index, str(index_path))

        # 保存文本块元数据（使用 pickle 保存复杂对象）
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        self.logger.info(f"索引已保存: {index_path}")
        self.logger.info(f"元数据已保存: {chunks_path}")

    @classmethod
    def load(cls, index_path: Path, chunks_path: Path) -> "VectorStore":
        """
        从文件加载索引

        参数:
            index_path: FAISS 索引文件路径
            chunks_path: 文本块元数据文件路径

        返回:
            加载完成的 VectorStore 实例
        """
        store = cls()

        if not index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {chunks_path}")

        # 加载 FAISS 索引
        store.index = faiss.read_index(str(index_path))
        store.dimension = store.index.d

        # 加载文本块元数据
        with open(chunks_path, "rb") as f:
            store.chunks = pickle.load(f)

        logger.info(
            f"加载索引: {len(store.chunks)} 个 chunks, 维度 {store.dimension}"
        )

        return store

    def is_empty(self) -> bool:
        """检查索引是否为空"""
        return self.index is None or self.index.ntotal == 0

    def get_stats(self) -> dict[str, Any]:
        """获取索引统计信息"""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "total_chunks": len(self.chunks),
        }


# 便捷函数

def build_index(
    chunks: list[TextChunk],
    embeddings: list[list[float]],
) -> VectorStore:
    """
    便捷函数：从 chunks 和 embeddings 构建索引

    使用示例:
        from knowledge_base.chunker import chunk_texts
        from knowledge_base.embedder import generate_embeddings_sync
        from knowledge_base.vector_store import build_index

        chunks = chunk_texts(pages)
        embeddings, valid_chunks = generate_embeddings_sync(chunks)
        store = build_index(valid_chunks, embeddings)
    """
    store = VectorStore()
    store.add(chunks, embeddings)
    return store


def save_index(
    store: VectorStore,
    index_path: Path,
    chunks_path: Path,
) -> None:
    """
    便捷函数：保存索引
    """
    store.save(index_path, chunks_path)


def load_index(
    index_path: Path,
    chunks_path: Path,
) -> VectorStore:
    """
    便捷函数：加载索引
    """
    return VectorStore.load(index_path, chunks_path)
