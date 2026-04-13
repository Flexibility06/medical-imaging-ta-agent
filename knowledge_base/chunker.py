"""
文本分块模块

功能：将 PDF 页面文本切分为适合检索的语义块
设计要点：
- 按语义段落分块，优先在段落边界处切分
- 支持 chunk 大小和 overlap 配置
- 保留完整的元数据（来源文件、页码等）
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .pdf_parser import PDFPage, estimate_tokens

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """
    文本块数据类

    属性:
        id: 唯一标识符
        text: 块内容
        file_name: 来源文件名
        page_number: 来源页码
        chunk_index: 在同源页面中的块序号
        start_pos: 在源文本中的起始位置
        end_pos: 在源文本中的结束位置
    """
    id: str
    text: str
    file_name: str
    page_number: int
    chunk_index: int
    start_pos: int
    end_pos: int

    @property
    def source_info(self) -> str:
        """返回来源信息字符串"""
        return f"{self.file_name} p.{self.page_number}"

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "id": self.id,
            "text": self.text,
            "file_name": self.file_name,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "source_info": self.source_info,
        }


class Chunker:
    """
    文本分块器类

    将长文本切分为固定大小的块，支持重叠保留上下文
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        """
        初始化分块器

        参数:
            chunk_size: 目标块大小（按 token 估算）
            chunk_overlap: 块之间的重叠大小（token）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """
        将文本按段落分割

        优先按双换行分割，如果没有则按单换行分割
        """
        # 先尝试按双换行（空行）分割
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # 如果段落太大，尝试按单换行进一步分割
        result = []
        for p in paragraphs:
            if estimate_tokens(p) > self.chunk_size * 1.5:
                # 段落太大，按单换行分割
                sub_paras = [s.strip() for s in p.split("\n") if s.strip()]
                result.extend(sub_paras)
            else:
                result.append(p)

        return result

    def _merge_paragraphs_into_chunks(
        self,
        paragraphs: list[str],
    ) -> list[tuple[int, int, str]]:
        """
        将段落合并为指定大小的块

        参数:
            paragraphs: 段落列表

        返回:
            块列表，每个元素为 (start_pos, end_pos, chunk_text)
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        current_start = 0
        position = 0

        for para in paragraphs:
            para_tokens = estimate_tokens(para)

            # 如果当前块加上新段落会超出限制，先保存当前块
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append((current_start, position, chunk_text))

                # 创建新块，保留部分重叠内容
                overlap_tokens = 0
                overlap_chunks = []
                for p in reversed(current_chunk):
                    p_tokens = estimate_tokens(p)
                    if overlap_tokens + p_tokens <= self.chunk_overlap:
                        overlap_chunks.insert(0, p)
                        overlap_tokens += p_tokens
                    else:
                        break

                current_chunk = overlap_chunks + [para]
                current_tokens = overlap_tokens + para_tokens
                current_start = position - len("\n\n".join(overlap_chunks))
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

            position += len(para) + 2  # +2 for "\n\n"

        # 处理最后一个块
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append((current_start, position, chunk_text))

        return chunks

    def chunk_page(self, page: PDFPage, chunk_id_prefix: str = "") -> list[TextChunk]:
        """
        将单个 PDF 页面分块

        参数:
            page: PDFPage 对象
            chunk_id_prefix: ID 前缀

        返回:
            TextChunk 列表
        """
        # 分割为段落
        paragraphs = self._split_into_paragraphs(page.text)

        if not paragraphs:
            return []

        # 合并为块
        chunk_data = self._merge_paragraphs_into_chunks(paragraphs)

        # 创建 TextChunk 对象
        chunks = []
        for idx, (start, end, text) in enumerate(chunk_data):
            chunk = TextChunk(
                id=f"{chunk_id_prefix}{page.file_name}_p{page.page_number}_c{idx}",
                text=text,
                file_name=page.file_name,
                page_number=page.page_number,
                chunk_index=idx,
                start_pos=start,
                end_pos=end,
            )
            chunks.append(chunk)

        return chunks

    def chunk_pages(
        self,
        pages: list[PDFPage],
    ) -> list[TextChunk]:
        """
        批量分块多个 PDF 页面

        参数:
            pages: PDFPage 列表

        返回:
            所有页面的 TextChunk 列表
        """
        all_chunks = []

        for page in pages:
            chunks = self.chunk_page(page)
            all_chunks.extend(chunks)

        self.logger.info(f"分块完成: {len(pages)} 页 → {len(all_chunks)} 个块")
        return all_chunks

    def get_chunk_stats(self, chunks: list[TextChunk]) -> dict:
        """
        获取分块统计信息

        返回:
            包含平均块大小、最大/最小块大小等统计信息的字典
        """
        if not chunks:
            return {}

        token_counts = [estimate_tokens(c.text) for c in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "median_tokens": sorted(token_counts)[len(token_counts) // 2],
        }


# 便捷函数

def chunk_texts(
    pages: list[PDFPage],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[TextChunk]:
    """
    便捷函数：将 PDF 页面分块

    使用示例:
        from knowledge_base.pdf_parser import parse_pdfs
        from knowledge_base.chunker import chunk_texts

        pages = parse_pdfs(Path("./data/raw"))
        chunks = chunk_texts(pages, chunk_size=512, chunk_overlap=64)
    """
    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_pages(pages)
