"""
arXiv 论文搜索工具

功能：搜索 arXiv 学术论文
特点：
- 免费，无需 API Key
- 返回论文标题、作者、摘要、链接等信息
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import arxiv

from .base import BaseTool

logger = logging.getLogger(__name__)


class ArxivSearchTool(BaseTool):
    """
    arXiv 论文搜索工具

    搜索 arXiv 上的学术论文，适用于查找最新研究进展和参考文献
    """

    def __init__(self, default_max_results: int = 5):
        self.default_max_results = default_max_results
        self.logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "search_arxiv"

    @property
    def description(self) -> str:
        return (
            "搜索 arXiv 上的学术论文。"
            "当学生询问最新研究进展、需要论文参考、或课程知识库中没有足够信息时使用。"
            "建议使用英文关键词搜索以获得更好结果。"
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词（建议使用英文）",
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大返回结果数（默认5）",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def _execute(self, **kwargs) -> str:
        """
        执行 arXiv 搜索

        参数:
            query: 搜索关键词
            max_results: 最大返回结果数

        返回:
            格式化的搜索结果字符串
        """
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", self.default_max_results)

        if not query:
            return "错误：搜索关键词不能为空"

        try:
            self.logger.info(f"搜索 arXiv: {query}")

            # 创建搜索客户端
            client = arxiv.Client()

            # 构建搜索查询
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            # 执行搜索（添加超时控制）
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                results = await asyncio.wait_for(
                    loop.run_in_executor(pool, lambda: list(client.results(search))),
                    timeout=60.0
                )

            if not results:
                return f"未找到与 '{query}' 相关的论文"

            # 格式化结果
            formatted_results = []
            for i, paper in enumerate(results, 1):
                # 截取摘要前 300 字符
                abstract = paper.summary[:300] + "..." if len(paper.summary) > 300 else paper.summary

                # 格式化作者列表
                authors = ", ".join(str(a) for a in paper.authors[:3])
                if len(paper.authors) > 3:
                    authors += " et al."

                formatted = (
                    f"[{i}] Title: {paper.title}\n"
                    f"    Authors: {authors}\n"
                    f"    Published: {paper.published.strftime('%Y-%m-%d')}\n"
                    f"    Abstract: {abstract}\n"
                    f"    URL: {paper.entry_id}"
                )
                formatted_results.append(formatted)

            return "\n\n".join(formatted_results)

        except asyncio.TimeoutError:
            return "arXiv 搜索超时，请稍后重试"
        except Exception as e:
            self.logger.error(f"arXiv 搜索失败: {e}")
            return f"搜索出错: {str(e)}"
