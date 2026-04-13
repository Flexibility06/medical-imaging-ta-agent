"""
网络搜索工具

功能：使用 DuckDuckGo 搜索互联网信息
特点：
- 免费，无需 API Key
- 支持速率限制配置
- 适合查找最新资讯、工具文档等
"""

import asyncio
import logging
from typing import Any

from duckduckgo_search import DDGS

from .base import BaseTool

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    网络搜索工具

    使用 DuckDuckGo 搜索互联网，适用于查找课程知识库和 arXiv 无法覆盖的信息
    """

    def __init__(
        self,
        default_max_results: int = 5,
        rate_limit_delay: float = 1.0,
    ):
        self.default_max_results = default_max_results
        self.rate_limit_delay = rate_limit_delay
        self.logger = logging.getLogger(__name__)
        self._last_search_time: float | None = None

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "在互联网上搜索信息。"
            "当需要查找最新资讯、工具文档、教程、或课程知识库和 arXiv 都无法覆盖的信息时使用。"
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询",
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大返回结果数（默认5）",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def _rate_limit(self):
        """简单的速率控制"""
        import time

        if self._last_search_time is not None:
            elapsed = time.time() - self._last_search_time
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)

        self._last_search_time = time.time()

    async def execute(self, **kwargs) -> str:
        """
        执行网络搜索

        参数:
            query: 搜索查询
            max_results: 最大返回结果数

        返回:
            格式化的搜索结果字符串
        """
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", self.default_max_results)

        if not query:
            return "错误：搜索查询不能为空"

        try:
            self.logger.info(f"搜索 Web: {query}")

            # 速率控制
            await self._rate_limit()

            # 执行搜索（DuckDuckGo 不是异步的，在线程中运行）
            loop = asyncio.get_event_loop()

            def _search():
                with DDGS() as ddgs:
                    results = ddgs.text(query, max_results=max_results)
                    return list(results)

            results = await loop.run_in_executor(None, _search)

            if not results:
                return f"未找到与 '{query}' 相关的网页"

            # 格式化结果
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get("title", "无标题")
                body = result.get("body", "无摘要")
                href = result.get("href", "无链接")

                # 截取摘要
                body = body[:250] + "..." if len(body) > 250 else body

                formatted = (
                    f"[{i}] {title}\n"
                    f"    {body}\n"
                    f"    URL: {href}"
                )
                formatted_results.append(formatted)

            return "\n\n".join(formatted_results)

        except Exception as e:
            self.logger.error(f"Web 搜索失败: {e}")
            return f"搜索出错: {str(e)}"
