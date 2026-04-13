"""
工具注册表模块

功能：
- 根据 tools_config.yaml 动态加载启用的工具
- 管理所有工具的注册和发现
- 提供工具列表和 OpenAI 格式的工具定义
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from .base import BaseTool

logger = logging.getLogger(__name__)


def load_tools_config(config_path: str | Path = "config/tools_config.yaml") -> dict:
    """
    加载工具配置文件

    参数:
        config_path: 配置文件路径

    返回:
        配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"工具配置文件不存在: {config_path}，使用默认配置")
        return {"tools": {}, "mcp": {"enabled": False}}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_tools(config_path: str | Path = "config/tools_config.yaml") -> list[BaseTool]:
    """
    加载所有启用的工具

    根据 tools_config.yaml 中的配置，动态加载并返回工具实例列表

    参数:
        config_path: 配置文件路径

    返回:
        启用的工具实例列表

    使用示例:
        tools = load_tools()
        for tool in tools:
            print(tool.name, tool.description)
    """
    config = load_tools_config(config_path)
    tools = []

    # 配置工具
    tools_config = config.get("tools", {})

    # 知识库搜索工具
    if tools_config.get("search_course_knowledge_base", {}).get("enabled", True):
        from .knowledge_base_search import KnowledgeBaseSearchTool

        kb_config = tools_config.get("search_course_knowledge_base", {})
        top_k = kb_config.get("default_top_k", 5)
        tools.append(KnowledgeBaseSearchTool(default_top_k=top_k))
        logger.info("已加载工具: search_course_knowledge_base")

    # arXiv 搜索工具
    if tools_config.get("search_arxiv", {}).get("enabled", True):
        from .arxiv_search import ArxivSearchTool

        arxiv_config = tools_config.get("search_arxiv", {})
        max_results = arxiv_config.get("default_max_results", 5)
        tools.append(ArxivSearchTool(default_max_results=max_results))
        logger.info("已加载工具: search_arxiv")

    # 网络搜索工具
    if tools_config.get("web_search", {}).get("enabled", True):
        from .web_search import WebSearchTool

        web_config = tools_config.get("web_search", {})
        max_results = web_config.get("default_max_results", 5)
        delay = web_config.get("rate_limit_delay", 1.0)
        tools.append(WebSearchTool(
            default_max_results=max_results,
            rate_limit_delay=delay
        ))
        logger.info("已加载工具: web_search")

    # 结构化推理工具
    if tools_config.get("sequential_thinking", {}).get("enabled", True):
        from .sequential_thinking import SequentialThinkingTool

        st_config = tools_config.get("sequential_thinking", {})
        max_thoughts = st_config.get("max_thoughts", 20)
        tools.append(SequentialThinkingTool(max_thoughts=max_thoughts))
        logger.info("已加载工具: sequential_thinking")

    # MCP 工具（可选）
    mcp_config = config.get("mcp", {})
    if mcp_config.get("enabled", False):
        try:
            from .mcp_bridge import MCPBridge

            mcp_bridge = MCPBridge(mcp_config)
            mcp_tools = mcp_bridge.load_mcp_tools()
            tools.extend(mcp_tools)
            logger.info(f"已加载 {len(mcp_tools)} 个 MCP 工具")
        except Exception as e:
            logger.warning(f"加载 MCP 工具失败: {e}")

    logger.info(f"总共加载 {len(tools)} 个工具")
    return tools


def get_tool_registry(tools: list[BaseTool]) -> dict[str, BaseTool]:
    """
    构建工具名称到工具实例的映射

    参数:
        tools: 工具列表

    返回:
        {工具名: 工具实例} 的字典
    """
    return {tool.name: tool for tool in tools}


def get_openai_tools(tools: list[BaseTool]) -> list[dict]:
    """
    获取 OpenAI 格式的工具定义列表

    参数:
        tools: 工具列表

    返回:
        OpenAI function calling 格式的工具定义列表
    """
    return [tool.to_openai_tool() for tool in tools]


class ToolRegistry:
    """
    工具注册表类

    封装工具管理和调用功能
    """

    def __init__(self, config_path: str | Path = "config/tools_config.yaml"):
        self.tools = load_tools(config_path)
        self._registry = get_tool_registry(self.tools)
        self.logger = logging.getLogger(__name__)

    def get(self, name: str) -> BaseTool | None:
        """根据名称获取工具"""
        return self._registry.get(name)

    def list_tools(self) -> list[str]:
        """获取所有工具名称列表"""
        return list(self._registry.keys())

    def get_openai_definitions(self) -> list[dict]:
        """获取 OpenAI 格式的工具定义"""
        return get_openai_tools(self.tools)

    async def execute(self, name: str, arguments: dict) -> str:
        """
        执行指定工具

        参数:
            name: 工具名称
            arguments: 工具参数

        返回:
            工具执行结果
        """
        tool = self.get(name)
        if tool is None:
            return f"错误：未找到工具 '{name}'"

        # 验证参数
        valid, error = tool.validate_parameters(arguments)
        if not valid:
            return f"参数错误: {error}"

        try:
            return await tool.execute(**arguments)
        except Exception as e:
            self.logger.error(f"工具 {name} 执行失败: {e}")
            return f"工具执行出错: {str(e)}"
