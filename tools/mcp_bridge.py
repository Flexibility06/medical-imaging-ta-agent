"""
MCP 协议桥接器

功能：连接 MCP (Model Context Protocol) Server，将 MCP 工具转换为本地工具
设计要点：
- MCP 为可选依赖，未安装时静默禁用
- 支持通过 stdio 连接到 MCP Server
- 动态发现 MCP Server 提供的工具
"""

import logging
from typing import Any

# 条件导入：mcp SDK 为可选依赖
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from .base import BaseTool

logger = logging.getLogger(__name__)


class MCPToolWrapper(BaseTool):
    """
    MCP 工具包装器

    将 MCP 工具包装为本地 BaseTool 接口
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters_schema: dict,
        session: Any,  # ClientSession
    ):
        self._name = name
        self._description = description
        self._parameters_schema = parameters_schema
        self._session = session

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict:
        return self._parameters_schema

    async def _execute(self, **kwargs) -> str:
        """通过 MCP 调用工具"""
        try:
            result = await self._session.call_tool(self._name, arguments=kwargs)
            # 提取文本内容
            content = []
            for item in result.content:
                if hasattr(item, 'text'):
                    content.append(item.text)
            return "\n".join(content) if content else str(result)
        except Exception as e:
            return f"MCP 工具调用失败: {str(e)}"


class MCPBridge:
    """
    MCP 桥接器

    管理 MCP Server 连接和工具转换
    """

    def __init__(self, mcp_config: dict):
        self.config = mcp_config
        self.sessions: list[Any] = []  # ClientSession 列表
        self.logger = logging.getLogger(__name__)

        if not MCP_AVAILABLE:
            self.logger.info("MCP SDK 未安装，MCP 功能不可用")

    async def connect_servers(self) -> list[BaseTool]:
        """
        连接到所有配置的 MCP Server

        返回:
            所有 MCP Server 提供的工具列表
        """
        if not MCP_AVAILABLE:
            return []

        all_tools = []
        servers = self.config.get("servers", [])

        for server_config in servers:
            try:
                tools = await self._connect_server(server_config)
                all_tools.extend(tools)
            except Exception as e:
                self.logger.error(f"连接 MCP Server {server_config.get('name', 'unknown')} 失败: {e}")

        return all_tools

    async def _connect_server(self, server_config: dict) -> list[BaseTool]:
        """连接单个 MCP Server"""
        name = server_config.get("name", "unnamed")
        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env", {})

        self.logger.info(f"连接 MCP Server: {name}")

        # 创建 Server 参数
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )

        # 建立连接
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # 初始化
                await session.initialize()

                # 获取工具列表
                tools_response = await session.list_tools()

                # 转换为本地工具
                wrapped_tools = []
                for tool in tools_response.tools:
                    wrapped = MCPToolWrapper(
                        name=tool.name,
                        description=tool.description or "",
                        parameters_schema=tool.inputSchema,
                        session=session,
                    )
                    wrapped_tools.append(wrapped)

                self.sessions.append(session)
                self.logger.info(f"MCP Server {name}: 发现 {len(wrapped_tools)} 个工具")
                return wrapped_tools

    def load_mcp_tools(self) -> list[BaseTool]:
        """
        同步加载 MCP 工具（供非异步代码调用）

        注意：这个方法会创建新的事件循环
        """
        if not MCP_AVAILABLE:
            return []

        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环已在运行，需要特殊处理
                self.logger.warning("事件循环已在运行，无法同步加载 MCP 工具")
                return []
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(self.connect_servers())
        finally:
            loop.close()

    async def close(self):
        """关闭所有 MCP 连接"""
        # MCP SDK 会自动管理连接生命周期
        self.sessions.clear()
