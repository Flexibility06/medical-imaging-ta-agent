"""
工具基类定义模块

功能：定义所有工具的抽象基类和通用接口
设计要点：
- 统一的工具接口规范
- 支持转换为 OpenAI function calling 格式
- 便于后续扩展新的工具
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """
    所有工具的抽象基类

    子类必须实现 name、description、parameters_schema 属性和 execute 方法
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        工具名称

        供 LLM function calling 使用，必须是唯一的、描述性的英文标识符
        示例："search_course_knowledge_base"
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        工具功能描述

        LLM 根据此描述决定是否调用该工具。
        描述应该清晰说明：
        1. 工具的功能
        2. 在什么情况下应该使用
        3. 返回什么类型的结果
        """
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict:
        """
        JSON Schema 格式的参数定义

        遵循 OpenAI function calling 的参数格式：
        {
            "type": "object",
            "properties": {
                "param_name": {
                    "type": "string",
                    "description": "参数描述"
                }
            },
            "required": ["param_name"]
        }
        """
        ...

    # 工具结果最大长度限制（约 8000 字符 ≈ 2000 tokens）
    MAX_RESULT_LENGTH = 20000

    async def execute(self, **kwargs) -> str:
        """
        执行工具

        参数:
            **kwargs: 工具所需的参数，由 LLM 根据 parameters_schema 生成

        返回:
            工具执行结果的字符串表示
            建议格式清晰、包含关键信息

        异常:
            工具执行出错时应该捕获异常并返回错误信息字符串，而不是抛出异常
        """
        result = await self._execute(**kwargs)
        # 截断超长结果
        if len(result) > self.MAX_RESULT_LENGTH:
            result = result[:self.MAX_RESULT_LENGTH] + "\n... (结果已截断)"
        return result

    @abstractmethod
    async def _execute(self, **kwargs) -> str:
        """
        实际执行工具的逻辑（子类实现）

        返回:
            工具执行结果的字符串表示
        """
        ...

    def to_openai_tool(self) -> dict:
        """
        转换为 OpenAI function calling 格式

        返回:
            OpenAI 工具定义字典
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def validate_parameters(self, params: dict) -> tuple[bool, str]:
        """
        验证参数是否符合 schema（基础验证）

        参数:
            params: 待验证的参数字典

        返回:
            (是否有效, 错误信息) 元组
        """
        schema = self.parameters_schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # 检查必填参数
        for param in required:
            if param not in params:
                return False, f"缺少必填参数: {param}"

        # 检查参数类型（简单验证）
        for param, value in params.items():
            if param in properties:
                expected_type = properties[param].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    return (
                        False,
                        f"参数 {param} 类型错误: 期望 {expected_type}, 实际 {type(value).__name__}",
                    )

        return True, ""

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """检查值是否符合期望类型"""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        if expected_type not in type_map:
            return True  # 未知类型跳过验证

        return isinstance(value, type_map[expected_type])


class ToolResult:
    """
    工具执行结果封装类

    用于统一工具返回格式，包含成功/失败状态、结果内容、元数据等
    """

    def __init__(
        self,
        content: str,
        success: bool = True,
        metadata: dict | None = None,
    ):
        self.content = content
        self.success = success
        self.metadata = metadata or {}

    def __str__(self) -> str:
        return self.content

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "content": self.content,
            "success": self.success,
            "metadata": self.metadata,
        }
