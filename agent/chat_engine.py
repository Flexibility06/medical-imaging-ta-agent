"""
核心对话引擎（Agent Loop）

功能：
- 实现 Agent 主循环：接收用户输入 → 调用 LLM → 处理工具调用 → 返回结果
- 支持多轮工具调用
- 管理对话历史
- 工具调用状态展示

Agent Loop 流程：
    用户输入
        ↓
    构建消息列表（system prompt + 对话历史 + 用户消息）
        ↓
    ┌─→ 调用 LLM（附带工具定义）
    │       ↓
    │   LLM 响应是否包含 tool_calls?
    │       ├─ 是 → 执行工具 → 将结果追加到消息列表 → 回到循环顶部
    │       └─ 否 → 提取文本回答 → 输出给用户
    └───────┘
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.llm_client import get_llm_client, LLMError
from utils.logger import ChatLogger
from config.prompts import build_system_prompt
from tools import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """对话消息数据类"""
    role: str  # system / user / assistant / tool
    content: str | None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # for tool messages

    def to_dict(self) -> dict:
        """转换为 OpenAI 消息格式"""
        msg = {"role": self.role, "content": self.content}

        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name

        return msg


@dataclass
class ToolCallRecord:
    """工具调用记录"""
    tool: str
    arguments: dict
    result: str
    duration_ms: int


@dataclass
class ChatResponse:
    """对话响应数据类"""
    content: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    usage: dict | None = None
    duration_ms: int = 0


class ChatEngine:
    """
    对话引擎类

    实现 Agent Loop，管理对话流程和工具调用
    """

    def __init__(
        self,
        student_background: str | None = None,
        max_tool_calls: int = 10,
        max_history: int = 20,
    ):
        """
        初始化对话引擎

        参数:
            student_background: 学生背景（CS/BME/beginner）
            max_tool_calls: 单次对话最大工具调用次数
            max_history: 保留的最大对话轮数
        """
        self.llm_client = get_llm_client()
        self.tool_registry = ToolRegistry()
        self.chat_logger = ChatLogger()

        self.max_tool_calls = max_tool_calls
        self.max_history = max_history
        self.student_background = student_background

        # 构建 system prompt
        self.system_prompt = build_system_prompt(student_background)

        # 对话历史
        self.history: list[ChatMessage] = []

        # 会话 ID（用于日志）
        self.session_id = str(uuid.uuid4())[:8]

        self.logger = logging.getLogger(__name__)

    def _build_messages(self, user_message: str) -> list[dict]:
        """
        构建发送给 LLM 的消息列表

        包括：system prompt + 历史对话 + 当前用户消息
        """
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # 添加历史消息
        for msg in self.history:
            messages.append(msg.to_dict())

        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})

        # 保存用户消息到历史（放在这里确保顺序正确）
        self._add_to_history("user", user_message)

        return messages

    def _add_to_history(self, role: str, content: str | None, **kwargs) -> None:
        """添加消息到历史"""
        msg = ChatMessage(role=role, content=content, **kwargs)
        self.history.append(msg)

        # 限制历史长度，但要保持 tool_calls 和 tool 消息的完整性
        if len(self.history) > self.max_history * 2:  # *2 因为每轮有 assistant + user
            # 找到第一个有效的截断点（确保不切断 tool 调用对）
            cutoff_idx = len(self.history) - self.max_history * 2

            # 如果 cutoff 点前一个是 assistant 且有 tool_calls，向前调整
            while cutoff_idx > 0 and cutoff_idx < len(self.history):
                prev_msg = self.history[cutoff_idx - 1]
                if prev_msg.role == "assistant" and prev_msg.tool_calls:
                    cutoff_idx += 1
                    # 继续跳过所有相关的 tool 消息
                    while cutoff_idx < len(self.history) and self.history[cutoff_idx].role == "tool":
                        cutoff_idx += 1
                else:
                    break

            self.history = self.history[cutoff_idx:]

    async def chat(
        self,
        user_message: str,
        show_tool_status: bool = True,
    ) -> ChatResponse:
        """
        主对话方法（Agent Loop）

        参数:
            user_message: 用户输入
            show_tool_status: 是否在终端显示工具调用状态

        返回:
            ChatResponse 对象
        """
        import time
        start_time = time.time()

        # 重置 sequential thinking 工具状态（每次新查询开始时）
        thinking_tool = self.tool_registry.get("sequential_thinking")
        if thinking_tool and hasattr(thinking_tool, 'reset'):
            thinking_tool.reset()

        # 检查空输入
        user_message = user_message.strip()
        if not user_message:
            return ChatResponse(content="请输入有效的问题。", duration_ms=0)

        tool_call_records: list[ToolCallRecord] = []
        messages = self._build_messages(user_message)

        # 获取工具定义
        tools = self.tool_registry.get_openai_definitions()

        # Agent Loop
        for round_num in range(self.max_tool_calls):
            # 调用 LLM
            try:
                result = await self.llm_client.chat_completion(
                    messages=messages,
                    tools=tools if tools else None,
                    temperature=0.7,
                )
            except LLMError as e:
                return ChatResponse(
                    content=f"调用 LLM 失败: {e}",
                    tool_calls=tool_call_records,
                    duration_ms=int((time.time() - start_time) * 1000),
                )

            # 检查是否有工具调用
            if not result.get("tool_calls"):
                # 没有工具调用，直接返回结果
                assistant_content = result.get("content") or ""

                # 保存助手回复到历史（用户消息已在 _build_messages 中保存）
                self._add_to_history("assistant", assistant_content)

                # 记录日志
                total_duration = int((time.time() - start_time) * 1000)
                self.chat_logger.log(
                    session_id=self.session_id,
                    background=self.student_background,
                    user_message=user_message,
                    tool_calls=tool_call_records,
                    assistant_response=assistant_content,
                    total_duration_ms=total_duration,
                )

                return ChatResponse(
                    content=assistant_content,
                    tool_calls=tool_call_records,
                    usage=result.get("usage"),
                    duration_ms=total_duration,
                )

            # 有工具调用，需要处理
            tool_calls = result["tool_calls"]

            # 添加 assistant 消息（包含 tool_calls）到消息列表和历史
            assistant_msg = ChatMessage(
                role="assistant",
                content=None,
                tool_calls=tool_calls
            )
            self.history.append(assistant_msg)
            messages.append(assistant_msg.to_dict())

            # 执行每个工具调用
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                import json
                try:
                    arguments = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    arguments = {}

                tool_call_id = tool_call["id"]

                # 显示状态
                if show_tool_status:
                    status_icon = self._get_tool_status_icon(tool_name)
                    print(f"{status_icon} 正在调用 {tool_name}...")

                # 执行工具
                tool_start = time.time()
                try:
                    tool_result = await self.tool_registry.execute(tool_name, arguments)
                except Exception as e:
                    tool_result = f"工具执行出错: {e}"
                tool_duration = int((time.time() - tool_start) * 1000)

                # 记录工具调用
                tool_call_records.append(ToolCallRecord(
                    tool=tool_name,
                    arguments=arguments,
                    result=tool_result[:200] + "..." if len(tool_result) > 200 else tool_result,
                    duration_ms=tool_duration,
                ))

                # 添加工具结果到消息列表和历史
                tool_msg = ChatMessage(
                    role="tool",
                    content=tool_result,
                    tool_call_id=tool_call_id,
                    name=tool_name
                )
                self.history.append(tool_msg)
                messages.append(tool_msg.to_dict())

        # 超过最大工具调用次数，强制返回
        error_content = "工具调用次数过多，请简化您的问题。"
        self._add_to_history("assistant", error_content)

        total_duration = int((time.time() - start_time) * 1000)
        self.chat_logger.log(
            session_id=self.session_id,
            background=self.student_background,
            user_message=user_message,
            tool_calls=tool_call_records,
            assistant_response=error_content,
            total_duration_ms=total_duration,
        )

        return ChatResponse(
            content=error_content,
            tool_calls=tool_call_records,
            duration_ms=total_duration,
        )

    def _get_tool_status_icon(self, tool_name: str) -> str:
        """获取工具状态图标"""
        icons = {
            "search_course_knowledge_base": "🔍",
            "search_arxiv": "📄",
            "web_search": "🌐",
            "sequential_thinking": "🤔",
        }
        return icons.get(tool_name, "🔧")

    def clear_history(self) -> None:
        """清空对话历史"""
        self.history = []
        # 重置 sequential thinking 工具状态
        thinking_tool = self.tool_registry.get("sequential_thinking")
        if thinking_tool and hasattr(thinking_tool, 'reset'):
            thinking_tool.reset()
        self.logger.info("对话历史已清空")

    def get_history_summary(self) -> str:
        """获取对话历史摘要"""
        user_msgs = [m for m in self.history if m.role == "user"]
        return f"当前会话共 {len(user_msgs)} 轮对话"
