"""
结构化推理工具（Sequential Thinking）

功能：为 Agent 提供结构化的多步推理能力
设计要点：
- 模仿 Anthropic 的 Sequential Thinking MCP
- 支持逐步思考、回顾、修正
- 纯内部逻辑，不调用外部 API
- 状态在对话期间保持
"""

from typing import Any

from .base import BaseTool


class SequentialThinkingTool(BaseTool):
    """
    结构化多步推理工具

    让 Agent 在面对复杂问题时能显式地进行分步推理，并允许修正之前的思路
    """

    def __init__(self, max_thoughts: int = 20):
        self.max_thoughts = max_thoughts
        self.thoughts: list[dict] = []  # 思考链

    @property
    def name(self) -> str:
        return "sequential_thinking"

    @property
    def description(self) -> str:
        return (
            "用于复杂问题的分步推理。"
            "当面对需要多步分析的问题（如调试代码逻辑、分析医学影像处理流程、设计实验方案等）时，"
            "使用此工具逐步思考。每次调用记录一个思考步骤，可以回顾和修正之前的步骤。"
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "当前步骤的思考内容",
                },
                "thought_number": {
                    "type": "integer",
                    "description": "当前步骤编号（从1开始）",
                },
                "total_thoughts": {
                    "type": "integer",
                    "description": "预计总步骤数（可在过程中调整）",
                },
                "next_thought_needed": {
                    "type": "boolean",
                    "description": "是否需要继续下一步思考",
                },
                "is_revision": {
                    "type": "boolean",
                    "description": "是否是对之前步骤的修正",
                    "default": False,
                },
                "revises_thought": {
                    "type": "integer",
                    "description": "修正的是哪个步骤编号",
                },
            },
            "required": [
                "thought",
                "thought_number",
                "total_thoughts",
                "next_thought_needed",
            ],
        }

    async def _execute(self, **kwargs) -> str:
        """
        执行思考步骤

        参数:
            thought: 当前步骤的思考内容
            thought_number: 当前步骤编号
            total_thoughts: 预计总步骤数
            next_thought_needed: 是否继续下一步
            is_revision: 是否为修正
            revises_thought: 修正的目标步骤

        返回:
            当前思考链的格式化摘要
        """
        thought = kwargs.get("thought", "")
        thought_number = kwargs.get("thought_number", 0)
        total_thoughts = kwargs.get("total_thoughts", 0)
        next_thought_needed = kwargs.get("next_thought_needed", False)
        is_revision = kwargs.get("is_revision", False)
        revises_thought = kwargs.get("revises_thought")

        # 验证步骤编号
        if thought_number < 1 or thought_number > self.max_thoughts:
            return f"错误：步骤编号必须在 1-{self.max_thoughts} 之间"

        # 创建思考记录
        thought_record = {
            "thought": thought,
            "thought_number": thought_number,
            "total_thoughts": total_thoughts,
            "next_thought_needed": next_thought_needed,
            "is_revision": is_revision,
            "revises_thought": revises_thought,
        }

        # 处理修正或添加新步骤
        if is_revision and revises_thought:
            # 修正现有步骤
            for i, t in enumerate(self.thoughts):
                if t["thought_number"] == revises_thought:
                    self.thoughts[i] = thought_record
                    break
            else:
                # 如果没找到要修正的步骤，直接添加
                self.thoughts.append(thought_record)
        else:
            # 添加新步骤
            # 检查是否已存在该编号的步骤
            existing_idx = None
            for i, t in enumerate(self.thoughts):
                if t["thought_number"] == thought_number:
                    existing_idx = i
                    break

            if existing_idx is not None:
                self.thoughts[existing_idx] = thought_record
            else:
                self.thoughts.append(thought_record)

        # 按步骤编号排序
        self.thoughts.sort(key=lambda x: x["thought_number"])

        # 构建返回结果
        return self._format_thoughts(next_thought_needed)

    def _format_thoughts(self, is_final: bool) -> str:
        """格式化思考链"""
        lines = []

        if is_final:
            lines.append("【推理完成】完整思考链：")
        else:
            lines.append(f"【思考中】步骤 {self.thoughts[-1]['thought_number']}/{self.thoughts[-1]['total_thoughts']}：")

        lines.append("")

        for t in self.thoughts:
            prefix = "📝"
            if t.get("is_revision"):
                prefix = "🔄"
            elif t["thought_number"] == self.thoughts[-1]["thought_number"] and not is_final:
                prefix = "👉"

            lines.append(f"{prefix} 步骤 {t['thought_number']}: {t['thought']}")

        return "\n".join(lines)

    def reset(self):
        """重置思考链"""
        self.thoughts = []
