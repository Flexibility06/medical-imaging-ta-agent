"""
问答日志模块

功能：
- 记录每次问答的完整信息
- JSON Lines 格式，便于后续分析
- 包含工具调用记录
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import settings

logger = logging.getLogger(__name__)


class ChatLogger:
    """
    对话日志记录器

    记录每次问答的详细信息到 JSON Lines 文件
    """

    def __init__(self, log_dir: Path | None = None):
        self.log_dir = log_dir or settings.LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 日志文件路径（按日期命名）
        today = datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"chat_{today}.jsonl"

        self.logger = logging.getLogger(__name__)

    def log(
        self,
        session_id: str,
        background: str | None,
        user_message: str,
        tool_calls: list,
        assistant_response: str,
        total_duration_ms: int,
    ) -> None:
        """
        记录一次对话

        参数:
            session_id: 会话 ID
            background: 学生背景
            user_message: 用户消息
            tool_calls: 工具调用记录列表
            assistant_response: 助手回复
            total_duration_ms: 总耗时（毫秒）
        """
        try:
            # 限制字段长度，防止超大内容
            user_message_truncated = user_message[:1000] if len(user_message) > 1000 else user_message
            assistant_response_truncated = assistant_response[:200] + "..." if len(assistant_response) > 200 else assistant_response

            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "session_id": session_id,
                "student_background": background,
                "user_message": user_message_truncated,
                "tool_calls": [
                    {
                        "tool": tc.tool,
                        "args": tc.arguments,
                        "result_summary": tc.result[:100] + "..." if len(tc.result) > 100 else tc.result,
                        "duration_ms": tc.duration_ms,
                    }
                    for tc in tool_calls
                ],
                "assistant_response": assistant_response_truncated,
                "total_duration_ms": total_duration_ms,
            }

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            # 日志失败不应影响主流程
            self.logger.error(f"写入日志失败: {e}")

    def get_recent_logs(self, n: int = 10) -> list[dict]:
        """
        获取最近的 n 条日志

        用于分析学生常见问题等场景
        """
        if not self.log_file.exists():
            return []

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # 解析最近的 n 条
            recent = []
            for line in lines[-n:]:
                try:
                    recent.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            return recent
        except Exception as e:
            self.logger.error(f"读取日志失败: {e}")
            return []


class SimpleLogger:
    """
    简易日志记录器（用于调试）

    只输出到控制台，不写入文件
    """

    def log(self, **kwargs) -> None:
        """空实现，不记录"""
        pass
