"""
响应格式化模块

功能：
- 使用 rich 库渲染 Markdown 格式的回答
- 代码高亮、表格渲染等
"""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box

# 全局控制台实例（使用标准输出）
console = Console()


class ResponseFormatter:
    """
    响应格式化器

    负责将 LLM 的文本回答渲染为美观的终端输出
    """

    def __init__(self):
        self.console = Console()

    def format_response(self, content: str, show_metadata: bool = False) -> None:
        """
        格式化并显示回答

        参数:
            content: LLM 生成的回答内容（Markdown 格式）
            show_metadata: 是否显示元数据
        """
        # 渲染 Markdown
        md = Markdown(content)
        self.console.print(md)

    def format_code(self, code: str, language: str = "python") -> None:
        """
        格式化并显示代码块

        参数:
            code: 代码内容
            language: 代码语言（用于语法高亮）
        """
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.console.print(syntax)

    def print_welcome(self, name: str = "", background: str | None = None) -> None:
        """打印欢迎信息"""
        title = "🎓 医学影像课程智能助教"
        subtitle = "输入你的问题，输入 /help 查看帮助\n"

        if name:
            subtitle = f"你好，{name}！{subtitle}"

        if background:
            bg_map = {
                "CS": "计算机/AI 背景",
                "BME": "生物医学工程背景",
                "beginner": "零基础",
            }
            bg_text = bg_map.get(background, background)
            subtitle += f"\n[当前配置: {bg_text}模式]"

        self.console.print(Panel.fit(
            f"[bold blue]{title}[/bold blue]\n\n[cyan]{subtitle}[/cyan]",
            border_style="blue",
        ))

    def print_help(self) -> None:
        """打印帮助信息"""
        help_text = """
[bold]可用命令：[/bold]

  /help          显示帮助信息
  /clear         清空对话历史
  /profile       查看/修改学生画像
  /exit, /quit   退出程序

[bold]使用提示：[/bold]

• 直接输入问题即可与智能助教对话
• Agent 会自动判断是否需要搜索知识库、arXiv 或网络
• 回答会标注信息来源（课件、论文、网页等）
        """
        self.console.print(Panel(help_text, title="帮助", border_style="green"))

    def print_tool_status(self, tool_name: str, status: str) -> None:
        """
        打印工具调用状态

        参数:
            tool_name: 工具名称
            status: 状态描述（如 "搜索中..."、"完成"）
        """
        icons = {
            "search_course_knowledge_base": "🔍",
            "search_arxiv": "📄",
            "web_search": "🌐",
            "sequential_thinking": "🤔",
        }
        icon = icons.get(tool_name, "🔧")

        # 使用淡色显示状态
        self.console.print(f"[dim]{icon} {tool_name}: {status}[/dim]")

    def print_separator(self) -> None:
        """打印分隔线"""
        self.console.print("─" * 60, style="dim")

    def print_info(self, message: str) -> None:
        """打印信息消息"""
        self.console.print(f"[dim]{message}[/dim]")

    def print_error(self, message: str) -> None:
        """打印错误消息"""
        self.console.print(f"[red]❌ {message}[/red]")

    def print_success(self, message: str) -> None:
        """打印成功消息"""
        self.console.print(f"[green]✓ {message}[/green]")


# 全局格式化器实例
formatter = ResponseFormatter()


# 便捷函数

def print_response(content: str) -> None:
    """便捷函数：打印回答"""
    formatter.format_response(content)


def print_welcome(name: str = "", background: str | None = None) -> None:
    """便捷函数：打印欢迎信息"""
    formatter.print_welcome(name, background)


def print_help() -> None:
    """便捷函数：打印帮助"""
    formatter.print_help()
