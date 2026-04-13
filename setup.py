"""
知识库一键构建脚本

功能：串联 PDF 解析 → 文本分块 → Embedding 生成 → 索引构建 的完整流程
用法：python setup.py [--force]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# 确保可以导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import settings, Settings
from knowledge_base.pdf_parser import PDFParser, parse_pdfs
from knowledge_base.chunker import Chunker
from knowledge_base.embedder import Embedder
from knowledge_base.vector_store import VectorStore

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# Rich 控制台
console = Console()


def check_prerequisites() -> bool:
    """
    检查前置条件

    1. 检查 API 配置
    2. 检查 PDF 目录是否存在
    3. 检查是否有 PDF 文件
    """
    # 检查 API 配置
    missing = settings.validate()
    if missing:
        console.print(Panel.fit(
            f"[red]配置不完整，缺少: {', '.join(missing)}[/red]\n\n"
            "请按以下步骤配置:\n"
            "1. 复制 .env.example 为 .env\n"
            "2. 编辑 .env 文件，填入你的 API Key",
            title="配置错误",
            border_style="red",
        ))
        return False

    # 检查目录
    if not settings.RAW_PDF_DIR.exists():
        console.print(Panel.fit(
            f"[red]PDF 目录不存在: {settings.RAW_PDF_DIR}[/red]\n\n"
            "请创建目录并放入 PDF 课件:",
            title="目录错误",
            border_style="red",
        ))
        return False

    # 检查 PDF 文件
    pdf_files = list(settings.RAW_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        console.print(Panel.fit(
            f"[yellow]PDF 目录为空: {settings.RAW_PDF_DIR}[/yellow]\n\n"
            "请放入 PDF 课件后重试",
            title="无数据",
            border_style="yellow",
        ))
        return False

    return True


def display_summary(pages, chunks, cost_info):
    """显示处理摘要"""
    table = Table(title="处理摘要", show_header=True, header_style="bold magenta")
    table.add_column("项目", style="cyan")
    table.add_column("数值", style="green")

    table.add_row("PDF 页数", str(len(pages)))
    table.add_row("分块数量", str(len(chunks)))
    table.add_row("预估 Token 数", f"{cost_info['total_tokens']:,}")
    table.add_row("预估成本 (USD)", f"${cost_info['estimated_cost_usd']:.4f}")
    table.add_row("预估成本 (CNY)", f"¥{cost_info['estimated_cost_cny']:.4f}")

    console.print(table)


async def build_knowledge_base(force: bool = False):
    """
    构建知识库主流程

    参数:
        force: 是否强制重建（即使索引已存在）
    """
    console.print(Panel.fit(
        "[bold blue]医学影像课程知识库构建工具[/bold blue]",
        border_style="blue",
    ))

    # 检查前置条件
    if not check_prerequisites():
        return False

    # 检查是否已有索引
    if settings.INDEX_PATH.exists() and not force:
        console.print(Panel.fit(
            "[yellow]知识库索引已存在[/yellow]\n\n"
            f"路径: {settings.INDEX_PATH}\n\n"
            "如需重建，请使用 --force 参数",
            title="提示",
            border_style="yellow",
        ))
        return True

    # 确保目录存在
    settings.ensure_directories()

    # 步骤 1: 解析 PDF
    console.print("\n[bold]步骤 1/4:[/bold] 解析 PDF 文件...")
    parser = PDFParser()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("解析中...", total=None)
        pages = list(parser.parse_directory(settings.RAW_PDF_DIR))
        progress.update(task, description=f"✓ 解析完成: {len(pages)} 页")

    if not pages:
        console.print("[red]没有成功解析任何页面，请检查 PDF 文件[/red]")
        return False

    # 步骤 2: 文本分块
    console.print("\n[bold]步骤 2/4:[/bold] 文本分块...")
    chunker = Chunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    chunks = chunker.chunk_pages(pages)

    # 显示分块统计
    stats = chunker.get_chunk_stats(chunks)
    console.print(f"  ✓ 生成 {stats['total_chunks']} 个块")
    console.print(f"  • 平均大小: {stats['avg_tokens']:.0f} tokens")
    console.print(f"  • 大小范围: {stats['min_tokens']:.0f} - {stats['max_tokens']:.0f} tokens")

    # 步骤 3: 生成 Embeddings
    console.print("\n[bold]步骤 3/4:[/bold] 生成 Embeddings...")

    embedder = Embedder()

    # 显示成本估算
    cost_info = embedder.estimate_cost(chunks)
    display_summary(pages, chunks, cost_info)

    # 确认继续
    if cost_info['estimated_cost_usd'] > 1.0:  # 超过 $1 需要确认
        confirm = input("\n预估成本较高，是否继续? (y/n): ").strip().lower()
        if confirm != 'y':
            console.print("[yellow]已取消[/yellow]")
            return False

    # 已在 asyncio.run() 的事件循环内，不能再用 embed_chunks_sync（内部会 asyncio.run）
    embeddings, valid_chunks = await embedder.embed_chunks_with_progress(
        chunks, show_progress=True
    )

    if not embeddings:
        console.print("[red]Embedding 生成失败，请检查 API 配置[/red]")
        return False

    console.print(f"  ✓ 成功生成 {len(embeddings)} 个 embeddings")

    # 步骤 4: 构建并保存索引
    console.print("\n[bold]步骤 4/4:[/bold] 构建向量索引...")

    store = VectorStore()
    store.add(valid_chunks, embeddings)
    store.save(settings.INDEX_PATH, settings.CHUNKS_PATH)

    # 显示结果
    console.print(Panel.fit(
        f"[green]✓ 知识库构建完成！[/green]\n\n"
        f"索引文件: {settings.INDEX_PATH}\n"
        f"元数据文件: {settings.CHUNKS_PATH}\n"
        f"总计: {len(valid_chunks)} 个向量块",
        title="成功",
        border_style="green",
    ))

    return True


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="构建医学影像课程知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python setup.py              # 构建知识库（如果已存在则跳过）
  python setup.py --force      # 强制重建知识库
        """
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重建知识库（即使索引已存在）",
    )

    args = parser.parse_args()

    try:
        success = asyncio.run(build_knowledge_base(force=args.force))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print("\n[yellow]已取消[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]错误: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
