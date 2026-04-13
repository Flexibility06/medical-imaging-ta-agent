"""
PDF 解析模块

功能：批量解析 PDF 课件，提取每页的文本内容
设计要点：
- 使用 PyMuPDF (fitz) 提取文本
- 保留文件路径、文件名、页码等元数据
- 处理解析失败的文件，记录错误但不中断
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class PDFPage:
    """
    PDF 页面数据类

    属性:
        file_path: PDF 文件的完整路径
        file_name: PDF 文件名（不含路径）
        page_number: 页码（从 1 开始）
        text: 提取的文本内容
    """
    file_path: Path
    file_name: str
    page_number: int
    text: str

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "file_path": str(self.file_path),
            "file_name": self.file_name,
            "page_number": self.page_number,
            "text": self.text,
        }


class PDFParser:
    """
    PDF 解析器类

    负责将 PDF 文件解析为结构化的页面文本数据
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_pdf(self, pdf_path: Path) -> list[PDFPage]:
        """
        解析单个 PDF 文件

        参数:
            pdf_path: PDF 文件路径

        返回:
            PDFPage 对象列表（每页一个）

        异常:
            解析失败时记录错误并返回空列表
        """
        pages = []
        file_name = pdf_path.name

        try:
            # 打开 PDF 文档
            doc = fitz.open(pdf_path)
            self.logger.info(f"正在解析: {file_name} ({len(doc)} 页)")

            # 遍历每一页
            for page_num in range(len(doc)):
                page = doc[page_num]

                # 提取文本
                text = page.get_text()

                # 跳过空白页
                if not text.strip():
                    continue

                # 创建页面数据对象
                pdf_page = PDFPage(
                    file_path=pdf_path.resolve(),
                    file_name=file_name,
                    page_number=page_num + 1,  # 页码从 1 开始
                    text=text.strip(),
                )
                pages.append(pdf_page)

            doc.close()
            self.logger.info(f"✓ {file_name}: 提取 {len(pages)} 页文本")

        except Exception as e:
            self.logger.error(f"❌ 解析失败 {file_name}: {e}")
            # 解析失败返回空列表，不中断整体流程

        return pages

    def parse_directory(
        self,
        directory: Path,
        pattern: str = "*.pdf",
    ) -> Iterator[PDFPage]:
        """
        批量解析目录中的所有 PDF 文件

        参数:
            directory: PDF 文件所在目录
            pattern: 文件匹配模式，默认 "*.pdf"

        返回:
            PDFPage 生成器，逐个产出页面

        使用示例:
            parser = PDFParser()
            for page in parser.parse_directory(Path("./data/raw")):
                print(page.file_name, page.page_number)
        """
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")

        # 获取所有 PDF 文件
        pdf_files = list(directory.glob(pattern))
        pdf_files.sort()

        self.logger.info(f"找到 {len(pdf_files)} 个 PDF 文件")

        total_pages = 0
        for pdf_path in pdf_files:
            pages = self.parse_pdf(pdf_path)
            total_pages += len(pages)
            yield from pages

        self.logger.info(f"总共解析 {total_pages} 页文本")

    def parse_and_save(
        self,
        input_dir: Path,
        output_path: Path,
        pattern: str = "*.pdf",
    ) -> int:
        """
        解析 PDF 并保存为中间格式（JSON Lines）

        参数:
            input_dir: PDF 输入目录
            output_path: 输出文件路径（.jsonl）
            pattern: 文件匹配模式

        返回:
            解析的页面总数
        """
        import json

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for page in self.parse_directory(input_dir, pattern):
                f.write(json.dumps(page.to_dict(), ensure_ascii=False) + "\n")
                count += 1

        self.logger.info(f"已保存到: {output_path} ({count} 页)")
        return count


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量（粗略估算）

    中文：约 1 字符 ≈ 1 token
    英文：约 4 字符 ≈ 1 token

    这是一个简化估算，实际 token 数取决于具体分词器
    """
    # 简单估算：中文每个字符算 1 token，英文每个单词算 1 token
    import re

    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_words = len(re.findall(r"[a-zA-Z]+", text))
    other_chars = len(text) - chinese_chars - sum(len(w) for w in re.findall(r"[a-zA-Z]+", text))

    # 粗略估算：中文字符 + 英文单词 + 其他字符/2
    return chinese_chars + english_words + other_chars // 2


# 便捷函数

def parse_pdfs(input_dir: Path, pattern: str = "*.pdf") -> list[PDFPage]:
    """
    便捷函数：解析目录中的所有 PDF

    使用示例:
        pages = parse_pdfs(Path("./data/raw"))
        for p in pages:
            print(f"{p.file_name} p.{p.page_number}")
    """
    parser = PDFParser()
    return list(parser.parse_directory(input_dir, pattern))
