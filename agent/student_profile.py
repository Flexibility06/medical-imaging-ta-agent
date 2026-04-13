"""
学生画像管理模块

功能：
- 交互式采集学生背景信息
- 持久化存储为 JSON
- 根据背景提供差异化 System Prompt 片段
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import settings

logger = logging.getLogger(__name__)

# 学生背景类型
BackgroundType = Literal["CS", "BME", "beginner", None]


@dataclass
class StudentProfile:
    """
    学生画像数据类

    属性:
        name: 学生姓名（可选）
        background: 背景类型 CS/BME/beginner
        interests: 感兴趣的领域（可选）
        goals: 学习目标（可选）
    """
    name: str = ""
    background: BackgroundType = None
    interests: str = ""
    goals: str = ""

    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StudentProfile":
        """从字典创建"""
        return cls(**data)


class ProfileManager:
    """
    学生画像管理器

    负责画像的采集、加载、保存
    """

    def __init__(self, profile_path: Path | None = None):
        self.profile_path = profile_path or settings.PROFILE_PATH
        self.logger = logging.getLogger(__name__)

    def interactive_setup(self) -> StudentProfile:
        """
        交互式采集学生信息

        在 CLI 启动时调用，引导用户填写背景信息
        """
        print("\n" + "=" * 50)
        print("🎓 欢迎使用医学影像课程智能助教！")
        print("=" * 50)
        print("\n为了给你提供更个性化的回答，请告诉我一些你的背景信息：")
        print("（直接按回车可跳过）\n")

        # 姓名
        name = input("1. 你的名字（可选）: ").strip()

        # 背景选择
        print("\n2. 你的专业背景：")
        print("   [1] 计算机/AI 背景 (CS)")
        print("   [2] 生物医学工程/医学背景 (BME)")
        print("   [3] 零基础/其他 (beginner)")
        print("   [4] 暂不指定")

        bg_choice = input("   请选择 (1-4): ").strip()
        background_map = {
            "1": "CS",
            "2": "BME",
            "3": "beginner",
            "4": None,
        }
        background = background_map.get(bg_choice, None)

        # 兴趣领域
        print("\n3. 你对哪些领域特别感兴趣？（如：CT影像、MRI、深度学习、分割等）")
        interests = input("   兴趣领域（可选）: ").strip()

        # 学习目标
        print("\n4. 你的学习目标是什么？")
        goals = input("   学习目标（可选）: ").strip()

        # 创建画像
        profile = StudentProfile(
            name=name,
            background=background,
            interests=interests,
            goals=goals,
        )

        # 保存
        self.save(profile)

        print("\n" + "=" * 50)
        print(f"✅ 已保存你的画像！")
        if background:
            bg_desc = {
                "CS": "计算机/AI 背景",
                "BME": "生物医学工程/医学背景",
                "beginner": "零基础",
            }
            print(f"   背景: {bg_desc.get(background, background)}")
        print("=" * 50 + "\n")

        return profile

    def load(self) -> StudentProfile | None:
        """
        从文件加载学生画像

        返回:
            StudentProfile 对象，如果不存在则返回 None
        """
        if not self.profile_path.exists():
            return None

        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return StudentProfile.from_dict(data)
        except Exception as e:
            self.logger.error(f"加载画像失败: {e}")
            return None

    def save(self, profile: StudentProfile) -> None:
        """保存学生画像到文件"""
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)

        self.logger.info(f"画像已保存: {self.profile_path}")

    def exists(self) -> bool:
        """检查画像文件是否存在"""
        return self.profile_path.exists()

    def get_or_create(self) -> StudentProfile:
        """
        获取现有画像，如果不存在则交互式创建

        这是启动时的主要入口
        """
        profile = self.load()
        if profile is None:
            profile = self.interactive_setup()
        return profile


def get_background_prompt_suffix(background: BackgroundType) -> str:
    """
    根据背景获取 System Prompt 附加片段

    用于动态拼接差异化的 System Prompt
    """
    suffixes = {
        "CS": """
【学生背景：计算机/AI】
该学生具有计算机科学背景。在解释时：
- 重点阐述医学影像专业概念（如 DICOM 格式、HU 值、影像模态差异、解剖结构等）
- 编程和AI相关内容可以适当简略
- 当推荐学习资源时，侧重医学影像基础知识
""",
        "BME": """
【学生背景：生物医学工程】
该学生具有生物医学工程背景。在解释时：
- 重点阐述 AI/编程相关概念（如 CNN 原理、Python 数据处理、模型训练流程等）
- 医学影像领域知识可以适当简略
- 当推荐学习资源时，侧重编程和深度学习入门
""",
        "beginner": """
【学生背景：零基础】
该学生为跨学科零基础。在解释时：
- 使用通俗语言和类比
- 提供更详细的步骤拆解
- 适当降低技术深度
- 推荐基础先修资源
""",
    }

    return suffixes.get(background, "")
