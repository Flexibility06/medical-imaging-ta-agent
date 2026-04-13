"""
全局配置模块

功能：集中管理所有环境变量和配置项，支持从 .env 文件加载
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    """配置类，所有配置项都从这里读取"""

    # 项目路径
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_PDF_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    INDEX_DIR = DATA_DIR / "index"
    LOGS_DIR = PROJECT_ROOT / "logs"

    # LLM 配置
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

    # Embedding 配置（可与 LLM 不同）
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "") or LLM_API_KEY
    EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "") or LLM_BASE_URL
    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME", "text-embedding-3-small"
    )
    # 单次请求最多送入的文本条数（SiliconFlow 等为 64；OpenAI 常见为 100+）
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

    # 向量数据库配置
    INDEX_PATH = INDEX_DIR / "faiss_index.bin"
    CHUNKS_PATH = INDEX_DIR / "chunks.pkl"

    # 学生画像配置
    PROFILE_PATH = DATA_DIR / "student_profile.json"

    # 检索配置
    DEFAULT_TOP_K = 5
    CHUNK_SIZE = 512  # tokens（估算）
    CHUNK_OVERLAP = 64

    # Agent 配置
    MAX_TOOL_CALLS = 10  # 单次对话最大工具调用轮次
    MAX_HISTORY_ROUNDS = 20  # 保留的对话轮数

    @classmethod
    def ensure_directories(cls) -> None:
        """确保所有必要的目录都存在"""
        for dir_path in [
            cls.RAW_PDF_DIR,
            cls.PROCESSED_DIR,
            cls.INDEX_DIR,
            cls.LOGS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls) -> list[str]:
        """
        验证配置是否完整
        返回缺失的配置项列表（空列表表示验证通过）
        """
        missing = []
        if not cls.LLM_API_KEY:
            missing.append("LLM_API_KEY")
        if not cls.LLM_BASE_URL:
            missing.append("LLM_BASE_URL")
        return missing


# 全局配置实例
settings = Settings()
