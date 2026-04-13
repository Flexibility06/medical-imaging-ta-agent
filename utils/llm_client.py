"""
LLM 客户端模块

功能：封装 OpenAI 兼容 API 的调用，包括：
1. 聊天补全 (chat completion)
2. 文本嵌入 (embedding)
3. 工具调用 (tool calling)

设计要点：
- 支持从环境变量读取配置
- 内置重试机制和错误处理
- 统一的异步接口
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import openai
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import settings


class LLMClient:
    """
    LLM 客户端类

    封装所有与 LLM API 的交互，提供统一的调用接口
    """

    def __init__(self):
        # 创建 LLM 客户端实例
        self.llm_client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES,
        )

        # 创建 Embedding 客户端实例（可能与 LLM 不同）
        self.embedding_client = AsyncOpenAI(
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES,
        )

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        调用 LLM 进行聊天补全

        参数:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}, ...]
            tools: 工具定义列表（OpenAI function calling 格式），可选
            temperature: 采样温度，控制创造性
            max_tokens: 最大生成 token 数
            stream: 是否使用流式输出

        返回:
            API 响应字典，包含 content 和 tool_calls 等字段

        异常:
            抛出 LLMError 当 API 调用失败时
        """
        start_time = time.time()

        try:
            # 构建请求参数
            kwargs = {
                "model": settings.LLM_MODEL_NAME,
                "messages": messages,
                "temperature": temperature,
                "stream": stream,
            }

            # 只在提供时添加可选参数
            if tools:
                kwargs["tools"] = tools
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            # 调用 API
            response = await self.llm_client.chat.completions.create(**kwargs)

            # 解析响应
            if stream:
                # 流式输出暂不支持，明确报错
                raise LLMError("流式输出暂未支持，请将 stream 设为 False")

            # 非流式输出处理
            choice = response.choices[0]
            message = choice.message

            result = {
                "content": message.content,
                "tool_calls": None,
                "finish_reason": choice.finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
                "duration_ms": int((time.time() - start_time) * 1000),
                "raw_response": response,
            }

            # 提取 tool_calls（如果存在）
            if hasattr(message, "tool_calls") and message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]

            return result

        except openai.APIError as e:
            raise LLMError(f"LLM API 错误: {e.message}", status_code=e.status_code) from e
        except openai.APITimeoutError as e:
            raise LLMError(f"LLM API 超时（{settings.LLM_TIMEOUT}秒）", is_timeout=True) from e
        except Exception as e:
            raise LLMError(f"LLM 调用失败: {str(e)}") from e

    async def embedding(
        self,
        texts: str | list[str],
    ) -> dict[str, Any]:
        """
        获取文本的向量嵌入

        参数:
            texts: 单个文本字符串或文本列表

        返回:
            包含 embeddings 和 usage 的字典

        异常:
            抛出 LLMError 当 API 调用失败时
        """
        start_time = time.time()

        # 统一转换为列表
        is_single = isinstance(texts, str)
        input_texts = [texts] if is_single else texts

        try:
            response = await self.embedding_client.embeddings.create(
                model=settings.EMBEDDING_MODEL_NAME,
                input=input_texts,
            )

            embeddings = [item.embedding for item in response.data]

            return {
                "embeddings": embeddings[0] if is_single else embeddings,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
                "duration_ms": int((time.time() - start_time) * 1000),
            }

        except openai.APIError as e:
            raise LLMError(f"Embedding API 错误: {e.message}", status_code=e.status_code) from e
        except Exception as e:
            raise LLMError(f"Embedding 调用失败: {str(e)}") from e

    async def batch_embedding(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """
        批量获取文本嵌入，自动分批处理

        参数:
            texts: 文本列表
            batch_size: 每批处理的数量；默认 settings.EMBEDDING_BATCH_SIZE

        返回:
            嵌入向量列表
        """
        bs = batch_size if batch_size is not None else settings.EMBEDDING_BATCH_SIZE
        all_embeddings = []

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            result = await self.embedding(batch)
            batch_embeddings = result["embeddings"]

            # 确保是列表的列表
            if not isinstance(batch_embeddings[0], list):
                batch_embeddings = [batch_embeddings]

            all_embeddings.extend(batch_embeddings)

            # 简单的速率控制，避免过快请求
            if i + bs < len(texts):
                await asyncio.sleep(0.1)

        return all_embeddings


class LLMError(Exception):
    """
    LLM 调用异常类

    属性:
        message: 错误信息
        status_code: HTTP 状态码（如果有）
        is_timeout: 是否为超时错误
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        is_timeout: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.is_timeout = is_timeout

    def __str__(self) -> str:
        return self.message


# 全局客户端实例（单例模式）
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """
    获取 LLM 客户端单例

    为什么用单例：避免重复创建连接池，提高性能
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


# 便捷函数，直接调用而无需管理实例

async def chat(
    messages: list[dict[str, Any]],
    tools: list[dict] | None = None,
    temperature: float = 0.7,
    **kwargs,
) -> dict[str, Any]:
    """
    便捷函数：调用聊天补全
    使用示例：
        result = await chat([{"role": "user", "content": "你好"}])
        print(result["content"])
    """
    client = get_llm_client()
    return await client.chat_completion(messages, tools, temperature, **kwargs)


async def embed(texts: str | list[str]) -> dict[str, Any]:
    """
    便捷函数：获取文本嵌入
    使用示例：
        result = await embed("这是一段文本")
        vector = result["embeddings"]
    """
    client = get_llm_client()
    return await client.embedding(texts)
