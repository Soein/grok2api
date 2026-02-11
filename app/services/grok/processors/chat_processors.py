"""
聊天响应处理器
"""

import asyncio
import uuid
import re
import threading
from typing import Any, AsyncGenerator, AsyncIterable, Dict, Optional

import orjson
from curl_cffi.requests.errors import RequestsError

from app.core.config import get_config
from app.core.logger import logger
from app.core.exceptions import UpstreamException
from .base import (
    BaseProcessor,
    StreamIdleTimeoutError,
    _with_idle_timeout,
    _normalize_stream_line,
    _collect_image_urls,
    _is_http2_stream_error,
)

# Agent 模型列表（需要特殊处理 thinking 的模型）
AGENT_MODELS = [
    'grok-4-heavy',
    'grok-4',
    'grok-4-mini-thinking-tahoe',
    'grok-3-deepsearch',
    'grok-3-deepersearch',
    'grok-3-reasoning'
]

# Thinking 相关的消息标签
THINKING_TAGS = {
    'header',
    'summary',
    'raw_function_result',
    'citedWebSearchResults',
    'tool_usage_card'
}


def process_model_response(response: Dict[str, Any], model: str, show_thinking: bool = None) -> Dict[str, Optional[str]]:
    """
    分类处理 Grok API 响应，区分 thinking 和 content

    Args:
        response: Grok API 响应数据
        model: 模型名称
        show_thinking: 是否显示 thinking 内容（None 则使用全局配置）

    Returns:
        {"token": 内容, "type": "thinking"/"content"/"heartbeat"/None}
    """
    result = {"token": None, "type": None}

    # 忽略缓存的图像生成响应
    if response.get("cachedImageGenerationResponse"):
        return result

    message_tag = response.get("messageTag")
    token = response.get("token")

    # 心跳消息
    if message_tag == 'heartbeat':
        result["type"] = 'heartbeat'
        return result

    # 对于非 Agent 模型，如果收到 modelResponse 就直接返回
    if model not in AGENT_MODELS and response.get("modelResponse"):
        return result

    # 最终响应（包含完整消息）
    if response.get("modelResponse") and isinstance(response["modelResponse"], dict):
        final_message = response["modelResponse"].get("message")
        if final_message:
            result["token"] = final_message
            result["type"] = 'content'
        return result

    # 判断是否是 thinking 内容（仅对 Agent 模型）
    is_thinking_content = False
    if model in AGENT_MODELS:
        if (message_tag in THINKING_TAGS or
            response.get("isThinking") or
            response.get("messageStepId")):
            is_thinking_content = True

    if is_thinking_content:
        # 根据配置决定是否显示 thinking
        if show_thinking is None:
            show_thinking = get_config("chat.thinking")
        if not show_thinking:
            return result

        # 返回 thinking 内容
        if token:
            result["token"] = token
            result["type"] = 'thinking'
        return result

    # 普通 token
    if token:
        result["token"] = token
        result["type"] = 'content'
        return result

    return result


class StreamProcessor(BaseProcessor):
    """流式响应处理器"""

    def __init__(self, model: str, token: str = "", think: bool = None):
        super().__init__(model, token)
        self.response_id: str = None
        self.fingerprint: str = ""
        self.think_opened: bool = False
        self.role_sent: bool = False
        self.filter_tags = get_config("chat.filter_tags")
        self.image_format = get_config("app.image_format")
        self._tag_buffer: str = ""
        self._in_filter_tag: bool = False

        if think is None:
            self.show_think = get_config("chat.thinking")
        else:
            self.show_think = think

    def _filter_token(self, token: str) -> str:
        """过滤 token 中的特殊标签（如 <grok:render>...</grok:render>），支持跨 token 的标签过滤"""
        if not self.filter_tags:
            return token

        result = []
        i = 0
        while i < len(token):
            char = token[i]

            if self._in_filter_tag:
                self._tag_buffer += char
                if char == ">":
                    if "/>" in self._tag_buffer:
                        self._in_filter_tag = False
                        self._tag_buffer = ""
                    else:
                        for tag in self.filter_tags:
                            if f"</{tag}>" in self._tag_buffer:
                                self._in_filter_tag = False
                                self._tag_buffer = ""
                                break
                i += 1
                continue

            if char == "<":
                remaining = token[i:]
                tag_started = False
                for tag in self.filter_tags:
                    if remaining.startswith(f"<{tag}"):
                        tag_started = True
                        break
                    if len(remaining) < len(tag) + 1:
                        for j in range(1, len(remaining) + 1):
                            if f"<{tag}".startswith(remaining[:j]):
                                tag_started = True
                                break

                if tag_started:
                    self._in_filter_tag = True
                    self._tag_buffer = char
                    i += 1
                    continue

            result.append(char)
            i += 1

        return "".join(result)

    def _sse(self, content: str = "", role: str = None, finish: str = None) -> str:
        """构建 SSE 响应"""
        delta = {}
        if role:
            delta["role"] = role
            delta["content"] = ""
        elif content:
            delta["content"] = content

        chunk = {
            "id": self.response_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.fingerprint,
            "choices": [
                {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
            ],
        }
        return f"data: {orjson.dumps(chunk).decode()}\n\n"

    def _heartbeat_sse(self) -> str:
        """构建 SSE 心跳响应（使用注释格式，客户端会忽略但保持连接）"""
        return ": ping\n\n"

    async def process(
        self, response: AsyncIterable[bytes]
    ) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        idle_timeout = get_config("timeout.stream_idle_timeout")

        try:
            async for line in _with_idle_timeout(response, idle_timeout, self.model):
                line = _normalize_stream_line(line)
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue

                resp = data.get("result", {}).get("response", {})

                # 使用 process_model_response 分类响应
                classified = process_model_response(resp, self.model, self.show_think)
                msg_type = classified.get("type")
                token = classified.get("token")

                if (llm := resp.get("llmInfo")) and not self.fingerprint:
                    self.fingerprint = llm.get("modelHash", "")
                if rid := resp.get("responseId"):
                    self.response_id = rid

                if not self.role_sent:
                    yield self._sse(role="assistant")
                    self.role_sent = True

                # 注意：Grok API 的心跳消息（msg_type=='heartbeat'）会被自动忽略
                # 因为外层 stream_with_heartbeat() 已经提供主动心跳机制，无需转发
                # 如果需要转发，取消注释以下代码：
                # if msg_type == 'heartbeat':
                #     yield self._sse("")  # 或自定义心跳格式
                #     continue

                # 图像生成进度
                if img := resp.get("streamingImageGenerationResponse"):
                    if self.show_think:
                        if not self.think_opened:
                            yield self._sse("<think>\n")
                            self.think_opened = True
                        idx = img.get("imageIndex", 0) + 1
                        progress = img.get("progress", 0)
                        yield self._sse(
                            f"正在生成第{idx}张图片中，当前进度{progress}%\n"
                        )
                    continue

                # modelResponse
                if mr := resp.get("modelResponse"):
                    if self.think_opened and self.show_think:
                        if msg := mr.get("message"):
                            yield self._sse(msg + "\n")
                        yield self._sse("</think>\n")
                        self.think_opened = False

                    # 处理生成的图片
                    for url in _collect_image_urls(mr):
                        parts = url.split("/")
                        img_id = parts[-2] if len(parts) >= 2 else "image"

                        if self.image_format == "base64":
                            try:
                                dl_service = self._get_dl()
                                base64_data = await dl_service.to_base64(
                                    url, self.token, "image"
                                )
                                if base64_data:
                                    yield self._sse(f"![{img_id}]({base64_data})\n")
                                else:
                                    final_url = await self.process_url(url, "image")
                                    yield self._sse(f"![{img_id}]({final_url})\n")
                            except Exception as e:
                                logger.warning(
                                    f"Failed to convert image to base64, falling back to URL: {e}"
                                )
                                final_url = await self.process_url(url, "image")
                                yield self._sse(f"![{img_id}]({final_url})\n")
                        else:
                            final_url = await self.process_url(url, "image")
                            yield self._sse(f"![{img_id}]({final_url})\n")

                    if (
                        (meta := mr.get("metadata", {}))
                        .get("llm_info", {})
                        .get("modelHash")
                    ):
                        self.fingerprint = meta["llm_info"]["modelHash"]
                    continue

                # 处理 thinking 和 content
                if msg_type == 'thinking':
                    # Thinking 内容（已由分类器根据配置过滤）
                    if not self.think_opened:
                        yield self._sse("<think>\n")
                        self.think_opened = True
                    if token:
                        filtered = self._filter_token(token)
                        if filtered:
                            yield self._sse(filtered)
                elif msg_type == 'content':
                    # 普通内容（关闭可能打开的 thinking 标签）
                    if self.think_opened:
                        yield self._sse("</think>\n")
                        self.think_opened = False
                    if token:
                        filtered = self._filter_token(token)
                        if filtered:
                            yield self._sse(filtered)

            if self.think_opened:
                yield self._sse("</think>\n")
            yield self._sse(finish="stop")
            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            logger.debug("Stream cancelled by client", extra={"model": self.model})
        except StreamIdleTimeoutError as e:
            raise UpstreamException(
                message=f"Stream idle timeout after {e.idle_seconds}s",
                status_code=504,
                details={
                    "error": str(e),
                    "type": "stream_idle_timeout",
                    "idle_seconds": e.idle_seconds,
                },
            )
        except RequestsError as e:
            if _is_http2_stream_error(e):
                logger.warning(f"HTTP/2 stream error: {e}", extra={"model": self.model})
                raise UpstreamException(
                    message="Upstream connection closed unexpectedly",
                    status_code=502,
                    details={"error": str(e), "type": "http2_stream_error"},
                )
            logger.error(f"Stream request error: {e}", extra={"model": self.model})
            raise UpstreamException(
                message=f"Upstream request failed: {e}",
                status_code=502,
                details={"error": str(e)},
            )
        except Exception as e:
            logger.error(
                f"Stream processing error: {e}",
                extra={"model": self.model, "error_type": type(e).__name__},
            )
            raise
        finally:
            await self.close()


class CollectProcessor(BaseProcessor):
    """非流式响应处理器"""

    def __init__(self, model: str, token: str = ""):
        super().__init__(model, token)
        self.image_format = get_config("app.image_format")
        self.filter_tags = get_config("chat.filter_tags")

    def _filter_content(self, content: str) -> str:
        """过滤内容中的特殊标签"""
        if not content or not self.filter_tags:
            return content

        result = content
        for tag in self.filter_tags:
            pattern = rf"<{re.escape(tag)}[^>]*>.*?</{re.escape(tag)}>|<{re.escape(tag)}[^>]*/>"
            result = re.sub(pattern, "", result, flags=re.DOTALL)

        return result

    async def process(self, response: AsyncIterable[bytes]) -> dict[str, Any]:
        """处理并收集完整响应"""
        response_id = ""
        fingerprint = ""
        content = ""
        idle_timeout = get_config("timeout.stream_idle_timeout")

        try:
            async for line in _with_idle_timeout(response, idle_timeout, self.model):
                line = _normalize_stream_line(line)
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue

                resp = data.get("result", {}).get("response", {})

                if (llm := resp.get("llmInfo")) and not fingerprint:
                    fingerprint = llm.get("modelHash", "")

                if mr := resp.get("modelResponse"):
                    response_id = mr.get("responseId", "")
                    content = mr.get("message", "")

                    if urls := _collect_image_urls(mr):
                        content += "\n"
                        for url in urls:
                            parts = url.split("/")
                            img_id = parts[-2] if len(parts) >= 2 else "image"

                            if self.image_format == "base64":
                                try:
                                    dl_service = self._get_dl()
                                    base64_data = await dl_service.to_base64(
                                        url, self.token, "image"
                                    )
                                    if base64_data:
                                        content += f"![{img_id}]({base64_data})\n"
                                    else:
                                        final_url = await self.process_url(url, "image")
                                        content += f"![{img_id}]({final_url})\n"
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to convert image to base64, falling back to URL: {e}"
                                    )
                                    final_url = await self.process_url(url, "image")
                                    content += f"![{img_id}]({final_url})\n"
                            else:
                                final_url = await self.process_url(url, "image")
                                content += f"![{img_id}]({final_url})\n"

                    if (
                        (meta := mr.get("metadata", {}))
                        .get("llm_info", {})
                        .get("modelHash")
                    ):
                        fingerprint = meta["llm_info"]["modelHash"]

        except asyncio.CancelledError:
            logger.debug("Collect cancelled by client", extra={"model": self.model})
        except StreamIdleTimeoutError as e:
            logger.warning(f"Collect idle timeout: {e}", extra={"model": self.model})
        except RequestsError as e:
            if _is_http2_stream_error(e):
                logger.warning(
                    f"HTTP/2 stream error in collect: {e}", extra={"model": self.model}
                )
            else:
                logger.error(f"Collect request error: {e}", extra={"model": self.model})
        except Exception as e:
            logger.error(
                f"Collect processing error: {e}",
                extra={"model": self.model, "error_type": type(e).__name__},
            )
        finally:
            await self.close()

        content = self._filter_content(content)

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": fingerprint,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "refusal": None,
                        "annotations": [],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "text_tokens": 0,
                    "audio_tokens": 0,
                    "image_tokens": 0,
                },
                "completion_tokens_details": {
                    "text_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                },
            },
        }


__all__ = ["StreamProcessor", "CollectProcessor"]
