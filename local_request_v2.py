# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
import requests
import json
import time
import traceback
from typing import List, Dict, Any, Optional, Union
from func_timeout import func_set_timeout
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from dataclasses import dataclass
from cachetools import TTLCache, cachedmethod
import operator
from cachetools.keys import hashkey
from openai import OpenAI

from global_config import (
    DEEPSEEK_API_KEY, DEEPSEEK_ENDPOINT, DEEPSEEK_API_KEY_AGENT_B,
    DASHSCOPE_API_KEY, DASHSCOPE_ENDPOINT, DASHSCOPE_DEPLOYMENT,
)

try:
    from log import logger
except:
    import logging

    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-usage accounting. Accumulates prompt/completion tokens from REAL API
# responses only (cache hits never reach _make_*_request, so they don't count).
# Used by astabench_bridge to surface SPAR's true model cost on the leaderboard.
# ---------------------------------------------------------------------------
_TOKEN_USAGE = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def reset_token_usage() -> None:
    _TOKEN_USAGE.update(input_tokens=0, output_tokens=0, calls=0)


def get_token_usage() -> Dict[str, int]:
    return dict(_TOKEN_USAGE)


def _add_token_usage(usage) -> None:
    if usage is None:
        return
    _TOKEN_USAGE["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
    _TOKEN_USAGE["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    _TOKEN_USAGE["calls"] += 1


@dataclass
class ModelConfig:
    """Configuration for LLM models"""

    url: str
    max_len: int
    temperature: float = 0.8
    model_name: str = ""
    top_p: float = 0.9
    top_k: int = 20
    min_p: int = 0
    retry_attempts: int = 5
    timeout: int = 200
    think_bool: bool = False
    openai_client: Optional[Any] = None
    reasoning_effort: Optional[str] = None  # DeepSeek thinking: "low"/"medium"/"high"


# Model configurations
# buid your model locally, and set the url
MODEL_CONFIGS = {

    "llama3-70b": ModelConfig(
        url="http://0.0.0.0:9087/v1/chat/completions",
        max_len=131072,
    ),

    "Qwen3-8B": ModelConfig(
        url="http://0.0.0.0:9096/v1",
        max_len=131072,
        model_name="Qwen/Qwen3-8B",
        think_bool=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        openai_client=OpenAI(
            api_key="EMPTY",
            base_url="http://0.0.0.0:9096/v1",
        ),
    ),
    "Qwen3-14B": ModelConfig(
        url="http://0.0.0.0:9095/v1",
        max_len=131072,
        model_name="Qwen/Qwen3-14B",
        think_bool=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        openai_client=OpenAI(
            api_key="EMPTY",
            base_url="http://0.0.0.0:9095/v1",
        ),
    ),
    # Aliyun DashScope — routed through _make_qwen3_request (key contains "Qwen3")
    "Qwen3-32B": ModelConfig(
        url=DASHSCOPE_ENDPOINT,
        max_len=131072,
        model_name=DASHSCOPE_DEPLOYMENT,
        think_bool=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        openai_client=OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_ENDPOINT,
        ),
    ),
    # DeepSeek V4 Flash — fast, no thinking mode
    "DeepSeek-V4": ModelConfig(
        url=DEEPSEEK_ENDPOINT,
        max_len=65536,
        model_name="deepseek-v4-flash",
        think_bool=False,
        reasoning_effort=None,
        temperature=0.7,
        openai_client=OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_ENDPOINT,
        ),
    ),
    # DeepSeek V4 Pro — higher quality with thinking mode
    "DeepSeek-V4-Pro": ModelConfig(
        url=DEEPSEEK_ENDPOINT,
        max_len=65536,
        model_name="deepseek-v4-pro",
        think_bool=True,
        reasoning_effort="high",
        temperature=0.7,
        openai_client=OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_ENDPOINT,
        ),
    ),
    # DeepSeek V4 Pro for Agent B — thinking mode (reasoning_effort high), its own
    # API key, and NO temperature/top_p (unsupported in thinking mode; omitted in
    # _make_deepseek_request). Mirrors bge/test_deepseek_thinking.py.
    "DeepSeek-V4-Pro-AgentB": ModelConfig(
        url=DEEPSEEK_ENDPOINT,
        max_len=65536,
        model_name="deepseek-v4-pro",
        think_bool=True,
        reasoning_effort="high",
        openai_client=OpenAI(
            api_key=DEEPSEEK_API_KEY_AGENT_B,
            base_url=DEEPSEEK_ENDPOINT,
        ),
    ),
    "Qwen3-32B-think": ModelConfig(
        url="http://0.0.0.0:9094/v1",
        max_len=131072,
        model_name="Qwen/Qwen3-32B",
        think_bool=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        openai_client=OpenAI(
            api_key="EMPTY",
            base_url="http://0.0.0.0:9094/v1",
        ),
    ),
}


class LLMClient:
    def __init__(self):
        self.session = self._create_session()
        self.cache = TTLCache(maxsize=100, ttl=3600)  # Cache responses for 1 hour

    def _create_session(self) -> requests.Session:
        """Create a session with retry logic"""
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        session.mount("http://", HTTPAdapter(max_retries=retries))
        return session

    def _cache_key(self, url: str, data: Dict[str, Any], timeout: int) -> tuple:
        """Custom cache key function that makes data hashable"""
        data_str = json.dumps(data, sort_keys=True)  # Serialize dict to string
        return hashkey(url, data_str, timeout)  # Use cachetools' hashkey

    @cachedmethod(operator.attrgetter("cache"), key=_cache_key)
    def _make_request_cached(
        self, url: str, data: Dict[str, Any], timeout: int
    ) -> Optional[str]:
        """Cached HTTP request — callers should use _make_request."""
        try:
            if "DeepSeek" in data["model"]:
                return self._make_deepseek_request(data)
            elif "Qwen3" in data["model"]:
                return self._make_qwen3_request(data)
            else:
                response = self.session.post(url, json=data, timeout=timeout)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return None

    def _make_request(
        self, url: str, data: Dict[str, Any], timeout: int
    ) -> Optional[str]:
        """Make HTTP request with caching; evicts failed (None) entries so retries work."""
        result = self._make_request_cached(url, data, timeout)
        if result is None:
            self.cache.pop(self._cache_key(url, data, timeout), None)
        return result

    def _make_qwen3_request(self, data: Dict[str, Any]) -> Optional[str]:
        """Handle Qwen3 requests via an OpenAI-compatible endpoint (DashScope or local vLLM)."""
        # logger.info("_make_qwen3_request")
        try:
            openai_client = MODEL_CONFIGS[data["model"]].openai_client
            chat_response = openai_client.chat.completions.create(
                model=MODEL_CONFIGS[data["model"]].model_name,
                messages=data["messages"],
                temperature=data.get("temperature", 0.7),
                top_p=data.get("top_p", 0.8),
                presence_penalty=1.5,
                extra_body={"enable_thinking": data.get("think_bool", False)},
            )
            _add_token_usage(getattr(chat_response, "usage", None))
            message = chat_response.choices[0].message
            # DashScope returns the answer in `content`; thinking-mode servers
            # may instead populate `reasoning_content`.
            msg = message.content or getattr(message, "reasoning_content", None)
            if msg and "</think>" in msg:
                msg = msg.split("</think>")[-1]
            return msg
        except Exception as e:
            logger.error(
                f"Qwen3 request failed: {str(e)}--{traceback.format_exc()}"
            )
            return None

    def _make_deepseek_request(self, data: Dict[str, Any]) -> Optional[str]:
        """Handle DeepSeek requests with optional thinking mode via extra_body."""
        try:
            config = MODEL_CONFIGS[data["model"]]
            kwargs: Dict[str, Any] = dict(
                model=config.model_name,
                messages=data["messages"],
                max_tokens=data.get("max_tokens", config.max_len),
            )
            if config.reasoning_effort:
                # Thinking mode: temperature / top_p are unsupported -> do NOT send
                # them (matches bge/test_deepseek_thinking.py).
                kwargs["reasoning_effort"] = config.reasoning_effort
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            else:
                kwargs["temperature"] = data.get("temperature", config.temperature)
            chat_response = config.openai_client.chat.completions.create(**kwargs)
            _add_token_usage(getattr(chat_response, "usage", None))
            message = chat_response.choices[0].message
            # thinking mode: answer is in content; thought is in reasoning_content
            content = message.content or getattr(message, "reasoning_content", None)
            if content and "</think>" in content:
                content = content.split("</think>")[-1]
            return content
        except Exception as e:
            logger.error(f"DeepSeek request failed: {str(e)}--{traceback.format_exc()}")
            return None

    def format_messages(
        self, messages: Union[str, List[Dict[str, str]]], model_name: str
    ) -> List[Dict[str, str]]:
        """Format messages for the model"""
        if isinstance(messages, str):
            system_content = (
                "You are a helpful assistant" if "r1" not in model_name else ""
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": messages},
            ]

        # Handle special cases for r1 and qwq models
        if "r1" in model_name or "qwq" in model_name:
            if len(messages) == 2 and messages[0]["role"] == "system":
                system = messages[0]["content"]
                prompt = messages[1]["content"]
                messages = [
                    {
                        "role": "user",
                        "content": f"{system}\n\n{prompt}\n<think>\n",
                    }
                ]

        # Ensure think tag for r1 models
        if "r1" in model_name and ("<think>" not in messages[-1]["content"]):
            messages[-1]["content"] += "\n<think>\n"

        return messages


client = LLMClient()


@func_set_timeout(600)
def get_from_llm(
    messages: Union[str, List[Dict[str, str]]], model_name: str = "Qwen25-7B", **kwargs
) -> Optional[str]:
    """
    Get response from LLM with improved error handling and retries.

    Args:
        messages: Input messages (string or list of message dicts)
        model_name: Name of the model to use
        **kwargs: Additional parameters to override defaults

    Returns:
        Generated response or None if failed
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODEL_CONFIGS[model_name]


    # Format messages
    formatted_messages = client.format_messages(messages, model_name)

    # Prepare request data
    data = {
        "model": model_name,
        "messages": formatted_messages,
        "tools": [],
        "temperature": kwargs.get("temperature", config.temperature),
        "top_p": kwargs.get("top_p", config.top_p),
        "n": 1,
        "max_tokens": kwargs.get("max_len", config.max_len),
        "stream": False,
    }

    logger.info(f"Requesting {model_name} at {config.url}")

    # Make request with retries
    for attempt in range(config.retry_attempts):
        try:
            response = client._make_request(config.url, data, config.timeout)
            if response:
                if "</think>" in response:
                    response = response.split("</think>")[-1]
                return response
            else:
                logger.error(f"Failed: {response} -- {traceback.format_exc()}")
                secs = 5
                time.sleep(secs)
                pass
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == config.retry_attempts - 1:
                logger.error(f"All attempts failed for {model_name}")
                logger.error(traceback.format_exc())
    return None

