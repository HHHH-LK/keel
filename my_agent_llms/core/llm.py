import os
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from openai import OpenAI

load_dotenv()


class MyLLM:
    """统一的多厂商 LLM 客户端封装。"""

    OPENAI_COMPATIBLE_PROVIDERS = {
        "openai",
        "aliyun",
        "modelscope",
        "zhipu",
        "ollama",
        "vllm",
        "local",
    }
    LOCAL_PROVIDERS = {"ollama", "vllm", "local"}
    PROVIDERS = OPENAI_COMPATIBLE_PROVIDERS | {"auto", "anthropic", "gemini"}

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "auto",
        **kwargs,
    ):
        normalized_provider = provider.lower()
        if normalized_provider not in self.PROVIDERS:
            raise ValueError(f"不支持的 provider: {provider}")

        self.model = model or os.getenv("LLM_MODEL_ID")

        if normalized_provider == "auto":
            normalized_provider = self._auto_detect_provider(api_key, base_url, self.model)
            if normalized_provider == "auto":
                raise ValueError("无法自动识别供应商，请显式指定 provider")

        self.provider = normalized_provider
        self.api_key, self.base_url = self._resolve_credentials(api_key, base_url)
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens")
        self.timeout = kwargs.get("timeout", 60)

        self._validate_config()
        self.client = self._build_client()

    @staticmethod
    def _auto_detect_provider(
        api_key: Optional[str],
        base_url: Optional[str],
        model: Optional[str],
    ) -> str:
        actual_model = (model or os.getenv("LLM_MODEL_ID") or "").lower()
        actual_api_key = api_key or os.getenv("LLM_API_KEY")
        actual_base_url = (base_url or os.getenv("LLM_BASE_URL") or "").lower()

        if actual_base_url:
            if "generativelanguage.googleapis.com" in actual_base_url:
                return "gemini"
            if "api.anthropic.com" in actual_base_url:
                return "anthropic"
            if "dashscope.aliyuncs.com" in actual_base_url:
                return "aliyun"
            if "api-inference.modelscope.cn" in actual_base_url:
                return "modelscope"
            if "open.bigmodel.cn" in actual_base_url:
                return "zhipu"
            if "api.openai.com" in actual_base_url:
                return "openai"
            if "localhost" in actual_base_url or "127.0.0.1" in actual_base_url:
                if ":11434" in actual_base_url:
                    return "ollama"
                if ":8000" in actual_base_url:
                    return "vllm"
                return "local"

        if actual_model:
            if actual_model.startswith("gemini"):
                return "gemini"
            if actual_model.startswith("claude"):
                return "anthropic"
            if actual_model.startswith(("qwen", "qwq")):
                return "aliyun" if os.getenv("DASHSCOPE_API_KEY") else "auto"
            if actual_model.startswith(("glm", "charglm")):
                return "zhipu"

        if actual_api_key:
            if actual_api_key.startswith("sk-ant-"):
                return "anthropic"
            if actual_api_key.startswith("ms-"):
                return "modelscope"
            if actual_api_key.startswith("AIza"):
                return "gemini"
            if actual_api_key.startswith("sk-"):
                return "openai"

        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("DASHSCOPE_API_KEY"):
            return "aliyun"
        if os.getenv("MODELSCOPE_API_KEY"):
            return "modelscope"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ZHIPU_API_KEY"):
            return "zhipu"

        return "auto"

    def _resolve_credentials(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        if self.provider == "gemini":
            resolved_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("GEMINI_BASE_URL")

        elif self.provider == "anthropic":
            resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")

        elif self.provider == "aliyun":
            resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = (
                base_url
                or os.getenv("DASHSCOPE_BASE_URL")
                or os.getenv("LLM_BASE_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

        elif self.provider == "modelscope":
            resolved_api_key = api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = (
                base_url
                or os.getenv("MODELSCOPE_BASE_URL")
                or os.getenv("LLM_BASE_URL")
                or "https://api-inference.modelscope.cn/v1/"
            )

        elif self.provider == "openai":
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = (
                base_url
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("LLM_BASE_URL")
                or "https://api.openai.com/v1"
            )

        elif self.provider == "zhipu":
            resolved_api_key = api_key or os.getenv("ZHIPU_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = (
                base_url
                or os.getenv("ZHIPU_BASE_URL")
                or os.getenv("LLM_BASE_URL")
                or "https://open.bigmodel.cn/api/paas/v4/"
            )

        elif self.provider == "ollama":
            resolved_api_key = api_key or "ollama"
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "http://localhost:11434/v1"

        elif self.provider == "vllm":
            resolved_api_key = api_key or "vllm"
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "http://localhost:8000/v1"

        elif self.provider == "local":
            resolved_api_key = api_key or "local"
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "http://localhost:8000/v1"

        else:
            raise ValueError(f"不支持的 provider: {self.provider}")

        return resolved_api_key, resolved_base_url

    def _validate_config(self) -> None:
        if not self.model:
            raise ValueError("model 未提供，请传入 model 或在环境变量中配置 LLM_MODEL_ID。")

        if self.provider not in {"gemini", "anthropic"} and not self.base_url:
            raise ValueError("base_url 未提供，请传入 base_url 或在环境变量中配置。")

        if self.provider not in self.LOCAL_PROVIDERS and not self.api_key:
            raise ValueError(f"{self.provider} 的 API Key 未提供。")

    def _build_client(self):
        if self.provider in self.OPENAI_COMPATIBLE_PROVIDERS:
            return OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

        if self.provider == "anthropic":
            client_kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            return Anthropic(**client_kwargs)

        if self.provider == "gemini":
            # Gemini 官方推荐通过 Google GenAI SDK 直连默认端点。
            return genai.Client(api_key=self.api_key)

        raise ValueError(f"不支持的 provider: {self.provider}")

    def think(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = 0.5,
        stream: bool = True,
    ) -> str:
        """统一的模型调用入口。"""
        if not messages:
            raise ValueError("messages 不能为空")

        normalized_messages = self._normalize_messages(messages)
        actual_temperature = self.temperature if temperature is None else temperature

        try:
            if self.provider in self.OPENAI_COMPATIBLE_PROVIDERS:
                return self._think_openai_compatible(normalized_messages, actual_temperature, stream)

            if self.provider == "anthropic":
                return self._think_anthropic(normalized_messages, actual_temperature)

            if self.provider == "gemini":
                return self._think_gemini(normalized_messages, actual_temperature)

            raise ValueError(f"暂不支持的 provider: {self.provider}")

        except Exception as e:
            raise RuntimeError(f"{self.provider} 调用失败: {e}") from e

    def _think_openai_compatible(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        stream: bool,
    ) -> str:
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**request_kwargs)

        if stream:
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
                    collected_content.append(content)
            print()
            return "".join(collected_content)

        return response.choices[0].message.content or ""

    def _think_anthropic(
        self,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> str:
        system_prompt, anthropic_messages = self._split_system_messages(messages)
        request_kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens or 1024,
        }

        if system_prompt:
            request_kwargs["system"] = system_prompt

        response = self.client.messages.create(**request_kwargs)

        text_parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)

        result = "".join(text_parts)
        print(result)
        return result

    def _think_gemini(
        self,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> str:
        system_prompt, user_prompt = self._split_system_and_prompt_text(messages)
        config_kwargs = {"temperature": temperature}

        if self.max_tokens is not None:
            config_kwargs["max_output_tokens"] = self.max_tokens
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        result = getattr(response, "text", "") or ""
        print(result)
        return result

    @staticmethod
    def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized_messages = []
        supported_roles = {"system", "user", "assistant", "tool"}

        for msg in messages:
            if "role" in msg and "content" in msg:
                role = str(msg["role"]).strip().lower() or "user"
                content = str(msg["content"])
            else:
                matched_role = next((role for role in supported_roles if role in msg), None)
                if matched_role is None:
                    raise ValueError(
                        "messages 格式错误，必须包含 role/content，或使用 {'user': '...'} 这类简写格式。"
                    )
                role = matched_role
                content = str(msg[matched_role])

            if role not in supported_roles:
                role = "user"

            normalized_messages.append({"role": role, "content": content})

        return normalized_messages

    @staticmethod
    def _split_system_messages(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
        system_parts = []
        normal_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_parts.append(content)
            elif role in {"user", "assistant"}:
                normal_messages.append({"role": role, "content": content})
            else:
                normal_messages.append({"role": "user", "content": content})

        if not normal_messages:
            normal_messages = [{"role": "user", "content": "请继续。"}]

        return "\n".join(system_parts), normal_messages

    @staticmethod
    def _split_system_and_prompt_text(messages: list[dict[str, str]]) -> tuple[str, str]:
        system_parts = []
        lines = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_parts.append(content)
                continue

            lines.append(f"{role.upper()}: {content}")

        lines.append("ASSISTANT:")
        return "\n".join(system_parts), "\n\n".join(lines)
