# Entity/Core/Core_engine.py

import os
import sys
import backoff
import requests
import json
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError
import logging

from Entity.Utils.Config import Config

logger = logging.getLogger("RustServer")


class CoreEngine(ABC):
    """Abstract base class for core LMM-Engine"""

    def __init__(
            self,
            base_url: Optional[str] = None,
            model: Optional[str] = None,
            rate_limit: int = -1,
            temperature: Optional[float] = None,
            **kwargs,
    ):
        self.base_url = base_url
        self.model = model
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.temperature = temperature
        self.llm_client = None

    @abstractmethod
    def generate(
            self,
            messages: List[Dict[str, Any]],
            temperature: float = 0.0,
            max_new_tokens: Optional[int] = None,
            **kwargs,
    ) -> str:
        """Generate a response given input messages"""
        pass

    @abstractmethod
    def generate_with_thinking(
            self,
            messages: List[Dict[str, Any]],
            temperature: float = 0.0,
            max_new_tokens: Optional[int] = None,
            **kwargs,
    ) -> str:
        """Generate a response with internal reasoning (if supported)"""
        pass


class RustServer(CoreEngine):
    """Engine for communicating with a locally hosted Rust LLM server"""

    def __init__(
            self,
            base_url: Optional[str] = None,
            model: Optional[str] = None,
            rate_limit: int = -1,
            temperature: Optional[float] = None,
            **kwargs,
    ):
        rust_config = Config.get_rust_config()
        super().__init__(
            base_url=base_url or rust_config["base_url"],
            model=model or rust_config["model"],
            rate_limit=rate_limit,
            temperature=temperature,
            **kwargs,
        )
        self.timeout = rust_config["timeout"]
        self.max_tokens = rust_config["max_tokens"]

    def _build_payload(
            self,
            messages,
            temperature,
            max_new_tokens,
            **kwargs,
    ):

        """Build payload in the format your Rust server expects"""
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            **kwargs,
        }
        if self.model:
            payload["model"] = self.model
        return payload

    def _handle_streaming_response(self, response) -> Dict[str, Any]:
        """Handle Server-Sent Events (SSE) streaming responses."""
        try:
            full_content = ""
            thinking_content = ""
            in_thinking = False

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')

                    # Skip empty lines and comments
                    if not line.strip() or line.startswith(':'):
                        continue

                    # Parse SSE format: "data: {json}"
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove "data: " prefix

                        if data_str == '[DONE]':
                            break

                        try:
                            data = json.loads(data_str)

                            # FIX: Handle None responses safely
                            if data is None:
                                continue

                            # Extract content from OpenAI streaming format
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})

                                # FIX: Handle None values in delta
                                if delta is None:
                                    continue

                                # Check for thinking vs response content
                                if 'thinking' in delta and delta['thinking'] is not None:
                                    thinking_content += str(delta['thinking'])
                                    in_thinking = True
                                elif 'content' in delta and delta['content'] is not None:
                                    full_content += str(delta['content'])
                                    in_thinking = False

                        except json.JSONDecodeError:
                            # FIX: Skip invalid JSON instead of breaking
                            continue
                        except Exception as e:
                            logger.debug(f"Error parsing SSE data: {e}")
                            continue

            # FIX: Ensure we return valid strings
            result = {"response": full_content.strip() if full_content else ""}
            if thinking_content.strip():
                result["thinking"] = thinking_content.strip()

            return result

        except Exception as e:
            logger.error(f"Streaming response handling failed: {e}")
            return {"response": ""}  # FIX: Return empty response instead of error message

    def _post(self, payload: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        # FIX: Add request validation
        if not payload or "messages" not in payload:
            raise ValueError("Invalid payload: missing 'messages'")

        # CORRECT ENDPOINT SELECTION
        if payload.get("stream", False) or payload.get("thinking", False):
            endpoint = "/generate/stream"
        else:
            endpoint = "/generate"

        try:
            # FIX: Add timeout validation
            actual_timeout = timeout or self.timeout
            if actual_timeout <= 0:
                actual_timeout = 300  # Default 5 minutes

            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                timeout=actual_timeout,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()

            # FIX: Validate response content
            if endpoint == "/generate/stream":
                result = self._handle_streaming_response(response)
            else:
                result = response.json()

            # FIX: Better backend availability check
            if isinstance(result, dict) and result.get("response") in ["Backend unavailable", None, ""]:
                logger.error("Backend returned empty or unavailable response")
                raise RuntimeError("LLM backend server is not responding properly")

            return result

        except requests.exceptions.Timeout:
            logger.error(f"Rust server request timed out after {actual_timeout} seconds")
            raise RuntimeError(f"Rust server timeout after {actual_timeout} seconds")

        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Rust server at {self.base_url}")
            raise RuntimeError(f"Cannot connect to Rust server at {self.base_url}")

        except requests.exceptions.HTTPError as e:
            logger.error(f"Rust server HTTP error: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Rust server HTTP error: {e.response.status_code}")

        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Rust server: {e}")
            raise RuntimeError("Invalid JSON response from Rust server")

        except requests.RequestException as e:
            logger.error(f"Rust server request failed: {e}")
            raise RuntimeError(f"Rust server communication error: {e}")

    @staticmethod
    def _safe_extract(result: Dict[str, Any], key: str) -> str:
        """Extract key from RustServer result, fallback to alternatives"""
        # FIX: Handle None results safely
        if result is None:
            return ""

        alternatives = ["response", "output", "answer", "content"]
        value = result.get(key, None)
        if value is None:
            for alt in alternatives:
                if alt in result:
                    value = result[alt]
                    break
        if value is None:
            logger.warning("RustServer response missing key '%s': %s", key, result)
            return ""
        if isinstance(value, (dict, list)):
            return str(value)
        return str(value) if value is not None else ""

    def _trim_conversation_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Trim conversation to essential context only"""
        if len(messages) <= 4:
            return messages

        # Keep system prompt and last 2 exchanges
        essential_messages = [messages[0]]  # System prompt

        # Add last 2 user-assistant exchanges (4 messages)
        # Assuming the last 4 messages are the last two exchanges (user, assistant, user, assistant)
        essential_messages.extend(messages[-4:])

        logger.debug(f"Trimmed context from {len(messages)} to {len(essential_messages)} messages")
        return essential_messages

    @backoff.on_exception(
        backoff.expo, (RequestException, Timeout, ConnectionError, HTTPError), max_time=600
    )
    def generate(
            self,
            messages: List[Dict[str, Any]],
            temperature: float = 0.0,
            max_new_tokens: Optional[int] = None,
            **kwargs,
    ) -> str:
        """Generate response using Rust server API"""
        # FIX: Validate messages
        if not messages:
            logger.warning("Empty messages provided to generate")
            return ""

        # NEW: Trim messages to essential context
        trimmed_messages = self._trim_conversation_context(messages)

        payload = self._build_payload(trimmed_messages, temperature, max_new_tokens, **kwargs)
        result = self._post(payload, timeout=self.timeout)
        return self._safe_extract(result, "response")

    @backoff.on_exception(
        backoff.expo, (RequestException, Timeout, ConnectionError, HTTPError), max_time=900
    )
    def generate_with_thinking(
            self,
            messages: List[Dict[str, Any]],
            temperature: float = 0.0,
            max_new_tokens: Optional[int] = None,
            **kwargs,
    ) -> str:
        """Generate response with 'thinking' enabled"""
        # FIX: Validate messages
        if not messages:
            logger.warning("Empty messages provided to generate_with_thinking")
            return ""

        payload = self._build_payload(
            messages, temperature, max_new_tokens,
            stream=True,  # ADD THIS for proper streaming
            **kwargs
        )
        result = self._post(payload, timeout=self.timeout)

        thinking_content = self._safe_extract(result, "thinking")
        response_content = self._safe_extract(result, "response")

        if thinking_content and response_content:
            return (
                f"<thoughts>\n{thinking_content}\n</thoughts>\n\n"
                f"<answer>\n{response_content}\n</answer>\n"
            )

        return response_content