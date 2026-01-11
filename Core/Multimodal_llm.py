import sys
import os
import base64
import io
import tempfile
import mimetypes
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, BinaryIO
from PIL import Image
import numpy as np
from Entity.Core.Core_engine import RustServer


class MultiLLM:
    def __init__(self, engine_parameters=None, system_prompt=None, engine=None):
        if engine is None:
            if engine_parameters is not None:
                self.engine = RustServer(**engine_parameters)
            else:
                raise ValueError("engine_parameters must be provided")
        else:
            self.engine = engine

        self.messages = []
        self.system_prompt = system_prompt or "You are an Intelligent Computer Operating Agent - CUA.!"
        self.reset()

    def encode_image(self, image_content: Union[str, Path, bytes, BinaryIO, Image.Image, np.ndarray]) -> str:
        """
        Robust image encoding that handles ALL input types.
        Returns: Base64 encoded string
        """
        # If already base64 string, return as-is
        if isinstance(image_content, str) and self._is_base64(image_content):
            return image_content

        try:
            # Handle file paths (string or Path object)
            if isinstance(image_content, (str, Path)):
                if isinstance(image_content, Path):
                    image_path = str(image_content)
                else:
                    image_path = image_content

                if image_path.startswith(('http://', 'https://')):
                    return image_path  # Return URL as-is for now

                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                return base64.b64encode(image_bytes).decode('utf-8')

            # Handle bytes and byte arrays
            elif isinstance(image_content, (bytes, bytearray)):
                return base64.b64encode(image_content).decode('utf-8')

            # Handle file-like objects
            elif hasattr(image_content, 'read'):
                if hasattr(image_content, 'seek'):
                    image_content.seek(0)
                image_bytes = image_content.read()
                if isinstance(image_bytes, str):
                    image_bytes = image_bytes.encode('utf-8')
                return base64.b64encode(image_bytes).decode('utf-8')

            # Handle PIL Image objects
            elif hasattr(image_content, 'save'):
                img_byte_arr = io.BytesIO()
                format = getattr(image_content, 'format', None) or 'PNG'
                image_content.save(img_byte_arr, format=format, optimize=True)
                return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            # Handle NumPy arrays
            elif isinstance(image_content, np.ndarray):
                pil_image = Image.fromarray(image_content)
                return self.encode_image(pil_image)

            else:
                raise ValueError(f"Unsupported image content type: {type(image_content)}")

        except Exception as e:
            raise ValueError(f"Failed to encode image: {str(e)}") from e

    def _is_base64(self, s: str) -> bool:
        """Check if string is already base64 encoded"""
        try:
            if len(s) % 4 != 0:
                return False
            base64.b64decode(s, validate=True)
            return True
        except:
            return False

    def _convert_to_rust_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert to Rust server format while PRESERVING multimodal content.
        The Rust server proxy expects OpenAI-compatible format.
        """
        converted_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # If content is a list (multimodal), preserve the structure
            if isinstance(content, list):
                multimodal_content = []
                for item in content:
                    if item.get("type") == "text":
                        # Keep text as-is
                        multimodal_content.append(item)
                    elif item.get("type") == "image_url":
                        # PRESERVE image data in OpenAI format
                        image_url = item["image_url"]["url"]

                        # Ensure base64 images have proper data URL format
                        if image_url.startswith("data:"):
                            # Already formatted, keep as-is
                            multimodal_content.append(item)
                        elif self._is_base64(image_url):
                            # Convert plain base64 to data URL
                            formatted_url = f"data:image/jpeg;base64,{image_url}"
                            multimodal_content.append({
                                "type": "image_url",
                                "image_url": {"url": formatted_url, "detail": "high"}
                            })
                        else:
                            # Keep other image URLs as-is
                            multimodal_content.append(item)

                converted_messages.append({
                    "role": role,
                    "content": multimodal_content
                })
            else:
                # Simple text content
                converted_messages.append({
                    "role": role,
                    "content": content
                })

        return converted_messages

    def reset(self):
        """Reset messages with system prompt"""
        self.messages = [
            {"role": "system", "content": self.system_prompt}  # Simple string!
        ]

    def add_system_prompt(self, system_prompt):
        """Update system prompt"""
        self.system_prompt = system_prompt
        if len(self.messages) > 0 and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = system_prompt
        else:
            self.messages.insert(0, {"role": "system", "content": system_prompt})

    def remove_message_at(self, index):
        """Remove a message at a given index"""
        if 0 <= index < len(self.messages):
            self.messages.pop(index)

    def replace_message_at(self, index, text_content, image_content=None, image_detail="high"):
        """Replace a message at a given index"""
        if 0 <= index < len(self.messages):
            if image_content:
                # Build proper multimodal content
                base64_image = self.encode_image(image_content)
                data_url = f"data:image/jpeg;base64,{base64_image}"
                content = [
                    {"type": "text", "text": text_content},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": image_detail}}
                ]
            else:
                content = text_content

            self.messages[index] = {"role": self.messages[index]["role"], "content": content}

    def add_message(self, text_content, image_content=None, role=None, image_detail="high", put_text_last=False):
        """Add a new message with proper multimodal support"""

        if role is None:
            # Auto-detect role
            if not self.messages or self.messages[-1]["role"] == "system":
                role = "user"
            elif self.messages[-1]["role"] == "user":
                role = "assistant"
            elif self.messages[-1]["role"] == "assistant":
                role = "user"

        # Build proper multimodal content
        if image_content:
            images = image_content if isinstance(image_content, list) else [image_content]
            content_items = []

            # Add text first (unless put_text_last is True)
            if not put_text_last and text_content:
                content_items.append({"type": "text", "text": text_content})

            # Add images with proper base64 data URLs
            for img in images:
                base64_img = self.encode_image(img)
                data_url = f"data:image/jpeg;base64,{base64_img}"
                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": image_detail}
                })

            # Add text last if requested
            if put_text_last and text_content:
                content_items.append({"type": "text", "text": text_content})

            content = content_items
        else:
            # Simple text content
            content = text_content

        self.messages.append({"role": role, "content": content})

    def get_response(self, user_message=None, messages=None, temperature=0.0, max_new_tokens=None, thinking=False,
                     **kwargs):
        """Generate response using engine"""
        if messages is None:
            messages = self.messages

        if user_message:
            messages.append({"role": "user", "content": user_message})

        # Convert to Rust format if needed (handles both formats)
        rust_messages = self._convert_to_rust_format(messages)

        if thinking:
            return self.engine.generate_with_thinking(rust_messages, temperature=temperature,
                                                      max_new_tokens=max_new_tokens, **kwargs)

        return self.engine.generate(rust_messages, temperature=temperature, max_new_tokens=max_new_tokens, **kwargs)

    # Alternative approach for image handling
    def add_message_with_image_data(self, text_content, image_data=None, role=None):
        """
        Alternative: Store images separately and let Rust server handle them
        if it supports image endpoints
        """
        if role is None:
            # Auto-detect role
            pass

        message = {"role": role, "content": text_content}

        if image_data:
            message["images"] = [self.encode_image(img) for img in
                                 (image_data if isinstance(image_data, list) else [image_data])]

        self.messages.append(message)