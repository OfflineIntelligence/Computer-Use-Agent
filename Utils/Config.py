# Entity/Utils/Config.py

import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralized configuration management"""

    # Rust Server Configuration
    RUST_SERVER_URL = os.getenv("RUST_SERVER_URL")
    RUST_SERVER_MODEL = os.getenv("RUST_SERVER_MODEL")
    RUST_SERVER_TIMEOUT = int(os.getenv("RUST_SERVER_TIMEOUT"))
    RUST_SERVER_MAX_TOKENS = int(os.getenv("RUST_SERVER_MAX_TOKENS"))

    # Model Performance Parameters
    MODEL_CONTEXT_SIZE = int(os.getenv("MODEL_CONTEXT_SIZE"))
    MODEL_BATCH_SIZE = int(os.getenv("MODEL_BATCH_SIZE"))
    MODEL_THREADS = int(os.getenv("MODEL_THREADS"))
    MODEL_GPU_LAYERS = int(os.getenv("MODEL_GPU_LAYERS"))

    # Agent Memory Configuration
    AGENT_TRAJECTORY_MAXLEN = int(os.getenv("AGENT_TRAJECTORY_MAXLEN"))
    AGENT_ENABLE_REFLECTION = os.getenv("AGENT_ENABLE_REFLECTION").lower() == "true"
    AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE"))

    # Execution Configuration
    AGENT_MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES"))
    AGENT_RETRY_DELAY = float(os.getenv("AGENT_RETRY_DELAY"))

    # OCR Configuration
    TESSERACT_PATH = os.getenv("TESSERACT_PATH")
    TESSERACT_CONFIDENCE_THRESHOLD = int(os.getenv("TESSERACT_CONFIDENCE_THRESHOLD"))

    # Agent Configuration
    AGENT_SAFETY_MODE = os.getenv("AGENT_SAFETY_MODE")
    AGENT_PLATFORM = os.getenv("AGENT_PLATFORM")
    AGENT_COORDINATE_WIDTH = int(os.getenv("AGENT_COORDINATE_WIDTH"))
    AGENT_COORDINATE_HEIGHT = int(os.getenv("AGENT_COORDINATE_HEIGHT"))

    # Visual Analysis
    VISUAL_CHANGE_THRESHOLD = float(os.getenv("VISUAL_CHANGE_THRESHOLD"))
    OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD"))

    # Performance Optimizations
    AGENT_STREAM_RESPONSES = os.getenv("AGENT_STREAM_RESPONSES", "false").lower() == "true"
    AGENT_REQUEST_TIMEOUT = int(os.getenv("AGENT_REQUEST_TIMEOUT", "120"))

    @classmethod
    def get_rust_config(cls) -> Dict[str, Any]:
        """Get Rust server configuration"""
        return {
            "base_url": cls.RUST_SERVER_URL,
            "model": cls.RUST_SERVER_MODEL,
            "timeout": cls.RUST_SERVER_TIMEOUT,
            "max_tokens": cls.RUST_SERVER_MAX_TOKENS,
        }

    @classmethod
    def get_performance_config(cls) -> Dict[str, Any]:
        """Get performance optimization configuration"""
        return {
            "context_size": cls.MODEL_CONTEXT_SIZE,
            "batch_size": cls.MODEL_BATCH_SIZE,
            "threads": cls.MODEL_THREADS,
            "gpu_layers": cls.MODEL_GPU_LAYERS,
            "stream_responses": cls.AGENT_STREAM_RESPONSES,
            "request_timeout": cls.AGENT_REQUEST_TIMEOUT,
        }

    @classmethod
    def get_agent_config(cls) -> Dict[str, Any]:
        """Get agent configuration"""
        return {
            "platform": cls.AGENT_PLATFORM,
            "safety_mode": cls.AGENT_SAFETY_MODE,
            "coordinate_width": cls.AGENT_COORDINATE_WIDTH,
            "coordinate_height": cls.AGENT_COORDINATE_HEIGHT,
            "temperature": cls.AGENT_TEMPERATURE,
            "max_retries": cls.AGENT_MAX_RETRIES,
            "retry_delay": cls.AGENT_RETRY_DELAY,
            "stream_responses": cls.AGENT_STREAM_RESPONSES,
        }

    @classmethod
    def get_memory_config(cls) -> Dict[str, Any]:
        """Get memory management configuration"""
        return {
            "trajectory_maxlen": cls.AGENT_TRAJECTORY_MAXLEN,
            "enable_reflection": cls.AGENT_ENABLE_REFLECTION,
        }

    @classmethod
    def get_ocr_config(cls) -> Dict[str, Any]:
        """Get OCR configuration"""
        return {
            "tesseract_path": cls.TESSERACT_PATH,
            "confidence_threshold": cls.TESSERACT_CONFIDENCE_THRESHOLD,
        }

    @classmethod
    def get_visual_config(cls) -> Dict[str, Any]:
        """Get visual analysis configuration"""
        return {
            "change_threshold": cls.VISUAL_CHANGE_THRESHOLD,
            "ocr_confidence": cls.OCR_CONFIDENCE_THRESHOLD,
        }