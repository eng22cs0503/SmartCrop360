"""
Configuration management for SmartCrop360 application.

Loads all settings from environment variables with sensible defaults.
"""
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""
    
    # Application
    app_name: str = "SmartCrop360"
    app_version: str = "2.0.0"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Security
    cors_origins: List[str] = ["*"]
    max_file_size_mb: int = 10
    rate_limit_per_minute: int = 10
    rate_limit_query_per_minute: int = 60
    
    # Paths
    disease_data_path: str = "dataset/crop_disease_rag_10_plants_full.json"
    ml_model_path: str = "models/"
    upload_dir: str = "uploads/"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    log_rotation: str = "100 MB"
    
    # Cache
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000
    
    # Performance
    image_max_dimension: int = 1024
    image_min_dimension: int = 224
    ml_inference_timeout: int = 5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
