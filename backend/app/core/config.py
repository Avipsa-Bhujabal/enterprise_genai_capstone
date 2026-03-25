from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    openai_api_key: str = Field(default='')
    openai_model: str = Field(default='gpt-4o-mini')
    openai_base_url: str | None = Field(default=None)

    chroma_persist_dir: str = Field(default='./data/chroma_db')
    upload_dir: str = Field(default='./data/uploads')
    collection_name: str = Field(default='enterprise_docs')

    top_k: int = Field(default=4)
    chunk_size: int = Field(default=900)
    chunk_overlap: int = Field(default=150)

    @property
    def chroma_dir_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def upload_dir_path(self) -> Path:
        return Path(self.upload_dir)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.chroma_dir_path.mkdir(parents=True, exist_ok=True)
    settings.upload_dir_path.mkdir(parents=True, exist_ok=True)
    return settings
