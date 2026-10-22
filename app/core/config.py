from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Multimodel Knowledge Assistant"

    ollama_base_url: str= "http://localhost:11434"
    ollama_model: str = 'llama3.2'
    temperature: float = 0.0

    model_config = SettingsConfigDict(
        env = '.env',
        env_file_encoding='utf-8',
        extra = 'ignore'
    )

settings = Settings()
