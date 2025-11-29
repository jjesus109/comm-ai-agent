from pydantic_settings import BaseSettings


class Configuration(BaseSettings):
    port: int
    host: str
    log_level: str
    google_api_key: str
    temperature: int = 0
