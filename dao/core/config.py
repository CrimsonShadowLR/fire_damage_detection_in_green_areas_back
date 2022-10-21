from enum import Enum

from pydantic import BaseSettings


class DatabaseOption(str, Enum):
    POSTGRES = "postgres"
    SQLITE = "sqlite"


# Settings management
class Settings(BaseSettings):
    DATABASE: DatabaseOption = DatabaseOption.POSTGRES
    POSTGRES_HOST: str = ""
    POSTGRES_PORT: str = ""
    POSTGRES_USER: str = ""
    POSTGRES_PASSWORD: str = ""
    POSTGRES_NAME: str = ""
    SEARCH_SIMILARITY_THRESHOLD: float = 0.12

    class Config:
        env_file = ".env"


settings = Settings()