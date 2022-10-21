from pydantic import BaseSettings


class Settings(BaseSettings):
    API_VERSION: str = "0.0"
    API_URL_PREFIX: str = ""
    ROOT_PATH: str = ""

    class Config(BaseSettings.Config):
        env_file = ".env"


settings = Settings()
