from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ES_HOST: str = "localhost"
    ES_PORT: int = 9200
    
    class Config:
        env_file = ".env"

settings = Settings() 