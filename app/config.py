from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Smart Diagnosis API"
    MODEL_PATH: str = "models/model.pkl"
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    EMAIL_USER: str = "youremail@example.com"
    EMAIL_PASS: str = "yourpassword"

    class Config:
        env_file = ".env"

settings = Settings()

