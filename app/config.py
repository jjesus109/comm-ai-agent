from pydantic_settings import BaseSettings


class Configuration(BaseSettings):
    port: int
    host: str
    log_level: str
    google_api_key: str
    temperature: int = 0
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_phone_number: str
    twilio_phone_number_to: str
