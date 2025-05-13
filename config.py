import os
from dotenv import load_dotenv
load_dotenv(".env", override=True)
class Settings:
    QDRANT_HOST = os.getenv("QDRANT_HOST", None)
    GOOGLE_GEN_AI_API_KEY = os.getenv("GOOGLE_GEN_AI_API_KEY")

settings = Settings()