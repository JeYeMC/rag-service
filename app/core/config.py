# app/core/config.py
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):

    # ============================
    # 🔹 APP
    # ============================
    APP_NAME: str = "CRM RAG Service"
    HOST: str = Field("0.0.0.0", env="HOST")
    PORT: int = Field(8000, env="PORT")
    ENV: str = Field("development", env="ENV")

    # ============================
    # 🔹 PINECONE
    # ============================
    PINECONE_API_KEY: str | None = Field(None, env="PINECONE_API_KEY")
    PINECONE_INDEX: str = Field("rag-index", env="PINECONE_INDEX")
    PINECONE_ENV: str = Field("us-east-1", env="PINECONE_ENV")
    PINECONE_CLOUD: str = Field("aws", env="PINECONE_CLOUD")

    # ============================
    # 🔹 EMBEDDINGS
    # ============================
    # Valores posibles: "sentence_transformers" | "openai"
    EMB_PROVIDER: str = Field("sentence_transformers", env="EMB_PROVIDER")

    # Modelo HF para embeddings (coincide con tu .env)
    EMB_MODEL: str = Field(
        "intfloat/multilingual-e5-base",
        env="EMB_MODEL"
    )

    # OpenAI embeddings (solo si cambias provider)
    OPENAI_EMB_MODEL: str = Field(
        "text-embedding-3-large",
        env="OPENAI_EMB_MODEL"
    )

    # ============================
    # 🔹 LLM
    # ============================
    # Valores: "hf_inference" | "openai"
    LLM_PROVIDER: str = Field("hf_inference", env="LLM_PROVIDER")

    # HuggingFace Inference API (coincide con tu .env)
    HF_INFERENCE_API_KEY: str | None = Field(None, env="HF_INFERENCE_API_KEY")
    HF_API_URL: str | None = Field(None, env="HF_API_URL")
    HF_MODEL: str | None = Field(None, env="HF_MODEL")  # AHORA SÍ EXISTE

    # OpenAI LLM (solo si cambias provider)
    OPENAI_API_KEY: str | None = Field(None, env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field("gpt-4o-mini", env="OPENAI_MODEL")  # AHORA SÍ EXISTE

    # ============================
    # 🔹 RE-RANKER / SUMMARIZER
    # ============================
    CROSS_ENCODER_MODEL: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        env="CROSS_ENCODER_MODEL"
    )
    USE_LOCAL_SUMMARIZER: bool = Field(False, env="USE_LOCAL_SUMMARIZER")
    SUMMARIZER_MODEL: str = Field("google/pegasus-xsum", env="SUMMARIZER_MODEL")

    # ============================
    # 🔹 STORAGE PATHS
    # ============================
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    UPLOAD_DIR: Path = Field(
        BASE_DIR / "data" / "uploads",
        env="UPLOAD_DIR"
    )

    # ============================
    # 🔹 MISC
    # ============================
    DEBUG: bool = Field(True, env="DEBUG")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
