# app/core/logger.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from .config import settings

LOG_DIR = settings.BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "rag_service.log"


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("rag_service")
    if logger.handlers:
        return logger  # ya inicializado

    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    ch_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    ch.setFormatter(ch_formatter)

    # Rotating file handler
    fh = RotatingFileHandler(
        str(LOG_FILE),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    fh.setFormatter(fh_formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False

    # 🔹 LOG DE CONFIGURACIÓN DE PROVEEDORES (HF / OpenAI)
    logger.info("🔧 Logger inicializado")
    logger.info(f"🧠 EMBEDDINGS Provider: {settings.EMB_PROVIDER} | Modelo: {settings.EMB_MODEL}")
    logger.info(f"🤖 LLM Provider: {settings.LLM_PROVIDER} | Modelo HF: {settings.HF_MODEL}")

    return logger


logger = _build_logger()
