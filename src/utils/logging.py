from pydantic import BaseModel


class LogConfigBase(BaseModel):
    """Logging configuration to be set for the server"""

    LOGGER_NAME: str = "GreenEcoHubBase"
    LOG_FORMAT: str = "%(levelprefix)s | %(asctime)s | %(message)s"
    LOG_LEVEL: str = "DEBUG"
    LOG_FILE: str = "logs/run.log"

    # Logging config
    version: int = 1
    disable_existing_loggers: bool = False
    formatters: dict = {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    handlers: dict = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "file": {
            "formatter": "default",
            "class": "logging.FileHandler",
            "filename": LOG_FILE,
            "encoding": "utf-8",
            "mode": "a",  # Append mode
        },
    }
    loggers: dict = {
        LOGGER_NAME: {"handlers": ["default", "file"], "level": LOG_LEVEL},
    }