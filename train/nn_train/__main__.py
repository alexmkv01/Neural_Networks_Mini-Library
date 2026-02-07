"""Entry point for the nn_train pipeline stages."""

from logging.config import dictConfig

_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
    "loggers": {},
    "formatters": {
        "default": {
            "format": " {asctime} {module:10} {levelname:8}  {message}",
            "style": "{",
            "validate": True,
        }
    },
}
dictConfig(_LOGGING)
