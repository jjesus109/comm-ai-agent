import logging
import uuid
from contextvars import ContextVar
from typing import Optional

from app.config import Configuration

# Context variable para almacenar el correlation ID de la petición actual
correlation_id_var: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)


def get_correlation_id() -> Optional[str]:
    """
    Get the correlation ID from the current request context.

    Returns:
        Optional[str]: The correlation ID if available, None otherwise.
    """
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
    """
    Set current correlation ID in the context.
    Args:
        correlation_id (str): Correlation ID to set.
    """
    correlation_id_var.set(correlation_id)


def generate_correlation_id() -> str:
    """
    Generate a new unique correlation ID.

    Returns:
        str: A unique correlation ID in UUID format.
    """
    return str(uuid.uuid4())


class CorrelationIDFormatter(logging.Formatter):
    """
    Custom formatter that includes the correlation ID in the logs.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record including the correlation ID.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message with the correlation ID.
        """
        correlation_id = get_correlation_id()
        if correlation_id:
            record.correlation_id = correlation_id
        else:
            record.correlation_id = "N/A"
        return super().format(record)


def configure_logger() -> None:
    conf = Configuration()
    logger = logging.getLogger()
    logger.setLevel(conf.log_level)
    ch = logging.StreamHandler()
    ch.setLevel(conf.log_level)
    formatter = CorrelationIDFormatter(
        "[%(levelname)s] %(asctime)s [%(correlation_id)s]: %(name)s: %(message)s"
    )

    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger = logging.getLogger("twilio.http_client")
    logger.setLevel(logging.WARNING)
