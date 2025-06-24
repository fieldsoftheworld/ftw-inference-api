import json
import logging
import logging.handlers
import socket
from contextvars import ContextVar
from typing import Optional

import watchtower

from app.core.config import get_settings

request_id: ContextVar[str] = ContextVar("request_id", default="")
client_ip: ContextVar[str] = ContextVar("client_ip", default="")
endpoint: ContextVar[str] = ContextVar("endpoint", default="")


class ContextFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id.get("")
        record.client_ip = client_ip.get("")
        record.endpoint = endpoint.get("")
        record.hostname = socket.gethostname()
        return True


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "hostname": getattr(record, "hostname", ""),
            "pathname": record.pathname,
            "lineno": record.lineno,
        }

        if hasattr(record, "request_id") and record.request_id:
            log_entry["request_id"] = record.request_id

        if hasattr(record, "client_ip") and record.client_ip:
            log_entry["client_ip"] = record.client_ip

        if hasattr(record, "endpoint") and record.endpoint:
            log_entry["endpoint"] = record.endpoint

        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms

        if hasattr(record, "status_code"):
            log_entry["status_code"] = record.status_code

        if hasattr(record, "method"):
            log_entry["method"] = record.method

        if hasattr(record, "ml_metrics"):
            log_entry["ml_metrics"] = record.ml_metrics

        if hasattr(record, "ml_context"):
            log_entry["ml_context"] = record.ml_context

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class AppLogger:
    _instance: Optional["AppLogger"] = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.settings = get_settings()
        self._setup_logging()

    def _setup_logging(self):
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.settings.logging.level.upper()))

        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        context_filter = ContextFilter()

        if self.settings.logging.format == "json":
            formatter = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)

        if self.settings.cloudwatch.enabled:
            try:
                cloudwatch_handler = watchtower.CloudWatchLogsHandler(
                    log_group=self.settings.cloudwatch.log_group,
                    log_stream=f"{self.settings.cloudwatch.log_stream_prefix}-{socket.gethostname()}",
                    send_interval=self.settings.cloudwatch.send_interval,
                    max_batch_size=self.settings.cloudwatch.max_batch_size,
                    boto3_session=None,
                    create_log_group=True,
                    create_log_stream=True,
                )
                cloudwatch_handler.setFormatter(formatter)
                cloudwatch_handler.addFilter(context_filter)
                root_logger.addHandler(cloudwatch_handler)
            except Exception as e:
                logging.warning(f"Failed to initialize CloudWatch logging: {e}")

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        cls()
        return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    return AppLogger.get_logger(name)
