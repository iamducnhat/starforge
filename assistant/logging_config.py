import logging
import os
import sys


def setup_logging(verbose: bool = False):
    """
    Configures the logging system for the assistant.
    Internal logs are sent to stderr or a file, while UI remains on stdout.
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Console handler for stderr (internal logs)
    # We use stderr so stdout remains clean for tool outputs and markdown
    console_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(
        logging.WARNING
    )  # Only warnings/errors to stderr by default

    if verbose:
        console_handler.setLevel(logging.DEBUG)

    root_logger.addHandler(console_handler)

    # Optional file logging if environment variable is set
    log_file = os.getenv("ASSISTANT_LOG_FILE")
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            file_handler.setLevel(level)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(
                f"Warning: Could not set up log file {log_file}: {e}", file=sys.stderr
            )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
