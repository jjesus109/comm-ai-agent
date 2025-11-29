import logging

from app.config import Configuration


def configure_logger() -> None:
    conf = Configuration()
    logger = logging.getLogger()
    logger.setLevel(conf.log_level)
    ch = logging.StreamHandler()
    ch.setLevel(conf.log_level)
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s: %(name)s: %(message)s")

    ch.setFormatter(formatter)
    logger.addHandler(ch)
