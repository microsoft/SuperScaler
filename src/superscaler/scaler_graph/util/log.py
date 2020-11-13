import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] \
        - %(levelname)s: %(message)s')


def logger(logger_name="default"):
    logger = logging.getLogger(__name__)
    return logger
