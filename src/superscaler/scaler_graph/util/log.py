import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] \
- %(levelname)s: %(message)s')


def logger(logger_name="default"):
    logger = logging.getLogger(logger_name)
    return logger
