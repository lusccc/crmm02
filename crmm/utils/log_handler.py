import logging


class LogFormatHandler(logging.StreamHandler):
    # https://stackoverflow.com/questions/15870380/python-custom-logging-across-all-modules
    def __init__(self):
        logging.StreamHandler.__init__(self)
        # fmt = '[%(levelname)-2s%(asctime)s %(filename)-12s]: %(message)s'
        # fmt_date = '%Y-%m-%dT%T%Z'
        # formatter = logging.Formatter(fmt, fmt_date)
        self.setFormatter(ColorFormatter())


class ColorFormatter(logging.Formatter):
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    blue = "\x1b[34m"
    bold = "\x1b[1m"
    reset = "\x1b[0m"
    format = bold + blue + "%(asctime)s [%(levelname).1s] (%(filename)s:%(lineno)d):" + reset + " %(message)s"

    FORMATS = {
        logging.DEBUG: yellow + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)
