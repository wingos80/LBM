import logging
from settings import *
import platform

# Defining the default logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("core initialized")

# Create handler
handler = logging.StreamHandler()

# Custom formatter to apply color to log statements
class ColorFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        if record.levelname == "INFO":
            return f"{COLOR.GREEN}{message}{COLOR.END}"
        elif record.levelname == "WARNING":
            return f"{COLOR.YELLOW}{message}{COLOR.END}"
        elif record.levelname == "ERROR":
            return f"{COLOR.RED}{message}{COLOR.END}"
        
        return message

formatter = ColorFormatter(
    fmt="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d >>> %(message)s"
)
handler.setFormatter(formatter)

# Avoid adding multiple handlers if logger is re-used (important!)
if not logger.hasHandlers():
    logger.addHandler(handler)

if USE_LIBRARY=="jax":
    if USE_DEVICE=="gpu" and platform.system()=="Windows":
        logger.warning(f"JAX on windows does not have GPU support but was selected")
        USE_DEVICE = "cpu"
elif USE_LIBRARY=="numpy":
    if USE_DEVICE=="gpu":
        logger.warning(f"Numpy does not have GPU support")
        USE_DEVICE = "cpu"
        