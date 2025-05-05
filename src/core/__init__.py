print("in core init")
import logging
from settings import *
import platform

# # Uncomment this to expose all logs from imported libraries
# logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",)

# Defining the default logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("core initialized")

# Create handler
handler = logging.StreamHandler()

# Custom formatter to apply green only to INFO messages
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
    fmt="%(asctime)s [%(levelname)s] %(filename)s:line %(lineno)d >>> %(message)s"
)
handler.setFormatter(formatter)

# Avoid adding multiple handlers if logger is re-used (important!)
if not logger.hasHandlers():
    logger.addHandler(handler)

if USE_LIBRARY=="jax":
    if USE_DEVICE=="gpu" and platform.system()=="Windows":
        logger.warning(f"JAX on windows does not have GPU support")
        USE_DEVICE = "cpu"
elif USE_LIBRARY=="numpy":
    if USE_DEVICE=="gpu":
        logger.warning(f"Numpy does not have GPU support")
        USE_DEVICE = "cpu"
# else:
#     logger.error(f"Unknown library: {USE_LIBRARY=}, defaulting to JAX with CPU")
#     USE_LIBRARY = "jax"
#     USE_DEVICE = "cpu"