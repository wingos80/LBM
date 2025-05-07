import logging, platform, sys, os
import settings

COLOR = settings.COLOR

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

# Warn if JAX was imported before os envionrment variables are set
if "jax" in sys.modules:
    logger.warning(
        "JAX already imported, any JAX settings configured may not be applied"
    )

if settings.USE_LIBRARY == "jax":
    # Warn if user tries to use JAX with gpu on windows
    if settings.USE_DEVICE == "cuda" and platform.system() == "Windows":
        logger.warning(f"JAX on windows does not have GPU support but was selected")
        settings.USE_DEVICE = "cpu"
    os.environ["JAX_PLATFORMS"] = settings.USE_DEVICE
    os.environ["JAX_PLATFORM_NAME"] = settings.USE_DEVICE
elif settings.USE_LIBRARY == "numpy":
    if settings.USE_DEVICE == "gpu":
        logger.warning(f"Numpy does not have GPU support")
        settings.USE_DEVICE = "cpu"
