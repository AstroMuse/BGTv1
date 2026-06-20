import logging
import sys
from datetime import datetime
import os

# Create log directory if it doesn't exist
log_dir = "./log"
os.makedirs(log_dir, exist_ok=True)

# Per-run log file: timestamp (to the second) + PID so each invocation of the
# script gets its own log instead of appending to a shared daily file.
run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20250407_153012
logging_file_path = os.path.join(log_dir, f"search_pipe_{run_stamp}_{os.getpid()}.log")

# logging_file_path = os.path.join(log_dir, f"server_pipe_test.log")

# Configure handlers
handlers = [logging.FileHandler(logging_file_path), logging.StreamHandler(sys.stdout)]

# Set logging level (DEBUG overrides INFO)
level = logging.INFO
# level = logging.DEBUG

# Configure basic logging
logging.basicConfig(
    level=level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers,
)

# Create logger
logger = logging.getLogger(__name__)

# Example usage
logger.debug("This is a debug message")
logger.info("This is an info message")
