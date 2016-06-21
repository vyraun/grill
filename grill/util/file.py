import os
import grill.core.config as cfg

# Get the path of a file in the log directory
def logpath(filename):
    return os.path.expanduser(os.path.join(cfg.LOG_DIR, filename))
