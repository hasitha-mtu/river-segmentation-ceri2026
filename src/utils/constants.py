"""Project-wide constants"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Image specifications
ORIGINAL_SIZE = (5280, 3956)
WORKING_SIZE = (512, 512)

# Training
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-4

# Feature extraction
N_LUMINANCE_FEATURES = 8
N_CHROMINANCE_FEATURES = 10
N_TOTAL_FEATURES = 18