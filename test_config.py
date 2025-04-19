# Import necessary library
from config import Config

# Config class
class TestConfig(Config):
    # Changed training settings for testing
    batch_size = 16    # 128 -> 16
    base_epochs = 5    # 100 -> 5
    inc_epochs = 2     # 10 -> 2

