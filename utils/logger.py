"""
Logging utilities for the unified basket prediction pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import time
from datetime import datetime
import torch

def log_system_info():
    """Log system information."""
    logger = get_logger(__name__)
    logger.info(f"System information:")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"GPU count: {torch.cuda.device_count()}")

def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "pytorch_pipeline/logs",
    experiment_name: Optional[str] = None
) -> None:
    """
    Set up logging configuration for the pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file
        log_dir: Directory for log files
        experiment_name: Name for the experiment (used in filename)
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create log directory
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            log_filename = f"{experiment_name}_{timestamp}.log"
        else:
            log_filename = f"basket_prediction_{timestamp}.log"
        
        log_file_path = log_dir_path / log_filename
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging to file: {log_file_path}")
    
    root_logger.info(f"Logging initialized with level: {log_level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


class TimedLogger:
    """Context manager for timing code blocks and logging results."""
    
    def __init__(self, logger: logging.Logger, message: str, level: int = logging.INFO):
        """
        Initialize timed logger.
        
        Args:
            logger: Logger instance to use
            message: Message to log (will append timing info)
            level: Log level to use
        """
        self.logger = logger
        self.message = message
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.level, f"Starting: {self.message}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if exc_type is None:
                self.logger.log(self.level, f"Completed: {self.message} (took {elapsed:.2f}s)")
            else:
                self.logger.log(logging.ERROR, f"Failed: {self.message} (took {elapsed:.2f}s)")


class MetricsLogger:
    """Helper class for logging training and evaluation metrics."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize metrics logger.
        
        Args:
            logger: Logger instance to use
        """
        self.logger = logger
    
    def log_training_metrics(self, epoch: int, metrics: dict, phase: str = "train"):
        """
        Log training metrics.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric names and values
            phase: Training phase (train, val, test)
        """
        # Handle nested metrics dictionaries
        flat_metrics = []
        for k, v in metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat_metrics.append(f"{k}_{sub_k}: {sub_v:.4f}")
            else:
                flat_metrics.append(f"{k}: {v:.4f}")
        metric_str = " | ".join(flat_metrics)
        self.logger.info(f"Epoch {epoch:3d} [{phase:5s}] - {metric_str}")
    
    def log_evaluation_metrics(self, metrics: dict, dataset: str = "test"):
        """
        Log evaluation metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            dataset: Dataset name (test, val, etc.)
        """
        self.logger.info(f"=== {dataset.upper()} EVALUATION RESULTS ===")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, dict):
                # Handle nested metrics (e.g., recall@k)
                for k, v in metric_value.items():
                    self.logger.info(f"{metric_name}@{k}: {v:.4f}")
            else:
                self.logger.info(f"{metric_name}: {metric_value:.4f}")
        self.logger.info("=" * 40)
    
    def log_model_info(self, model, total_params: int):
        """
        Log model information.
        
        Args:
            model: PyTorch model instance
            total_params: Total number of parameters
        """
        self.logger.info(f"Model: {model.__class__.__name__}")
        self.logger.info(f"Total parameters: {total_params:,}")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    # Minimal sanity check
    import tempfile
    import shutil
    
    # Test logging setup
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            setup_logging(
                log_level="INFO",
                log_to_file=True,
                log_dir=temp_dir,
                experiment_name="test"
            )
            
            logger = get_logger(__name__)
            logger.info("Test log message")
            
            # Test timed logger
            with TimedLogger(logger, "test operation"):
                time.sleep(0.1)
            
            # Test metrics logger
            metrics_logger = MetricsLogger(logger)
            metrics_logger.log_training_metrics(1, {"loss": 0.5, "acc": 0.8})
            
            print("✓ Logging test completed successfully")
            
        except Exception as e:
            print(f"✗ Logging test failed: {e}") 