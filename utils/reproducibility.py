"""
Reproducibility utilities for the unified basket prediction pipeline.
"""

import random
import numpy as np
import torch
import logging
from typing import Optional, List
import os


logger = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """
    Set random seed for all random number generators to ensure reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Python's random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    logger.info(f"Random seed set to: {seed}")


def ensure_reproducibility(seed: int, deterministic: bool = True) -> None:
    """
    Ensure reproducible behavior across the pipeline.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (may reduce performance)
    """
    # Set random seeds
    set_random_seed(seed)
    
    if deterministic:
        # Make PyTorch operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for deterministic behavior
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Note: Some operations may still be non-deterministic
        # FUTURE_IMPROVEMENT: Add support for torch.use_deterministic_algorithms(True)
        # when it becomes more stable across different PyTorch versions
        
        logger.info("Deterministic behavior enabled")
        logger.warning(
            "Deterministic mode may reduce performance. "
            "Some operations may still be non-deterministic on GPU."
        )
    else:
        # Allow non-deterministic but potentially faster operations
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info("Non-deterministic (performance-optimized) mode enabled")


def get_device(device_str: str = "auto", multi_gpu_config: dict = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device_str: Device specification ("auto", "cpu", "cuda", "cuda:0", etc.)
        multi_gpu_config: Multi-GPU configuration dictionary
        
    Returns:
        torch.device: Primary device object for PyTorch operations
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Auto-selected device: {device} (GPU: {torch.cuda.get_device_name()})")
        else:
            device = torch.device("cpu")
            logger.info("Auto-selected device: CPU (CUDA not available)")
    else:
        device = torch.device(device_str)
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning(f"CUDA device requested ({device}) but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            logger.info(f"Using specified device: {device}")
    
    # Log multi-GPU information if enabled
    if multi_gpu_config and multi_gpu_config.get('enabled', False) and device.type == "cuda":
        log_multi_gpu_info(multi_gpu_config)
    
    return device


def get_available_gpus() -> List[int]:
    """
    Get list of available GPU device IDs.
    
    Returns:
        List of available GPU device IDs
    """
    if not torch.cuda.is_available():
        return []
    
    return list(range(torch.cuda.device_count()))


def validate_gpu_devices(device_ids: List[int] = None) -> List[int]:
    """
    Validate and return valid GPU device IDs.
    
    Args:
        device_ids: Requested GPU device IDs, None for all available
        
    Returns:
        List of valid GPU device IDs
        
    Raises:
        ValueError: If requested device IDs are invalid
    """
    available_gpus = get_available_gpus()
    
    if not available_gpus:
        raise ValueError("No CUDA devices available for multi-GPU training")
    
    if device_ids is None:
        return available_gpus
    
    # Validate requested device IDs
    invalid_devices = [dev_id for dev_id in device_ids if dev_id not in available_gpus]
    if invalid_devices:
        raise ValueError(
            f"Invalid GPU device IDs: {invalid_devices}. "
            f"Available devices: {available_gpus}"
        )
    
    return device_ids


def log_multi_gpu_info(multi_gpu_config: dict) -> None:
    """
    Log multi-GPU configuration and device information.
    
    Args:
        multi_gpu_config: Multi-GPU configuration dictionary
    """
    if not multi_gpu_config.get('enabled', False):
        return
    
    available_gpus = get_available_gpus()
    device_ids = multi_gpu_config.get('device_ids')
    strategy = multi_gpu_config.get('strategy', 'data_parallel')
    
    logger.info("=" * 50)
    logger.info("MULTI-GPU CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"Strategy: {strategy.upper()}")
    logger.info(f"Available GPUs: {len(available_gpus)} -> {available_gpus}")
    
    if device_ids is None:
        device_ids = available_gpus
        logger.info(f"Using all available GPUs: {device_ids}")
    else:
        logger.info(f"Using specified GPUs: {device_ids}")
    
    # Log individual GPU information
    for i, device_id in enumerate(device_ids):
        if device_id in available_gpus:
            gpu_name = torch.cuda.get_device_name(device_id)
            gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            logger.info(f"  GPU {device_id}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.warning(f"  GPU {device_id}: NOT AVAILABLE")
    
    output_device = multi_gpu_config.get('output_device')
    if output_device is not None:
        logger.info(f"Primary output device: {output_device}")
    else:
        logger.info(f"Primary output device: {device_ids} (default)")
    
    logger.info("=" * 50)


def setup_multi_gpu_environment(multi_gpu_config: dict) -> dict:
    """
    Set up multi-GPU environment and return validated configuration.
    
    Args:
        multi_gpu_config: Multi-GPU configuration dictionary
        
    Returns:
        Validated and updated multi-GPU configuration
        
    Raises:
        ValueError: If multi-GPU setup is invalid
    """
    # Make a copy to avoid modifying the original
    config = multi_gpu_config.copy()
    enabled = config.get('enabled', 'auto')
    
    # Handle auto-detection
    if enabled == 'auto':
        logger.info("Auto-detecting multi-GPU configuration...")
        
        if not torch.cuda.is_available():
            logger.info("CUDA not available. Auto-disabling multi-GPU training.")
            config['enabled'] = False
            return config
        
        available_gpus = get_available_gpus()
        if len(available_gpus) < 2:
            logger.info(f"Only {len(available_gpus)} GPU(s) available. Auto-disabling multi-GPU training.")
            config['enabled'] = False
            return config
        
        # Auto-enable multi-GPU
        logger.info(f"Auto-detected {len(available_gpus)} GPUs. Enabling multi-GPU training.")
        config['enabled'] = True
        
        # Set device_ids to all available if not specified
        if config.get('device_ids') is None:
            config['device_ids'] = available_gpus
            logger.info(f"Auto-setting device_ids to all available GPUs: {available_gpus}")
    
    # If explicitly disabled, return early
    if not config.get('enabled', False):
        return config
    
    # Validate CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Disabling multi-GPU training.")
        config['enabled'] = False
        return config
    
    # Validate device IDs
    try:
        device_ids = validate_gpu_devices(config.get('device_ids'))
        config['device_ids'] = device_ids
    except ValueError as e:
        logger.error(f"Multi-GPU validation failed: {e}")
        logger.warning("Falling back to single-GPU training.")
        config['enabled'] = False
        return config
    
    # Validate output device
    output_device = config.get('output_device')
    if output_device is not None and output_device not in device_ids:
        logger.warning(
            f"Output device {output_device} not in device_ids {device_ids}. "
            f"Using {device_ids[0]} as output device."
        )
        config['output_device'] = device_ids[0]
    
    # Check minimum GPU requirement
    if len(device_ids) < 2:
        logger.warning("Less than 2 GPUs available. Disabling multi-GPU training.")
        config['enabled'] = False
        return config
    
    logger.info(f"Multi-GPU setup validated successfully with {len(device_ids)} GPUs")
    return config


def log_system_info() -> None:
    """Log system and library version information for reproducibility."""
    logger.info("=== SYSTEM INFORMATION ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Python version: {'.'.join(map(str, [*__import__('sys').version_info[:3]]))}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: True")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA available: False")
    
    logger.info("=" * 27)


def _worker_init_fn(worker_id):
    """
    Worker initialization function for reproducible data loading.
    This function is defined at module level to avoid pickling issues
    with multiprocessing DataLoader workers.
    
    Args:
        worker_id: Worker process ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_reproducible_dataloader(dataset, batch_size: int, shuffle: bool = True, **kwargs):
    """
    Create a DataLoader with reproducible behavior.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        **kwargs: Additional DataLoader arguments
        
    Returns:
        torch.utils.data.DataLoader: Configured DataLoader
    """
    # Create generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=_worker_init_fn,  # Use module-level function
        generator=generator if shuffle else None,
        **kwargs
    )


class ReproducibilityContext:
    """Context manager for ensuring reproducibility within a code block."""
    
    def __init__(self, seed: int, deterministic: bool = True):
        """
        Initialize reproducibility context.
        
        Args:
            seed: Random seed to use
            deterministic: Whether to enable deterministic mode
        """
        self.seed = seed
        self.deterministic = deterministic
        
        # Store previous states
        self.prev_python_state = None
        self.prev_numpy_state = None
        self.prev_torch_state = None
        self.prev_cuda_state = None
        self.prev_deterministic = None
        self.prev_benchmark = None
    
    def __enter__(self):
        # Save current states
        self.prev_python_state = random.getstate()
        self.prev_numpy_state = np.random.get_state()
        self.prev_torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self.prev_cuda_state = torch.cuda.get_rng_state_all()
        self.prev_deterministic = torch.backends.cudnn.deterministic
        self.prev_benchmark = torch.backends.cudnn.benchmark
        
        # Set reproducible state
        ensure_reproducibility(self.seed, self.deterministic)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous states
        random.setstate(self.prev_python_state)
        np.random.set_state(self.prev_numpy_state)
        torch.set_rng_state(self.prev_torch_state)
        if torch.cuda.is_available() and self.prev_cuda_state is not None:
            torch.cuda.set_rng_state_all(self.prev_cuda_state)
        torch.backends.cudnn.deterministic = self.prev_deterministic
        torch.backends.cudnn.benchmark = self.prev_benchmark


if __name__ == "__main__":
    # Minimal sanity check
    import tempfile
    
    try:
        # Test basic reproducibility functions
        ensure_reproducibility(seed=123, deterministic=True)
        device = get_device("auto")
        log_system_info()
        
        # Test reproducibility context
        with ReproducibilityContext(seed=456):
            x = torch.randn(3, 3)
            print(f"Sample tensor: {x[0, 0].item():.4f}")
        
        print("✓ Reproducibility test completed successfully")
        
    except Exception as e:
        print(f"✗ Reproducibility test failed: {e}") 