"""
Configuration management for the unified basket prediction pipeline.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field, validator
import logging


logger = logging.getLogger(__name__)


class GRNNConfig(BaseModel):
    """GRNN-specific configuration."""
    rnn_hidden_dim: int = Field(64, description="RNN hidden dimension")
    rnn_num_layers: int = Field(1, description="Number of RNN layers")
    sequence_max_length: int = Field(10, description="Maximum sequence length")
    rnn_type: str = Field("LSTM", description="RNN type: LSTM or GRU")
    sequential_weight: float = Field(1.0, description="Weight for sequential features")
    
    @validator('rnn_type')
    def validate_rnn_type(cls, v):
        if v not in ['LSTM', 'GRU']:
            raise ValueError(f"rnn_type must be 'LSTM' or 'GRU', got '{v}'")
        return v


class TGNConfig(BaseModel):
    """TGN-specific configuration."""
    temporal_dim: int = Field(2, description="Temporal embedding dimension")
    temporal_encoding: str = Field("cosine", description="Temporal encoding method")
    temporal_normalization: str = Field("raw", description="Temporal normalization type")
    temporal_weight: float = Field(1.0, description="Weight for temporal features")
    
    @validator('temporal_encoding')
    def validate_temporal_encoding(cls, v):
        if v not in ['cosine', 'sinusoidal', 'learned']:
            raise ValueError(f"temporal_encoding must be one of: cosine, sinusoidal, learned")
        return v


class ModelConfig(BaseModel):
    """Model configuration schema."""
    variant: str = Field(..., description="Model variant: mitgnn, grnn, or tgn")
    embed_size: int = Field(64, description="Embedding dimension")
    layer_sizes: List[int] = Field([64, 64, 64], description="Layer sizes for GNN")
    adj_type: str = Field("norm", description="Adjacency matrix type")
    alg_type: str = Field("intent_conv", description="Algorithm type")
    num_intent: int = Field(5, description="Number of intents")
    sigma: float = Field(0.9, description="User-basket factor")
    # Prediction head control ('mlp' or 'dot_product')
    prediction_method: str = Field("mlp", description="Prediction head: 'mlp' or 'dot_product'")
    # Multimodal features
    use_price: bool = Field(False, description="Use price embedding for items if available")
    
    # Variant-specific configurations
    grnn: Optional[GRNNConfig] = None
    tgn: Optional[TGNConfig] = None
    
    @validator('variant')
    def validate_variant(cls, v):
        if v not in ['mitgnn', 'grnn', 'tgn']:
            raise ValueError(f"Invalid variant: {v}. Must be one of: mitgnn, grnn, tgn")
        return v


class DataConfig(BaseModel):
    """Data configuration schema."""
    data_path: str = Field(..., description="Path to data directory")
    datagroup: str = Field("groceries_data", description="Data group")
    dataset: str = Field("grocerieswithres", description="Dataset name")
    within_basket: bool = Field(True, description="Within-basket prediction flag")
    batch_size: int = Field(256, description="Training batch size")
    num_workers: int = Field(4, description="Number of data loader workers")
    pin_memory: bool = Field(True, description="Pin memory flag")
    sliding_window: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "enabled": False,
            "length": 5,
            "stride": 1,
            "mode": "baskets",  # "baskets" | "days"
            "include_current_basket": False,
            "aggregates": {
                "enabled": True,
                "features": [
                    "num_baskets",
                    "num_unique_items",
                    "avg_items_per_basket",
                    "time_since_last",
                    "median_time_gap"
                ],
                "decay_alpha": 0.8,
                "max_windows_per_user": 10
            }
        },
        description="Sliding-window configuration"
    )


class TrainingConfig(BaseModel):
    """Training configuration schema."""
    lr: float = Field(0.0001, description="Learning rate")
    weight_decay: float = Field(0.001, description="Weight decay")
    epochs: int = Field(200, description="Number of training epochs")
    test_start_epoch: int = Field(150, description="Epoch to start testing")
    loss_type: str = Field("bpr", description="Loss function type: 'bpr' (Bayesian Personalized Ranking) or 'bce' (Binary Cross Entropy)")
    dropout: float = Field(0.1, description="Dropout rate")
    node_dropout: float = Field(0.1, description="Node dropout rate")
    message_dropout: float = Field(0.1, description="Message dropout rate")
    verbose: int = Field(1, description="Evaluation interval")
    early_stopping: Dict[str, Any] = Field(default_factory=dict)
    save_checkpoints: bool = Field(True, description="Save model checkpoints")
    checkpoint_frequency: int = Field(50, description="Checkpoint frequency")
    compute_train_metrics: bool = Field(True, description="Compute training metrics")
    
    # Learning rate scheduler configuration
    scheduler: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "type": "step",
            "step_size": 50,
            "gamma": 0.8
        },
        description="Learning rate scheduler configuration"
    )
    
    @validator('loss_type')
    def validate_loss_type(cls, v):
        """Validate loss_type is either 'bpr' or 'bce'."""
        if v not in ['bpr', 'bce']:
            raise ValueError(f"loss_type must be 'bpr' or 'bce', got '{v}'")
        return v
    
    @validator('scheduler')
    def validate_scheduler_config(cls, v):
        """Validate scheduler configuration."""
        if not isinstance(v, dict):
            raise ValueError("scheduler must be a dictionary")
        
        # Valid scheduler types
        valid_types = [
            'step', 'multistep', 'exponential', 'cosine', 'warmup_cosine', 
            'warmup_linear', 'cosine_restart', 'polynomial', 'plateau', 'lambda'
        ]
        
        scheduler_type = v.get('type', 'step').lower()
        if scheduler_type not in valid_types:
            raise ValueError(f"Invalid scheduler type: {scheduler_type}. Must be one of: {valid_types}")
        
        # Validate warmup_epochs if present
        warmup_epochs = v.get('warmup_epochs', 0)
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        
        # Validate gamma for applicable schedulers
        if scheduler_type in ['step', 'multistep', 'exponential']:
            gamma = v.get('gamma', 0.8)
            if not (0 < gamma <= 1):
                raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        
        return v


class EvaluationConfig(BaseModel):
    """Evaluation configuration schema."""
    metrics: List[str] = Field(["recall", "ndcg", "hit_ratio", "precision"])
    top_k: List[int] = Field([5, 10, 20, 50])
    test_batch_size: int = Field(1000, description="Test batch size")
    eval_frequency: int = Field(1, description="Evaluation frequency")
    prediction_analysis: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "enabled": False,
            "max_examples": 50,
            "top_k_show": [5, 10, 20],
            "save_to_file": True,
            "show_item_names": True
        },
        description="Prediction analysis configuration for model visibility"
    )


class SystemConfig(BaseModel):
    """System configuration schema."""
    device: str = Field("auto", description="Device to use")
    seed: int = Field(123, description="Random seed")
    deterministic: bool = Field(True, description="Deterministic behavior")
    log_level: str = Field("INFO", description="Logging level")
    log_to_file: bool = Field(True, description="Log to file flag")
    log_dir: str = Field("pytorch_pipeline/logs", description="Log directory")
    tensorboard: Dict[str, Any] = Field(default_factory=dict)
    wandb: Dict[str, Any] = Field(default_factory=dict)
    
    # Multi-GPU configuration
    multi_gpu: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": "auto",  # "auto", True, or False - auto enables if 2+ GPUs available
            "strategy": "data_parallel",  # "data_parallel" or "distributed_data_parallel"
            "device_ids": None,  # List of GPU device IDs, None for all available
            "output_device": None,  # Primary device for outputs, None for device 0
            "find_unused_parameters": False  # For DDP only
        },
        description="Multi-GPU training configuration"
    )
    
    @validator('multi_gpu')
    def validate_multi_gpu_config(cls, v):
        """Validate multi-GPU configuration."""
        enabled = v.get('enabled', 'auto')
        
        # Validate enabled field
        if enabled not in ['auto', True, False]:
            raise ValueError(f"Invalid multi_gpu.enabled: {enabled}. Must be 'auto', True, or False")
        
        # Validate strategy if multi-GPU might be enabled
        if enabled == 'auto' or enabled is True:
            strategy = v.get('strategy', 'data_parallel')
            if strategy not in ['data_parallel', 'distributed_data_parallel']:
                raise ValueError(f"Invalid multi_gpu strategy: {strategy}. Must be 'data_parallel' or 'distributed_data_parallel'")
        
        return v


class Config(BaseModel):
    """Main configuration class."""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    system: SystemConfig
    plugins: Optional[Dict[str, Any]] = None
    paths: Optional[Dict[str, Any]] = None
    future_improvements: Optional[Dict[str, Any]] = None

    @validator('model')
    def validate_model_consistency(cls, v, values):
        """Validate model configuration consistency."""
        # Additional validation can be added here
        return v

    def get_data_directory(self) -> Path:
        """Get the full data directory path."""
        within_str = "within_basket" if self.data.within_basket else "next_basket"
        return Path(self.data.data_path) / self.data.datagroup / self.data.dataset / within_str

    def get_output_directory(self) -> Path:
        """Get the output directory path."""
        if self.paths and "output_dir" in self.paths:
            return Path(self.paths["output_dir"])
        return Path("pytorch_pipeline/outputs")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.dict()


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries, with override values taking precedence.
    
    Args:
        base: Base dictionary
        override: Override dictionary
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def _load_config_with_inheritance(config_path: Union[str, Path], 
                                  configs_root: Path = None,
                                  _visited: set = None) -> dict:
    """
    Load configuration with inheritance support.
    
    Args:
        config_path: Path to configuration file
        configs_root: Root directory for configs (for resolving relative paths)
        _visited: Set of visited config files (to prevent circular imports)
        
    Returns:
        Merged configuration dictionary
    """
    if _visited is None:
        _visited = set()
    
    config_path = Path(config_path)
    
    # Set configs root if not provided
    if configs_root is None:
        configs_root = config_path.parent.parent if config_path.parent.name == "base" else config_path.parent
    
    # Resolve absolute path for circular import detection
    abs_config_path = config_path.resolve()
    if abs_config_path in _visited:
        raise ValueError(f"Circular config inheritance detected: {abs_config_path}")
    _visited.add(abs_config_path)
    
    # Load current config
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        current_config = yaml.safe_load(f) or {}
    
    # Check for base config inheritance
    if 'base_config' in current_config:
        base_config_path = configs_root / current_config['base_config']
        
        # Recursively load base config
        base_config = _load_config_with_inheritance(
            base_config_path, 
            configs_root, 
            _visited.copy()  # Pass copy to avoid modifying original set
        )
        
        # Remove base_config key from current config
        current_config_without_base = {k: v for k, v in current_config.items() if k != 'base_config'}
        
        # Merge base config with current config (current takes precedence)
        merged_config = _deep_merge_dicts(base_config, current_config_without_base)
        
        logger.debug(f"Loaded config {config_path} with base {base_config_path}")
        return merged_config
    else:
        logger.debug(f"Loaded config {config_path} (no inheritance)")
        return current_config


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Config: Validated configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        # Load config with inheritance support
        config_dict = _load_config_with_inheritance(config_path)
        
        # Create and validate config
        config = Config(**config_dict)
        logger.info(f"Successfully loaded configuration from {config_path}")
        logger.info(f"Training loss_type: {config.training.loss_type}")
        
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration: {e}")
    except Exception as e:
        raise ValueError(f"Failed to validate configuration: {e}")


def save_config(config: Config, output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        output_path: Path to save the configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration saved to {output_path}")


if __name__ == "__main__":
    # Minimal sanity check
    from pathlib import Path
    
    # Test loading the default config
    config_path = Path(__file__).parent.parent / "configs" / "experiment.yaml"
    
    if config_path.exists():
        try:
            config = load_config(config_path)
            print(f"✓ Successfully loaded config with variant: {config.model.variant}")
            print(f"✓ Data directory: {config.get_data_directory()}")
        except Exception as e:
            print(f"✗ Failed to load config: {e}")
    else:
        print(f"✗ Config file not found: {config_path}") 