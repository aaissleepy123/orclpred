#!/usr/bin/env python3
"""
Main entry point for the Unified Basket Prediction Pipeline.

This script provides a single interface for training, testing, and inference
across all model variants (MITGNN, GRNN, TGN) using YAML configuration.

Usage:
    python3 run.py --config configs/experiment.yaml
    python3 run.py --config configs/experiment.yaml --mode test
    python3 run.py --config configs/experiment.yaml --mode inference
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils import (
    load_config, 
    setup_logging, 
    get_logger,
    ensure_reproducibility,
    get_device,
    log_system_info
)
from data import MITGNNDataModule, GRNNDataModule, TGNDataModule
from trainer import UnifiedTrainer
from evaluator import UnifiedEvaluator


logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Unified Basket Prediction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "inference"],
        default="train",
        help="Pipeline mode: train, test, or inference"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    
    parser.add_argument(
        "--device", 
        type=str,
        default=None,
        help="Override device from config (cpu, cuda, auto)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="Perform a dry run without actual training"
    )
    
    return parser.parse_args()


def create_data_module(config):
    """
    Create appropriate data module based on model variant.
    
    Args:
        config: Configuration object
        
    Returns:
        Data module instance
    """
    variant = config.model.variant.lower()
    
    if variant == "mitgnn":
        return MITGNNDataModule(config)
    elif variant == "grnn":
        return GRNNDataModule(config)
    elif variant == "tgn":
        return TGNDataModule(config)
    else:
        raise ValueError(f"Unknown model variant: {variant}")


def setup_experiment(config, args) -> tuple:
    """
    Set up experiment environment and components.
    
    Args:
        config: Configuration object
        args: Command line arguments
        
    Returns:
        Tuple of (data_module, trainer, evaluator, device)
    """
    # Override config with command line arguments
    if args.output_dir:
        config.paths = config.paths or {}
        config.paths["output_dir"] = args.output_dir
    
    if args.device:
        config.system.device = args.device
    
    if args.verbose:
        config.system.log_level = "DEBUG"
    
    # Set up reproducibility
    ensure_reproducibility(
        seed=config.system.seed,
        deterministic=config.system.deterministic
    )
    
    # Get device (with multi-GPU configuration)
    # device = get_device(config.system.device, config.system.multi_gpu)
    device = get_device(getattr(config.system, "device", "auto"))

    # Log system information
    log_system_info()
    
    # Create data module
    logger.info(f"Creating {config.model.variant.upper()} data module")
    data_module = create_data_module(config)
    data_module.setup()
    data_module.log_statistics()
    
    # Create trainer
    trainer = UnifiedTrainer(config, device)
    
    # Create evaluator
    evaluator = UnifiedEvaluator(config, data_module)
    
    return data_module, trainer, evaluator, device


def run_training(config, data_module, trainer, evaluator, device, args):
    """
    Run training pipeline.
    
    Args:
        config: Configuration object
        data_module: Data module instance
        trainer: Trainer instance
        evaluator: Evaluator instance
        device: Computing device
        args: Command line arguments
    """
    logger.info("=" * 50)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 50)
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual training will be performed")
        return
    
    # Set up trainer
    trainer.setup(data_module, evaluator)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming training from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()
    
    logger.info("Training completed successfully")


def run_testing(config, data_module, trainer, evaluator, device, args):
    """
    Run testing pipeline.
    
    Args:
        config: Configuration object
        data_module: Data module instance
        trainer: Trainer instance
        evaluator: Evaluator instance
        device: Computing device
        args: Command line arguments
    """
    logger.info("=" * 50)
    logger.info("STARTING TESTING PIPELINE")
    logger.info("=" * 50)
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual testing will be performed")
        return
    
    # Load model checkpoint
    checkpoint_path = args.resume
    if not checkpoint_path:
        # Look for latest checkpoint in output directory
        output_dir = Path(config.get_output_directory())
        checkpoint_dir = output_dir / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pth"))
            if checkpoints:
                checkpoint_path = max(checkpoints, key=lambda x: x.stat().st_mtime)
                logger.info(f"Using latest checkpoint: {checkpoint_path}")
    
    if not checkpoint_path:
        raise ValueError("No checkpoint specified and no checkpoints found")
    
    # Set up trainer and load model
    trainer.setup(data_module, evaluator)
    trainer.load_checkpoint(checkpoint_path)
    
    # Run evaluation
    test_metrics = trainer.evaluate(mode="test")
    
    # Log results
    logger.info("=" * 50)
    logger.info("TESTING RESULTS")
    logger.info("=" * 50)
    
    for metric_name, metric_value in test_metrics.items():
        if isinstance(metric_value, dict):
            for k, v in metric_value.items():
                logger.info(f"{metric_name}@{k}: {v:.4f}")
        else:
            logger.info(f"{metric_name}: {metric_value:.4f}")


def run_inference(config, data_module, trainer, evaluator, device, args):
    """
    Run inference pipeline.
    
    Args:
        config: Configuration object
        data_module: Data module instance
        trainer: Trainer instance
        evaluator: Evaluator instance
        device: Computing device
        args: Command line arguments
    """
    logger.info("=" * 50)
    logger.info("STARTING INFERENCE PIPELINE")
    logger.info("=" * 50)
    
    # FUTURE_IMPROVEMENT: Implement inference pipeline for production use
    logger.warning("Inference mode not yet implemented")
    logger.info("This would include:")
    logger.info("- Loading trained model")
    logger.info("- Processing new user-basket data")
    logger.info("- Generating recommendations")
    logger.info("- Saving results")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Set up logging
        experiment_name = f"{config.model.variant}_{config.data.dataset}"
        setup_logging(
            log_level=config.system.log_level,
            log_to_file=config.system.log_to_file,
            log_dir=config.system.log_dir,
            experiment_name=experiment_name
        )
        
        logger.info("=" * 60)
        logger.info("UNIFIED BASKET PREDICTION PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Mode: {args.mode.upper()}")
        logger.info(f"Model variant: {config.model.variant.upper()}")
        logger.info(f"Dataset: {config.data.dataset}")
        logger.info(f"Configuration: {args.config}")
        
        # Set up experiment
        data_module, trainer, evaluator, device = setup_experiment(config, args)
        
        # Run appropriate pipeline
        if args.mode == "train":
            run_training(config, data_module, trainer, evaluator, device, args)
        elif args.mode == "test":
            run_testing(config, data_module, trainer, evaluator, device, args)
        elif args.mode == "inference":
            run_inference(config, data_module, trainer, evaluator, device, args)
        
        logger.info("Pipeline completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 
