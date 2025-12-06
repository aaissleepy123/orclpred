"""
Learning rate scheduler utilities for the basket prediction pipeline.

This module provides a comprehensive set of learning rate schedulers including
warm-up strategies, cosine annealing, and other advanced scheduling techniques.
"""

import math
import logging
from typing import Dict, Any, Union, List, Optional
import warnings

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Learning rate scheduler with linear warm-up followed by cosine annealing.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of epochs for linear warm-up
        max_epochs: Total number of training epochs
        eta_min: Minimum learning rate (default: 0)
        last_epoch: Index of last epoch (default: -1)
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rates for all parameter groups."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warm-up phase
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class LinearWarmupLR(_LRScheduler):
    """
    Linear warm-up scheduler that increases LR from 0 to base_lr over warmup_epochs.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of epochs for linear warm-up
        last_epoch: Index of last epoch (default: -1)
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rates for all parameter groups."""
        if self.last_epoch >= self.warmup_epochs:
            return self.base_lrs
        else:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]


class CosineWarmupRestartLR(_LRScheduler):
    """
    Cosine annealing with warm restarts and optional warm-up.
    
    Args:
        optimizer: Wrapped optimizer
        T_0: Number of iterations for the first restart
        T_mult: Factor to increase T_i after each restart (default: 1)
        eta_min: Minimum learning rate (default: 0)
        warmup_epochs: Number of epochs for initial warm-up (default: 0)
        last_epoch: Index of last epoch (default: -1)
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        warmup_epochs: int = 0,
        last_epoch: int = -1
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.T_cur = 0
        self.T_i = T_0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rates for all parameter groups."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warm-up phase
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing with restarts
            epoch_in_cycle = self.last_epoch - self.warmup_epochs
            
            # Find current cycle
            self.T_cur = epoch_in_cycle
            cycle = 0
            T_i = self.T_0
            
            while self.T_cur >= T_i:
                self.T_cur -= T_i
                cycle += 1
                T_i *= self.T_mult
            
            self.T_i = T_i
            
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * self.T_cur / self.T_i))
                for base_lr in self.base_lrs
            ]


class PolynomialLR(_LRScheduler):
    """
    Polynomial learning rate decay with optional warm-up.
    
    Args:
        optimizer: Wrapped optimizer
        max_epochs: Total number of training epochs
        power: Power of the polynomial (default: 1.0 for linear decay)
        eta_min: Minimum learning rate (default: 0)
        warmup_epochs: Number of epochs for initial warm-up (default: 0)
        last_epoch: Index of last epoch (default: -1)
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        max_epochs: int,
        power: float = 1.0,
        eta_min: float = 0,
        warmup_epochs: int = 0,
        last_epoch: int = -1
    ):
        self.max_epochs = max_epochs
        self.power = power
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rates for all parameter groups."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warm-up phase
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Polynomial decay phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)  # Clamp to [0, 1]
            factor = (1 - progress) ** self.power
            
            return [
                self.eta_min + (base_lr - self.eta_min) * factor
                for base_lr in self.base_lrs
            ]


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_config: Dict[str, Any],
    max_epochs: int
) -> Optional[_LRScheduler]:
    """
    Create a learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_config: Scheduler configuration dictionary
        max_epochs: Total number of training epochs
        
    Returns:
        Learning rate scheduler instance or None if scheduler is disabled
    """
    if not scheduler_config.get('enabled', True):
        logger.info("Learning rate scheduler disabled")
        return None
    
    scheduler_type = scheduler_config.get('type', 'step').lower()
    
    try:
        if scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 50),
                gamma=scheduler_config.get('gamma', 0.8)
            )
            logger.info(f"Created StepLR scheduler (step_size={scheduler_config.get('step_size', 50)}, gamma={scheduler_config.get('gamma', 0.8)})")
            
        elif scheduler_type == 'multistep':
            milestones = scheduler_config.get('milestones', [50, 100, 150])
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=scheduler_config.get('gamma', 0.1)
            )
            logger.info(f"Created MultiStepLR scheduler (milestones={milestones}, gamma={scheduler_config.get('gamma', 0.1)})")
            
        elif scheduler_type == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=scheduler_config.get('gamma', 0.95)
            )
            logger.info(f"Created ExponentialLR scheduler (gamma={scheduler_config.get('gamma', 0.95)})")
            
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get('T_max', max_epochs),
                eta_min=scheduler_config.get('eta_min', 0)
            )
            logger.info(f"Created CosineAnnealingLR scheduler (T_max={scheduler_config.get('T_max', max_epochs)}, eta_min={scheduler_config.get('eta_min', 0)})")
            
        elif scheduler_type == 'warmup_cosine':
            warmup_epochs = scheduler_config.get('warmup_epochs', 5)
            scheduler = WarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=max_epochs,
                eta_min=scheduler_config.get('eta_min', 0)
            )
            logger.info(f"Created WarmupCosineAnnealingLR scheduler (warmup_epochs={warmup_epochs}, max_epochs={max_epochs}, eta_min={scheduler_config.get('eta_min', 0)})")
            
        elif scheduler_type == 'warmup_linear':
            warmup_epochs = scheduler_config.get('warmup_epochs', 5)
            scheduler = LinearWarmupLR(
                optimizer,
                warmup_epochs=warmup_epochs
            )
            logger.info(f"Created LinearWarmupLR scheduler (warmup_epochs={warmup_epochs})")
            
        elif scheduler_type == 'cosine_restart':
            T_0 = scheduler_config.get('T_0', 50)
            T_mult = scheduler_config.get('T_mult', 1)
            warmup_epochs = scheduler_config.get('warmup_epochs', 0)
            scheduler = CosineWarmupRestartLR(
                optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=scheduler_config.get('eta_min', 0),
                warmup_epochs=warmup_epochs
            )
            logger.info(f"Created CosineWarmupRestartLR scheduler (T_0={T_0}, T_mult={T_mult}, warmup_epochs={warmup_epochs})")
            
        elif scheduler_type == 'polynomial':
            power = scheduler_config.get('power', 1.0)
            warmup_epochs = scheduler_config.get('warmup_epochs', 0)
            scheduler = PolynomialLR(
                optimizer,
                max_epochs=max_epochs,
                power=power,
                eta_min=scheduler_config.get('eta_min', 0),
                warmup_epochs=warmup_epochs
            )
            logger.info(f"Created PolynomialLR scheduler (power={power}, warmup_epochs={warmup_epochs})")
            
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get('mode', 'max'),
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                threshold=scheduler_config.get('threshold', 1e-4),
                min_lr=scheduler_config.get('min_lr', 0)
            )
            logger.info(f"Created ReduceLROnPlateau scheduler (mode={scheduler_config.get('mode', 'max')}, factor={scheduler_config.get('factor', 0.5)}, patience={scheduler_config.get('patience', 10)})")
            
        elif scheduler_type == 'lambda':
            # Custom lambda scheduler
            lambda_func = scheduler_config.get('lambda_func')
            if lambda_func is None:
                # Default: linear decay
                lambda_func = lambda epoch: max(0.1, 1.0 - epoch / max_epochs)
            
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)
            logger.info("Created LambdaLR scheduler with custom function")
            
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
        return scheduler
        
    except Exception as e:
        logger.error(f"Failed to create scheduler '{scheduler_type}': {e}")
        logger.warning("Falling back to no scheduler")
        return None


def get_scheduler_config_template() -> Dict[str, Any]:
    """
    Get a template configuration for all available schedulers.
    
    Returns:
        Dictionary with scheduler configuration templates
    """
    return {
        "enabled": True,
        "type": "warmup_cosine",  # Default to warm-up cosine
        
        # Step scheduler options
        "step_size": 50,
        "gamma": 0.8,
        
        # MultiStep scheduler options
        "milestones": [50, 100, 150],
        
        # Cosine annealing options
        "T_max": None,  # Will be set to max_epochs if None
        "eta_min": 0,
        
        # Warm-up options (for applicable schedulers)
        "warmup_epochs": 5,
        
        # Cosine restart options
        "T_0": 50,
        "T_mult": 1,
        
        # Polynomial decay options
        "power": 1.0,
        
        # ReduceLROnPlateau options
        "mode": "max",  # 'max' for metrics like accuracy, 'min' for loss
        "factor": 0.5,
        "patience": 10,
        "threshold": 1e-4,
        "min_lr": 0,
        
        # Custom lambda function (for lambda scheduler)
        "lambda_func": None
    }


def validate_scheduler_config(config: Dict[str, Any], max_epochs: int) -> Dict[str, Any]:
    """
    Validate and clean up scheduler configuration.
    
    Args:
        config: Scheduler configuration
        max_epochs: Total number of training epochs
        
    Returns:
        Validated configuration
    """
    config = config.copy()  # Don't modify original
    
    scheduler_type = config.get('type', 'step').lower()
    
    # Set T_max to max_epochs if not specified for cosine schedulers
    if scheduler_type in ['cosine', 'warmup_cosine'] and config.get('T_max') is None:
        config['T_max'] = max_epochs
    
    # Validate warmup_epochs
    warmup_epochs = config.get('warmup_epochs', 0)
    if warmup_epochs >= max_epochs:
        logger.warning(f"warmup_epochs ({warmup_epochs}) >= max_epochs ({max_epochs}). Setting warmup_epochs to max_epochs // 5")
        config['warmup_epochs'] = max(1, max_epochs // 5)
    
    # Validate milestones for MultiStepLR
    if scheduler_type == 'multistep':
        milestones = config.get('milestones', [])
        if not milestones or max(milestones) >= max_epochs:
            logger.warning(f"Invalid milestones {milestones} for max_epochs={max_epochs}. Using default.")
            config['milestones'] = [max_epochs // 3, 2 * max_epochs // 3]
    
    return config
