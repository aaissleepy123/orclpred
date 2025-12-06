"""
Unified trainer for the basket prediction pipeline.

This trainer provides a generic training loop that supports all model variants
(MITGNN, GRNN, TGN) through a plugin system for variant-specific behavior.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import os, csv, re
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.config import Config
from utils.logger import MetricsLogger, TimedLogger
from utils.reproducibility import get_device, setup_multi_gpu_environment
from utils.schedulers import create_scheduler, validate_scheduler_config
from evaluator import PredictionAnalyzer


logger = logging.getLogger(__name__)


class UnifiedTrainer:
    """
    Unified trainer for basket prediction models.
    
    This trainer provides a generic training loop that can handle all model
    variants through a flexible plugin system for variant-specific behavior.
    """
    
    def __init__(self, config: Config, device: torch.device):
        """
        Initialize unified trainer.
        
        Args:
            config: Global configuration object
            device: Computing device
        """
        self.config = config
        self.device = device
        
        # Set up multi-GPU configuration
        self.multi_gpu_config = setup_multi_gpu_environment(config.system.multi_gpu)
        self.is_multi_gpu = self.multi_gpu_config.get('enabled', False)
        
        # Training components (set during setup)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.data_module = None
        self.evaluator = None
        self.prediction_analyzer = None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = None
        self.best_epoch = 0
        self.early_stopping_counter = 0
        
        # Logging and tracking
        self.metrics_logger = MetricsLogger(logger)
        self.tensorboard_writer = None
        
        # Output directories
        self.setup_directories()
        
        # Log trainer initialization
        if self.is_multi_gpu:
            device_count = len(self.multi_gpu_config['device_ids'])
            strategy = self.multi_gpu_config['strategy']
            logger.info(f"Trainer initialized for {config.model.variant.upper()} variant with {strategy.upper()} ({device_count} GPUs)")
        else:
            logger.info(f"Trainer initialized for {config.model.variant.upper()} variant (single GPU/CPU)")
                        
    def setup_directories(self) -> None:
        """Set up output directories for checkpoints, logs, etc."""
        self.output_dir = self.config.get_output_directory()
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.tensorboard_dir = self.output_dir / "tensorboard"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
    
    def setup(self, data_module, evaluator) -> None:
        """
        Set up trainer with data module and evaluator.
        
        Args:
            data_module: Data module instance
            evaluator: Evaluator instance
        """
        self.data_module = data_module
        self.evaluator = evaluator
        
        # Create model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Apply multi-GPU wrapper if enabled
        if self.is_multi_gpu:
            self.model = self._wrap_model_for_multi_gpu(self.model)
        
        # Log model information
        # For DataParallel models, get parameters from the module attribute
        model_for_info = self.model.module if hasattr(self.model, 'module') else self.model
        total_params = sum(p.numel() for p in model_for_info.parameters())
        self.metrics_logger.log_model_info(model_for_info, total_params)
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Create loss criterion
        self.criterion = self._create_criterion()
        
        # Set up tensorboard
        if self.config.system.tensorboard.get('enabled', True):
            self.tensorboard_writer = SummaryWriter(
                log_dir=self.tensorboard_dir / f"run_{int(time.time())}"
            )
        
        # Set up prediction analyzer
        self.prediction_analyzer = PredictionAnalyzer(self.config, self.data_module)
        
        logger.info("Trainer setup completed")
    
    def _create_model(self):
        """
        Create model based on variant.
        
        Returns:
            Model instance
        """
        # Import models here to avoid circular imports
        from models import MITGNNModel, GRNNModel, TGNModel
        
        data_config = self.data_module.get_data_config()
        variant = self.config.model.variant.lower()
        
        if variant == "mitgnn":
            model = MITGNNModel(self.config, data_config)
        elif variant == "grnn":
            model = GRNNModel(self.config, data_config)
        elif variant == "tgn":
            model = TGNModel(self.config, data_config)
        else:
            raise ValueError(f"Unknown model variant: {variant}")
        
        logger.info(f"Created {variant.upper()} model")
        return model

    def _unwrap_model(self):
        """Return the underlying model when wrapped in DataParallel, else the model itself."""
        return self.model.module if hasattr(self.model, 'module') else self.model
    
    def _wrap_model_for_multi_gpu(self, model: nn.Module) -> nn.Module:
        """
        Wrap model with DataParallel or DistributedDataParallel for multi-GPU training.
        
        Args:
            model: Model to wrap
            
        Returns:
            Wrapped model for multi-GPU training
        """
        strategy = self.multi_gpu_config.get('strategy', 'data_parallel')
        device_ids = self.multi_gpu_config.get('device_ids')
        output_device = self.multi_gpu_config.get('output_device')
        
        if strategy == 'data_parallel':
            # Use DataParallel for single-node multi-GPU
            wrapped_model = nn.DataParallel(
                model,
                device_ids=device_ids,
                output_device=output_device
            )
            logger.info(f"Model wrapped with DataParallel: devices={device_ids}, output_device={output_device}")
            
        elif strategy == 'distributed_data_parallel':
            # Future implementation for DistributedDataParallel
            logger.warning("DistributedDataParallel not yet implemented. Falling back to DataParallel.")
            wrapped_model = nn.DataParallel(
                model,
                device_ids=device_ids,
                output_device=output_device
            )
            logger.info(f"Model wrapped with DataParallel (fallback): devices={device_ids}, output_device={output_device}")
        else:
            raise ValueError(f"Unknown multi-GPU strategy: {strategy}")
        
        return wrapped_model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay
        )
        logger.info(f"Created Adam optimizer with lr={self.config.training.lr}")
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        scheduler_config = self.config.training.scheduler
        
        # Validate and clean up scheduler configuration
        validated_config = validate_scheduler_config(scheduler_config, self.config.training.epochs)
        
        # Create scheduler using the scheduler factory
        scheduler = create_scheduler(
            optimizer=self.optimizer,
            scheduler_config=validated_config,
            max_epochs=self.config.training.epochs
        )
        
        return scheduler
    
    def _create_criterion(self) -> nn.Module:
        """Create loss criterion."""
        loss_type = self.config.training.loss_type
        
        if loss_type == 'bpr':
            # BPR loss - no criterion needed, computed manually
            logger.info("Using BPR (Bayesian Personalized Ranking) loss")
            return None
        else:
            # Binary cross-entropy for basket prediction
            criterion = nn.BCEWithLogitsLoss()
            logger.info("Created BCEWithLogits loss criterion")
            return criterion
    
    def train(self) -> None:
        """
        Main training loop.
        """
        logger.info(f"Starting training for {self.config.training.epochs} epochs")
        
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch()
            # Validation/evaluation phase
            if epoch % self.config.evaluation.eval_frequency == 0:
                val_metrics = self.evaluate(mode="val")
                
                # Combine metrics for logging
                all_metrics = {**train_metrics, **val_metrics}
                self.metrics_logger.log_training_metrics(epoch, all_metrics, "combined")

                # save epoch history
                test_metrics = None
                if epoch >= getattr(self.config.training, "test_start_epoch", 10**9):
                    test_metrics = self.evaluate(mode="test")
        
                self.prediction_analyzer.log_epoch_metrics(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    test_metrics=test_metrics,
                    train_loss=(train_metrics.get("loss") if isinstance(train_metrics, dict) else None),
                )
                
                # Log to tensorboard
                if self.tensorboard_writer:
                    for name, value in all_metrics.items():
                        if isinstance(value, (int, float)):
                            self.tensorboard_writer.add_scalar(f"metrics/{name}", value, epoch)



                # Check for best model
                self._update_best_model(val_metrics, epoch)
                
                # Early stopping check
                if self._should_early_stop():
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Save checkpoint
            if self.config.training.save_checkpoints and epoch % self.config.training.checkpoint_frequency == 0:
                self._save_checkpoint(epoch)
            
            # Update learning rate
            if self.scheduler:
                # Handle different scheduler types
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau needs a metric value
                    # Use validation metric if available, otherwise skip
                    if epoch % self.config.evaluation.eval_frequency == 0 and 'val_metrics' in locals():
                        scheduler_config = self.config.training.scheduler
                        metric_name = scheduler_config.get('metric', 'recall@20')
                        
                        # Extract metric value for plateau scheduler
                        metric_value = val_metrics.get(metric_name)
                        if metric_value is None and '@' in metric_name:
                            # Try to find nested metric (e.g., recall@20 might be in recall dict)
                            metric_parts = metric_name.split('@')
                            if len(metric_parts) == 2 and metric_parts[0] in val_metrics:
                                metric_value = val_metrics[metric_parts[0]].get(int(metric_parts[1]))
                        
                        if metric_value is not None:
                            self.scheduler.step(metric_value)
                        else:
                            logger.warning(f"Could not find metric {metric_name} for ReduceLROnPlateau scheduler")
                else:
                    # Regular schedulers (step-based)
                    self.scheduler.step()
        
        # Final evaluation and save
        self._final_evaluation()
        self._save_checkpoint(self.current_epoch, is_final=True)
        
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Check if training metrics should be computed
        compute_train_metrics = getattr(self.config.training, 'compute_train_metrics', False)
        
        from collections import defaultdict
        user_basket_to_items = defaultdict(set) if compute_train_metrics else None
        
        if compute_train_metrics and self.current_epoch == 0:
            logger.info("Training metrics computation enabled")
        else:
            logger.info("Training metrics computation disabled")
        
        with TimedLogger(logger, f"Training epoch {self.current_epoch}"):
            progress_bar = tqdm(
                self.data_module.train_loader,
                desc=f"Epoch {self.current_epoch}",
                leave=False
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Collect training data for metrics (before adding negative samples)
                if compute_train_metrics:
                    try:
                        user_ids = batch['user_id'].cpu().numpy()
                        basket_ids = batch['basket_id'].cpu().numpy()
                        item_ids = batch['item_id'].cpu().numpy()
                        
                        for user_id, basket_id, item_id in zip(user_ids, basket_ids, item_ids):
                            user_basket_to_items[(user_id, basket_id)].add(item_id)
                    except Exception as e:
                        if batch_idx == 0:  # Only log once to avoid spam
                            logger.warning(f"Could not collect training metrics data: {e}")
                
                # Add negative sampling for BPR loss
                if self.config.training.loss_type == 'bpr':
                    batch = self._add_negative_samples(batch)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                
                # Compute loss
                loss = self._compute_loss(outputs, batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss/num_batches:.4f}"
                })
        
        # Base metrics
        metrics = {
            'train_loss': total_loss / num_batches if num_batches > 0 else 0.0
        }
        
        # Compute training ranking metrics if enabled
        if compute_train_metrics:
            if user_basket_to_items:
                logger.info(f"Computing training ranking metrics from {len(user_basket_to_items)} user-basket pairs")
                training_ranking_metrics = self._compute_ranking_metrics_from_data(user_basket_to_items, mode="train")
                
                # Add with train_ prefix
                for key, value in training_ranking_metrics.items():
                    metrics[f'train_{key}'] = value
                
                logger.info(f"Training metrics computed: {list(training_ranking_metrics.keys())}")
            else:
                logger.warning(f"Training metrics enabled but no user-basket pairs collected")
        elif not compute_train_metrics:
            logger.debug("Training metrics computation disabled")
        
        return metrics
    
    def _compute_ranking_metrics_from_data(self, user_basket_to_items: dict, mode: str = "train") -> Dict[str, float]:
        """
        Compute ranking metrics from collected user-basket-item data.
        
        Args:
            user_basket_to_items: Dictionary mapping (user_id, basket_id) -> set of item_ids
            mode: Mode for evaluation ("train", "val", "test")
            
        Returns:
            Dictionary of computed metrics
        """
        if not user_basket_to_items:
            return {}
        
        self.model.eval()
        
        all_rankings = []
        all_ground_truths = []
        user_basket_pairs = list(user_basket_to_items.keys())
        
        with torch.no_grad():
            model_for_scores = self._unwrap_model()
            # Process in batches for efficiency
            batch_size = 32
            for i in range(0, len(user_basket_pairs), batch_size):
                batch_pairs = user_basket_pairs[i:i+batch_size]
                
                batch_user_ids = torch.tensor([pair[0] for pair in batch_pairs], 
                                            dtype=torch.long, device=self.device)
                batch_basket_ids = torch.tensor([pair[1] for pair in batch_pairs], 
                                              dtype=torch.long, device=self.device)
                
                # Get scores for ALL items
                try:
                    all_item_scores = model_for_scores.get_all_item_scores(batch_user_ids, batch_basket_ids)
                    all_item_scores = all_item_scores.cpu().numpy()
                    
                    for j, (user_id, basket_id) in enumerate(batch_pairs):
                        item_scores = all_item_scores[j]
                        ground_truth_items = list(user_basket_to_items[(user_id, basket_id)])
                        
                        # No training item filtering for training metrics
                        item_rankings = np.argsort(-item_scores)  # Descending order
                        
                        all_rankings.append(item_rankings)
                        all_ground_truths.append(ground_truth_items)
                        
                except Exception as e:
                    logger.warning(f"Could not compute rankings for batch starting at {i}: {e}")
                    # Add empty rankings to maintain alignment
                    for j, (user_id, basket_id) in enumerate(batch_pairs):
                        ground_truth_items = list(user_basket_to_items[(user_id, basket_id)])
                        all_rankings.append(np.array([]))
                        all_ground_truths.append(ground_truth_items)
        
        # Compute metrics using the evaluator
        if all_rankings and any(len(ranking) > 0 for ranking in all_rankings):
            metrics = self.evaluator.compute_ranking_metrics(
                rankings=all_rankings,
                ground_truths=all_ground_truths,
                mode=mode
            )
        else:
            logger.warning(f"No valid rankings computed for {mode} metrics")
            metrics = {}
        
        self.model.train()  # Switch back to training mode
        return metrics

    def evaluate(self, mode: str = "val") -> Dict[str, Any]:
        """
        Evaluate model.
        
        Args:
            mode: Evaluation mode ("val" or "test")
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        # Use test loader for evaluation
        data_loader = self.data_module.test_loader
        
        with TimedLogger(logger, f"Evaluation ({mode})"):
            with torch.no_grad():
                # Use proper ranking evaluation like original TensorFlow
                metrics = self._evaluate_with_full_ranking(data_loader, mode)
        
        return metrics
    
    def _evaluate_with_full_ranking(self, data_loader, mode: str) -> Dict[str, float]:
        """
        Evaluate using full ranking like original TensorFlow implementation.
        For each user-basket pair, rank ALL items and compute metrics.
        """
        from collections import defaultdict
        
        # Initialize loss tracking
        total_loss = 0.0
        num_loss_batches = 0
        
        # Step 0: Build training user-item mapping for filtering (TensorFlow compatibility)
        train_user_items = {}
        if getattr(self.config.evaluation, 'filter_training_items', True):
            # Extract training user-item mappings from R_u2i matrix
            R_u2i = self.data_module.train_dataset.R_u2i.tocsr()  # Convert to CSR for efficient row slicing
            
            for user_id in range(R_u2i.shape[0]):
                # Get items for this user (non-zero entries in the row)
                user_items = R_u2i.getrow(user_id).nonzero()[1].tolist()
                if user_items:  # Only store if user has training items
                    train_user_items[user_id] = user_items
            
            logger.info(f"Built training item mapping for {len(train_user_items)} users")
        
        # Step 1: Collect all user-basket pairs and their ground truth items + compute loss
        user_basket_to_items = defaultdict(set)
        
        for batch in tqdm(data_loader, desc=f"Collecting {mode} data & computing loss", leave=False):
            batch = self._move_batch_to_device(batch)
            
            # Compute loss for this batch
            try:
                # Add negative sampling for BPR loss if needed
                if self.config.training.loss_type == 'bpr':
                    batch = self._add_negative_samples(batch)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute loss
                loss = self._compute_loss(outputs, batch)
                total_loss += loss.item()
                num_loss_batches += 1
            except Exception as e:
                logger.warning(f"Could not compute {mode} loss for batch: {e}")
            
            # Collect ground truth data
            user_ids = batch['user_id'].cpu().numpy()
            basket_ids = batch['basket_id'].cpu().numpy()
            item_ids = batch['item_id'].cpu().numpy()
            
            for user_id, basket_id, item_id in zip(user_ids, basket_ids, item_ids):
                user_basket_to_items[(user_id, basket_id)].add(item_id)
        
        logger.info(f"Found {len(user_basket_to_items)} unique user-basket pairs for evaluation")
        
        # Step 2: For each user-basket pair, get full item rankings
        all_rankings = []
        all_ground_truths = []
        
        # Process in batches for efficiency
        batch_size = 64
        user_basket_pairs = list(user_basket_to_items.keys())
        
        model_for_scores = self._unwrap_model()
        for i in tqdm(range(0, len(user_basket_pairs), batch_size), 
                     desc=f"Computing rankings ({mode})", leave=False):
            batch_pairs = user_basket_pairs[i:i+batch_size]
            
            # Extract user_ids and basket_ids for this batch
            batch_user_ids = torch.tensor([pair[0] for pair in batch_pairs], 
                                        dtype=torch.long, device=self.device)
            batch_basket_ids = torch.tensor([pair[1] for pair in batch_pairs], 
                                          dtype=torch.long, device=self.device)
            
            # Get scores for ALL items for these user-basket pairs
            # Shape: [batch_size, n_items] 
            all_item_scores = model_for_scores.get_all_item_scores(batch_user_ids, batch_basket_ids)
            all_item_scores = all_item_scores.cpu().numpy()
            
            # For each user-basket pair in this batch
            for j, (user_id, basket_id) in enumerate(batch_pairs):
                item_scores = all_item_scores[j]  # [n_items]
                ground_truth_items = list(user_basket_to_items[(user_id, basket_id)])
                
                # Apply training item filtering if enabled (TensorFlow compatibility)
                if getattr(self.config.evaluation, 'filter_training_items', True) and user_id in train_user_items:
                    user_training_items = set(train_user_items[user_id])
                    
                    # Create filtered item list (exclude training items)
                    candidate_items = []
                    candidate_scores = []
                    
                    for item_id in range(len(item_scores)):
                        if item_id not in user_training_items:
                            candidate_items.append(item_id)
                            candidate_scores.append(item_scores[item_id])
                    
                    if len(candidate_items) == 0:
                        logger.warning(f"No candidate items left for user {user_id} after filtering")
                        continue
                    
                    # Sort candidate items by score (descending)
                    candidate_scores = np.array(candidate_scores)
                    sorted_indices = np.argsort(-candidate_scores)
                    item_rankings = np.array([candidate_items[i] for i in sorted_indices])
                    
                else:
                    # No filtering: rank all items
                    item_rankings = np.argsort(-item_scores)  # Negative for descending order
                
                all_rankings.append(item_rankings)
                all_ground_truths.append(ground_truth_items)
        
        # Step 3: Use prediction analyzer if enabled
        if self.prediction_analyzer.enabled:
            self.prediction_analyzer.analyze_predictions(
                rankings=all_rankings,
                ground_truths=all_ground_truths,
                user_basket_pairs=user_basket_pairs,
                mode=mode,
                epoch=self.current_epoch # add epoch
            )
            
            # Show detailed examples for test mode
            if mode == "test":
                self.prediction_analyzer.show_detailed_examples(mode, num_examples=3)
        
        # Step 4: Compute metrics using the evaluator
        metrics = self.evaluator.compute_ranking_metrics(
            rankings=all_rankings,
            ground_truths=all_ground_truths,
            mode=mode
        )
        
        # Add validation/test loss to metrics
        if num_loss_batches > 0:
            avg_loss = total_loss / num_loss_batches
            metrics[f'{mode}_loss'] = avg_loss
            logger.info(f"{mode.upper()} Loss: {avg_loss:.4f}")
        
        return metrics
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to device."""
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            else:
                moved_batch[key] = value
        return moved_batch
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            outputs: Model outputs
            batch: Input batch
            
        Returns:
            Loss tensor
        """
        loss_type = self.config.training.loss_type
        
        if loss_type == 'bpr':
            # BPR loss computation
            return self._compute_bpr_loss(outputs, batch)
        else:
            # BCE loss computation
            return self._compute_bce_loss(outputs, batch)
    
    def _compute_bce_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
        """Compute BCE loss."""
        # Extract predictions and targets
        if isinstance(outputs, dict):
            predictions = outputs.get('predictions', outputs.get('logits'))
        else:
            predictions = outputs
        
        targets = batch.get('label')
        
        if targets is None:
            raise ValueError("No labels found in batch for loss computation")
        
        # Compute main loss
        main_loss = self.criterion(predictions.squeeze(), targets.float())
        
        # Add any auxiliary losses from the model
        aux_losses = outputs.get('aux_losses', {}) if isinstance(outputs, dict) else {}
        
        total_loss = main_loss
        for aux_name, aux_loss in aux_losses.items():
            total_loss = total_loss + self._reduce_aux_loss(aux_loss, main_loss)
        
        return total_loss
    
    def _compute_bpr_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
        """Compute BPR loss."""
        # For BPR loss, we need positive and negative scores
        # The batch should contain pos_scores and neg_scores
        if 'pos_scores' in outputs and 'neg_scores' in outputs:
            pos_scores = outputs['pos_scores']
            neg_scores = outputs['neg_scores']
        else:
            # If not provided, compute from predictions
            # This requires the batch to have positive and negative items
            raise NotImplementedError("BPR loss requires model to output pos_scores and neg_scores")
        
        # BPR loss: -log(sigmoid(pos_score - neg_score))
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        
        # Add auxiliary losses
        aux_losses = outputs.get('aux_losses', {}) if isinstance(outputs, dict) else {}
        
        total_loss = bpr_loss
        for aux_name, aux_loss in aux_losses.items():
            total_loss = total_loss + self._reduce_aux_loss(aux_loss, bpr_loss)
        
        return total_loss

    def _reduce_aux_loss(self, aux_loss: Any, base_loss: torch.Tensor) -> torch.Tensor:
        """Convert auxiliary loss to a scalar tensor on base loss device, reducing vectors by mean."""
        if not torch.is_tensor(aux_loss):
            aux_loss = torch.tensor(aux_loss, dtype=base_loss.dtype, device=base_loss.device)
        else:
            if aux_loss.device != base_loss.device:
                aux_loss = aux_loss.to(base_loss.device)
            if aux_loss.dim() > 0:
                # DataParallel may gather per-device scalars into a length-N vector; reduce to scalar
                aux_loss = aux_loss.mean()
        return aux_loss
    
    def _add_negative_samples(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Add negative samples to batch for BPR loss.
        
        Args:
            batch: Input batch
            
        Returns:
            Batch with negative item IDs added
        """
        if 'user_id' not in batch:
            return batch
        
        user_ids = batch['user_id']
        neg_item_ids = []
        
        # Sample negative item for each user
        for user_id in user_ids:
            neg_item = self.data_module.sample_negative_item(user_id.item())
            neg_item_ids.append(neg_item)
        
        # Add to batch
        batch['neg_item_id'] = torch.tensor(neg_item_ids, device=user_ids.device)
        
        return batch
    
    def _update_best_model(self, metrics: Dict[str, Any], epoch: int) -> None:
        """Update best model based on evaluation metrics."""
        early_stopping_config = self.config.training.early_stopping
        if not early_stopping_config.get('enabled', False):
            return
        
        metric_name = early_stopping_config.get('metric', 'recall@20')
        mode = early_stopping_config.get('mode', 'max')
        
        # Extract metric value
        current_metric = metrics.get(metric_name)
        if current_metric is None:
            # Try to find nested metric (e.g., recall@20 might be in recall dict)
            metric_parts = metric_name.split('@')
            if len(metric_parts) == 2 and metric_parts[0] in metrics:
                current_metric = metrics[metric_parts[0]].get(int(metric_parts[1]))
        
        if current_metric is None:
            logger.warning(f"Metric {metric_name} not found for early stopping")
            return
        
        # Check if this is the best metric
        is_best = False
        if self.best_metric is None:
            is_best = True
        elif mode == 'max' and current_metric > self.best_metric:
            is_best = True
        elif mode == 'min' and current_metric < self.best_metric:
            is_best = True
        
        if is_best:
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.early_stopping_counter = 0
            
            # Save best model
            self._save_checkpoint(epoch, is_best=True)
            logger.info(f"New best model: {metric_name}={current_metric:.4f}")
        else:
            self.early_stopping_counter += 1
    
    def _should_early_stop(self) -> bool:
        """Check if early stopping should be triggered."""
        early_stopping_config = self.config.training.early_stopping
        if not early_stopping_config.get('enabled', False):
            return False
        
        patience = early_stopping_config.get('patience', 20)
        return self.early_stopping_counter >= patience
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False) -> None:
        """Save model checkpoint."""
        # Get the actual model for saving (unwrap DataParallel if needed)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'config': self.config.to_dict(),
            'multi_gpu_config': self.multi_gpu_config
        }
        
        # Regular checkpoint
        if not is_best and not is_final:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        elif is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pth"
        else:  # is_final
            checkpoint_path = self.checkpoint_dir / "final_model.pth"
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state (handle DataParallel models)
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if resuming training
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric')
        self.best_epoch = checkpoint.get('best_epoch', 0)
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def _final_evaluation(self) -> None:
        """Perform final evaluation on test set."""
        if self.current_epoch >= self.config.training.test_start_epoch:
            logger.info("Performing final test evaluation")
            test_metrics = self.evaluate(mode="test")
            self.metrics_logger.log_evaluation_metrics(test_metrics, "test")
    

if __name__ == "__main__":
    # Minimal sanity check
    print("✓ UnifiedTrainer defined successfully")
    print("Note: Full testing requires model and data components") 
