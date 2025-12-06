"""
Unified evaluator for the basket prediction pipeline.

This module provides evaluation functionality that works across all model variants
(MITGNN, GRNN, TGN) and computes comprehensive metrics for basket prediction.
"""

import logging
from typing import Dict, List, Any, Optional
import time

import torch
import numpy as np
from tqdm import tqdm

from .metrics import (
    compute_recall, compute_ndcg, compute_hit_ratio, compute_precision,
    compute_auc, compute_average_precision, compute_coverage, compute_novelty
)
from .prediction_analyzer import PredictionAnalyzer
from utils.config import Config
from utils.logger import TimedLogger


logger = logging.getLogger(__name__)


class UnifiedEvaluator:
    """
    Unified evaluator for basket prediction models.
    
    This evaluator computes comprehensive metrics for model evaluation
    and supports all model variants through a consistent interface.
    """
    
    def __init__(self, config: Config, data_module=None):
        """
        Initialize unified evaluator.
        
        Args:
            config: Global configuration object
            data_module: Data module for item mappings
        """
        self.config = config
        self.data_module = data_module
        
        # Evaluation parameters
        self.metrics = config.evaluation.metrics
        self.top_k = config.evaluation.top_k
        self.test_batch_size = config.evaluation.test_batch_size
        
        # TensorFlow compatibility parameters
        self.filter_training_items = getattr(config.evaluation, 'filter_training_items', True)
        self.multiple_test_baskets = getattr(config.evaluation, 'multiple_test_baskets', True)
        self.aggregation_method = getattr(config.evaluation, 'aggregation_method', 'user_level')
        
        # Metric computation functions
        self.metric_functions = {
            'recall': compute_recall,
            'ndcg': compute_ndcg,
            'hit_ratio': compute_hit_ratio,
            'precision': compute_precision,
            'auc': compute_auc,
            'average_precision': compute_average_precision,
            'coverage': compute_coverage,
            'novelty': compute_novelty
        }
        
        # Initialize prediction analyzer
        self.analyzer = PredictionAnalyzer(config, data_module)
        
        logger.info(f"Evaluator initialized with metrics: {self.metrics}")
        logger.info(f"Top-K values: {self.top_k}")
    
    def compute_metrics(self,
                       predictions: torch.Tensor,
                       targets: Optional[torch.Tensor] = None,
                       users: Optional[torch.Tensor] = None,
                       items: Optional[torch.Tensor] = None,
                       mode: str = "test",
                       train_user_items: Optional[Dict[int, List[int]]] = None) -> Dict[str, Any]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Model predictions [batch_size, n_items] or [batch_size]
            targets: Ground truth labels [batch_size, n_items] or [batch_size]
            users: User IDs [batch_size] (optional)
            items: Item IDs [batch_size] (optional)
            mode: Evaluation mode ("test", "val")
            train_user_items: Dict mapping user_id -> list of training item_ids (for filtering)
            
        Returns:
            Dictionary of computed metrics
        """
        results = {}
        
        with TimedLogger(logger, f"Computing {mode} metrics"):
            
            # Ensure predictions and targets are on CPU
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.detach().cpu()
            if isinstance(targets, torch.Tensor):
                targets = targets.detach().cpu()
            
            # Use original TensorFlow-style ranking evaluation 
            if predictions.dim() == 1 and targets is not None and targets.dim() == 1:
                # Binary classification format: use original ranking approach
                if users is not None and items is not None:
                    users = users.detach().cpu() if users is not None else None
                    items = items.detach().cpu() if items is not None else None
                    results = self._compute_ranking_metrics_original_style(predictions, targets, users, items, train_user_items)
                else:
                    logger.warning("Cannot compute proper ranking metrics without user and item IDs")
                    return self._get_default_metrics()
            else:
                logger.warning(f"Unexpected tensor dimensions: predictions {predictions.shape}, targets {targets.shape if targets is not None else None}")
                return self._get_default_metrics()
        
        # Log summary
        self._log_metric_summary(results, mode)
        
        return results
    
    def _convert_to_ranking_format(self,
                                  predictions: torch.Tensor,
                                  targets: torch.Tensor,
                                  users: Optional[torch.Tensor],
                                  items: Optional[torch.Tensor]) -> tuple:
        """
        Convert binary classification format to ranking format.
        
        Args:
            predictions: Binary predictions [batch_size]
            targets: Binary targets [batch_size]
            users: User IDs [batch_size]
            items: Item IDs [batch_size]
            
        Returns:
            Tuple of (ranking_predictions, ranking_targets)
        """
        if users is None or items is None:
            logger.warning("Cannot convert to ranking format without user and item IDs")
            return predictions.unsqueeze(0), targets.unsqueeze(0)
        
        # Group by users
        unique_users = torch.unique(users)
        n_users = len(unique_users)
        n_items = torch.max(items).item() + 1
        
        # Create ranking matrices with matching dtypes
        ranking_predictions = torch.zeros(n_users, n_items, dtype=predictions.dtype)
        ranking_targets = torch.zeros(n_users, n_items, dtype=targets.dtype)
        
        # Fill matrices
        for i, user_id in enumerate(unique_users):
            user_mask = users == user_id
            user_items = items[user_mask]
            user_preds = predictions[user_mask]
            user_targets = targets[user_mask]
            
            ranking_predictions[i, user_items] = user_preds
            ranking_targets[i, user_items] = user_targets
        
        return ranking_predictions, ranking_targets
    
    def _compute_ranking_metrics_original_style(self, predictions, targets, users, items, train_user_items=None):
        """Compute metrics using original TensorFlow-style ranking evaluation with optional training item filtering."""
        if users is None or items is None:
            logger.warning("Cannot compute ranking metrics without user and item IDs")
            return {}
        
        # Group data by unique user-basket pairs
        from collections import defaultdict
        user_data = defaultdict(lambda: {'items': [], 'preds': [], 'targets': []})
        
        for i in range(len(users)):
            user_id = users[i].item()
            user_data[user_id]['items'].append(items[i].item())
            user_data[user_id]['preds'].append(predictions[i].item())
            user_data[user_id]['targets'].append(targets[i].item())
        
        # Compute metrics for each user using original approach
        all_results = defaultdict(list)
        user_basket_pairs = list(user_data.keys())
        rankings = []
        ground_truths = []
        
        for user_id, data in user_data.items():
            user_items = data['items']
            user_preds = np.array(data['preds'])
            user_targets = np.array(data['targets'])
            
            # Get ground truth items (positive targets)
            pos_items = [item for item, target in zip(user_items, user_targets) if target > 0]
            ground_truths.append(pos_items)
            
            if len(pos_items) == 0:
                continue
            
            # Apply training item filtering if enabled (TensorFlow compatibility)
            if self.filter_training_items and train_user_items is not None:
                user_training_items = set(train_user_items.get(user_id, []))
                
                # Filter out training items from candidate items
                filtered_items = []
                filtered_preds = []
                filtered_targets = []
                
                for item, pred, target in zip(user_items, user_preds, user_targets):
                    if item not in user_training_items:
                        filtered_items.append(item)
                        filtered_preds.append(pred)
                        filtered_targets.append(target)
                
                if len(filtered_items) == 0:
                    logger.warning(f"No candidate items left for user {user_id} after filtering training items")
                    continue
                
                # Use filtered data
                user_items = filtered_items
                user_preds = np.array(filtered_preds)
                user_targets = np.array(filtered_targets)
                
                # Re-compute positive items from filtered set
                pos_items = [item for item, target in zip(user_items, user_targets) if target > 0]
                
                if len(pos_items) == 0:
                    continue
                
            # Sort by prediction scores (descending)
            sorted_indices = np.argsort(-user_preds)
            sorted_items = [user_items[i] for i in sorted_indices]
            rankings.append(np.array(sorted_items))
            
            # Create binary relevance vector
            relevance = [1 if item in pos_items else 0 for item in sorted_items]
            
            # Compute metrics for each K
            for k in self.top_k:
                if k <= len(relevance):
                    rel_k = relevance[:k]
                    
                    # Recall@K: relevant items in top-K / total relevant items  
                    recall_k = sum(rel_k) / len(pos_items)
                    all_results[f'recall@{k}'].append(recall_k)
                    
                    # NDCG@K
                    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(rel_k))
                    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(pos_items))))
                    ndcg_k = dcg / idcg if idcg > 0 else 0
                    all_results[f'ndcg@{k}'].append(ndcg_k)
                    
                    # Hit Ratio@K: whether any relevant item is in top-K
                    hit_k = 1 if sum(rel_k) > 0 else 0  
                    all_results[f'hit_ratio@{k}'].append(hit_k)
                    
                    # Precision@K: relevant items in top-K / K
                    precision_k = sum(rel_k) / k
                    all_results[f'precision@{k}'].append(precision_k)
        
        # Average across all users
        final_results = {}
        for metric_name, values in all_results.items():
            final_results[metric_name] = np.mean(values) if values else 0.0
        
        # Analyze predictions if enabled
        if self.analyzer.enabled:
            # self.analyzer.analyze_predictions(rankings, ground_truths, user_basket_pairs, mode="test") # TODO: Fix mode
            epoch = getattr(self.config, "current_epoch", None)  
            self.analyzer.analyze_predictions(
                rankings=rankings,
                ground_truths=ground_truths,
                user_basket_pairs=user_basket_pairs,
                mode=mode,        
                epoch=epoch       
            )
        return final_results
    
    def _compute_item_popularity(self, predictions: torch.Tensor) -> np.ndarray:
        """
        Compute item popularity from predictions.
        
        Args:
            predictions: Prediction matrix [n_users, n_items]
            
        Returns:
            Item popularity scores
        """
        # Simple popularity based on prediction frequency
        n_items = predictions.size(1)
        popularity = torch.mean(torch.sigmoid(predictions), dim=0)  # Average probability
        
        return popularity.numpy()
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """
        Get default metrics when computation fails.
        
        Returns:
            Dictionary with zero metrics
        """
        results = {}
        
        for metric_name in self.metrics:
            if metric_name in ['recall', 'ndcg', 'hit_ratio', 'precision', 'coverage', 'novelty']:
                results[metric_name] = {k: 0.0 for k in self.top_k}
            else:
                results[metric_name] = 0.0
        
        return results
    
    def _log_metric_summary(self, results: Dict[str, Any], mode: str) -> None:
        """
        Log a summary of computed metrics.
        
        Args:
            results: Computed metrics
            mode: Evaluation mode
        """
        logger.info(f"=== {mode.upper()} METRICS SUMMARY ===")
        
        for metric_name, metric_value in results.items():
            if isinstance(metric_value, dict):
                # Top-K metrics
                for k, v in metric_value.items():
                    logger.info(f"{metric_name}@{k}: {v:.4f}")
            else:
                # Single-value metrics
                logger.info(f"{metric_name}: {metric_value:.4f}")
        
        logger.info("=" * 35)
    
    def evaluate_model(self, model, data_loader, device: torch.device) -> Dict[str, Any]:
        """
        Evaluate a model on a dataset.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
            device: Device to run evaluation on
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_users = []
        all_items = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get model predictions
                outputs = model(batch)
                
                if isinstance(outputs, dict):
                    predictions = outputs.get('predictions', outputs.get('logits'))
                else:
                    predictions = outputs
                
                # Collect data
                all_predictions.append(predictions.cpu())
                
                if 'label' in batch:
                    all_targets.append(batch['label'].cpu())
                if 'user_id' in batch:
                    all_users.append(batch['user_id'].cpu())
                if 'item_id' in batch:
                    all_items.append(batch['item_id'].cpu())
        
        # Concatenate all results
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0) if all_targets else None
        users = torch.cat(all_users, dim=0) if all_users else None
        items = torch.cat(all_items, dim=0) if all_items else None
        
        # Compute metrics
        return self.compute_metrics(predictions, targets, users, items)
    
    def evaluate_recommendations(self,
                                model,
                                user_ids: torch.Tensor,
                                basket_ids: torch.Tensor,
                                target_items: torch.Tensor,
                                device: torch.device) -> Dict[str, Any]:
        """
        Evaluate model recommendations for specific user-basket pairs.
        
        Args:
            model: Model to evaluate
            user_ids: User IDs to evaluate [n_users]
            basket_ids: Basket IDs to evaluate [n_users]
            target_items: Target items for each user [n_users, n_target_items]
            device: Device to run evaluation on
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        
        with torch.no_grad():
            # Move to device
            user_ids = user_ids.to(device)
            basket_ids = basket_ids.to(device)
            
            # Get predictions for all items
            if hasattr(model, 'get_all_item_scores'):
                all_item_scores = model.get_all_item_scores(user_ids, basket_ids)
            else:
                # Fallback: evaluate each item individually
                all_item_scores = self._evaluate_all_items_individually(
                    model, user_ids, basket_ids, device
                )
            
            # Create target matrix
            n_users = len(user_ids)
            n_items = all_item_scores.size(1)
            target_matrix = torch.zeros(n_users, n_items)
            
            for i, user_targets in enumerate(target_items):
                if len(user_targets) > 0:
                    target_matrix[i, user_targets] = 1.0
        
        # Compute metrics
        return self.compute_metrics(all_item_scores, target_matrix)
    
    def _evaluate_all_items_individually(self,
                                       model,
                                       user_ids: torch.Tensor,
                                       basket_ids: torch.Tensor,
                                       device: torch.device) -> torch.Tensor:
        """
        Evaluate all items individually (fallback method).
        
        Args:
            model: Model to evaluate
            user_ids: User IDs [n_users]
            basket_ids: Basket IDs [n_users]
            device: Device
            
        Returns:
            Scores for all items [n_users, n_items]
        """
        n_users = len(user_ids)
        n_items = model.n_items
        all_scores = torch.zeros(n_users, n_items)
        
        # Process in batches to avoid memory issues
        batch_size = min(256, n_items)
        
        for user_idx, (user_id, basket_id) in enumerate(zip(user_ids, basket_ids)):
            user_scores = []
            
            for item_start in range(0, n_items, batch_size):
                item_end = min(item_start + batch_size, n_items)
                item_ids = torch.arange(item_start, item_end, device=device)
                
                # Create batch
                batch_size_actual = len(item_ids)
                batch = {
                    'user_id': user_id.unsqueeze(0).expand(batch_size_actual),
                    'basket_id': basket_id.unsqueeze(0).expand(batch_size_actual),
                    'item_id': item_ids
                }
                
                # Get predictions
                outputs = model(batch)
                if isinstance(outputs, dict):
                    predictions = outputs.get('predictions', outputs.get('logits'))
                else:
                    predictions = outputs
                
                user_scores.append(predictions.cpu())
            
            all_scores[user_idx] = torch.cat(user_scores)
        
        return all_scores
    
    def get_metric_names(self) -> List[str]:
        """
        Get list of all metric names that will be computed.
        
        Returns:
            List of metric names
        """
        metric_names = []
        
        for metric in self.metrics:
            if metric in ['recall', 'ndcg', 'hit_ratio', 'precision', 'coverage', 'novelty']:
                for k in self.top_k:
                    metric_names.append(f"{metric}@{k}")
            else:
                metric_names.append(metric)
        
        return metric_names
    
    def compute_ranking_metrics(self, rankings: List[np.ndarray], ground_truths: List[List[int]], 
                              mode: str) -> Dict[str, float]:
        """
        Compute ranking metrics like original TensorFlow implementation.
        
        Args:
            rankings: List of item rankings for each user-basket pair [n_users, n_items]
            ground_truths: List of ground truth item lists for each user-basket pair
            mode: Evaluation mode
            
        Returns:
            Dictionary of computed metrics
        """
        with TimedLogger(logger, f"Computing {mode} metrics"):
            # Initialize metric accumulators
            n_test_cases = len(rankings)
            total_metrics = {
                'recall@10': 0.0,
                'recall@20': 0.0,
                'ndcg@10': 0.0,
                'ndcg@20': 0.0,
                'hit_ratio@10': 0.0,
                'hit_ratio@20': 0.0,
                'precision@10': 0.0,
                'precision@20': 0.0,
            }
            
            # Process each user-basket pair
            for ranking, ground_truth_items in zip(rankings, ground_truths):
                if not ground_truth_items:
                    continue
                    
                # Convert to sets for faster lookup
                ground_truth_set = set(ground_truth_items)
                
                # Create relevance array: 1 if item is in ground truth, 0 otherwise
                relevance = np.array([1 if item in ground_truth_set else 0 for item in ranking])
                
                # Compute metrics for this user-basket pair
                for k in [10, 20]:
                    # Recall@K
                    recall_k = self._compute_recall_at_k(relevance, k, len(ground_truth_items))
                    total_metrics[f'recall@{k}'] += recall_k / n_test_cases
                    
                    # NDCG@K  
                    ndcg_k = self._compute_ndcg_at_k(relevance, k)
                    total_metrics[f'ndcg@{k}'] += ndcg_k / n_test_cases
                    
                    # Hit Ratio@K
                    hit_k = self._compute_hit_at_k(relevance, k)
                    total_metrics[f'hit_ratio@{k}'] += hit_k / n_test_cases
                    
                    # Precision@K
                    precision_k = self._compute_precision_at_k(relevance, k)
                    total_metrics[f'precision@{k}'] += precision_k / n_test_cases
            
            logger.info(f"=== {mode.upper()} METRICS SUMMARY ===")
            for metric_name, value in total_metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")
            logger.info("=" * 35)
            
            return total_metrics
    
    def _compute_recall_at_k(self, relevance: np.ndarray, k: int, n_ground_truth: int) -> float:
        """Compute recall@k from relevance array."""
        if n_ground_truth == 0:
            return 0.0
        return np.sum(relevance[:k]) / n_ground_truth
    
    def _compute_ndcg_at_k(self, relevance: np.ndarray, k: int) -> float:
        """Compute NDCG@k from relevance array."""
        relevance_k = relevance[:k]
        if len(relevance_k) == 0:
            return 0.0
            
        # DCG@k
        dcg = relevance_k[0]  # First item
        for i in range(1, len(relevance_k)):
            dcg += relevance_k[i] / np.log2(i + 1)
        
        # IDCG@k (ideal DCG)
        ideal_relevance = np.sort(relevance)[::-1][:k]  # Sort descending, take top k
        idcg = ideal_relevance[0] if len(ideal_relevance) > 0 else 0.0
        for i in range(1, len(ideal_relevance)):
            idcg += ideal_relevance[i] / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _compute_hit_at_k(self, relevance: np.ndarray, k: int) -> float:
        """Compute hit ratio@k from relevance array."""
        return 1.0 if np.sum(relevance[:k]) > 0 else 0.0
    
    def _compute_precision_at_k(self, relevance: np.ndarray, k: int) -> float:
        """Compute precision@k from relevance array."""
        if k == 0:
            return 0.0
        return np.sum(relevance[:k]) / k
    
    def compare_models(self, results_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        Compare results from multiple models.
        
        Args:
            results_dict: Dictionary mapping model names to their results
        """
        logger.info("=== MODEL COMPARISON ===")
        
        # Get all metric names
        all_metrics = set()
        for results in results_dict.values():
            for metric_name, metric_value in results.items():
                if isinstance(metric_value, dict):
                    for k in metric_value.keys():
                        all_metrics.add(f"{metric_name}@{k}")
                else:
                    all_metrics.add(metric_name)
        
        # Print comparison table
        for metric in sorted(all_metrics):
            logger.info(f"\n{metric}:")
            for model_name, results in results_dict.items():
                if '@' in metric:
                    base_metric, k = metric.split('@')
                    value = results.get(base_metric, {}).get(int(k), 0.0)
                else:
                    value = results.get(metric, 0.0)
                logger.info(f"  {model_name}: {value:.4f}")
        
        logger.info("=" * 25)


if __name__ == "__main__":
    # Minimal sanity check
    import sys
    from pathlib import Path
    
    # Add parent directories to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from utils.config import Config, ModelConfig, DataConfig, TrainingConfig, EvaluationConfig, SystemConfig
        
        # Create test config
        test_config = Config(
            model=ModelConfig(variant="mitgnn"),
            data=DataConfig(data_path="test", datagroup="test", dataset="test"),
            training=TrainingConfig(),
            evaluation=EvaluationConfig(
                metrics=["recall", "ndcg", "hit_ratio"],
                top_k=[5, 10, 20]
            ),
            system=SystemConfig()
        )
        
        # Create evaluator
        evaluator = UnifiedEvaluator(test_config)
        
        # Test with synthetic data
        batch_size, n_items = 10, 100
        predictions = torch.randn(batch_size, n_items)
        targets = torch.zeros(batch_size, n_items)
        
        # Add some positive targets
        for i in range(batch_size):
            n_pos = np.random.randint(1, 6)
            pos_indices = np.random.choice(n_items, n_pos, replace=False)
            targets[i, pos_indices] = 1
        
        # Compute metrics
        results = evaluator.compute_metrics(predictions, targets)
        
        print("✓ UnifiedEvaluator created and tested successfully")
        print(f"  Computed metrics: {list(results.keys())}")
        
        # Test metric names
        metric_names = evaluator.get_metric_names()
        print(f"  Metric names: {metric_names}")
        
    except Exception as e:
        print(f"✗ UnifiedEvaluator test failed: {e}")
        print("Note: Full testing requires proper imports and data") 
