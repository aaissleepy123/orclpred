"""
Evaluation metrics for basket prediction models.

This module implements the core metrics used to evaluate basket prediction performance.
Each metric includes detailed documentation about its definition, intuition, and formula.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


logger = logging.getLogger(__name__)


def compute_recall(predictions: torch.Tensor, 
                  targets: torch.Tensor,
                  k_values: List[int] = [5, 10, 20, 50]) -> Dict[int, float]:
    """
    Compute Recall@K for basket prediction.
    
    Definition: Recall@K measures the fraction of relevant items that appear 
    in the top-K recommendations.
    
    Intuition: If a user has 10 relevant items and 6 of them appear in the 
    top-20 recommendations, then Recall@20 = 6/10 = 0.6.
    
    Formula: Recall@K = |Relevant ∩ Retrieved@K| / |Relevant|
    where Retrieved@K is the set of top-K predicted items.
    
    Args:
        predictions: Prediction scores [batch_size, n_items]
        targets: Ground truth binary labels [batch_size, n_items]
        k_values: List of K values to compute recall for
        
    Returns:
        Dictionary mapping K to Recall@K values
    """
    results = {}
    
    # Convert to numpy for easier processing
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    batch_size = predictions.shape[0]
    
    for k in k_values:
        recall_scores = []
        
        for i in range(batch_size):
            # Get top-K predictions
            top_k_indices = np.argsort(predictions[i])[-k:]
            
            # Get relevant items (positive targets)
            relevant_indices = np.where(targets[i] > 0)[0]
            
            if len(relevant_indices) == 0:
                # Skip users with no relevant items
                continue
            
            # Compute recall
            intersection = np.intersect1d(top_k_indices, relevant_indices)
            recall = len(intersection) / len(relevant_indices)
            recall_scores.append(recall)
        
        results[k] = np.mean(recall_scores) if recall_scores else 0.0
    
    return results


def compute_ndcg(predictions: torch.Tensor,
                targets: torch.Tensor, 
                k_values: List[int] = [5, 10, 20, 50]) -> Dict[int, float]:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K).
    
    Definition: NDCG@K measures the quality of ranking by considering both 
    relevance and position, with higher weight given to items ranked higher.
    
    Intuition: NDCG rewards putting relevant items at the top of the ranking.
    Getting a relevant item at position 1 contributes more than at position 10.
    
    Formula: 
    DCG@K = Σ(i=1 to K) (2^rel_i - 1) / log_2(i + 1)
    NDCG@K = DCG@K / IDCG@K
    where IDCG@K is the ideal DCG (best possible ranking).
    
    Args:
        predictions: Prediction scores [batch_size, n_items]
        targets: Ground truth binary labels [batch_size, n_items]
        k_values: List of K values to compute NDCG for
        
    Returns:
        Dictionary mapping K to NDCG@K values
    """
    results = {}
    
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    batch_size = predictions.shape[0]
    
    for k in k_values:
        ndcg_scores = []
        
        for i in range(batch_size):
            # Get top-K predictions
            top_k_indices = np.argsort(predictions[i])[-k:][::-1]  # Descending order
            
            # Get relevance scores for top-K items
            relevance_scores = targets[i][top_k_indices]
            
            # Compute DCG@K
            dcg = 0.0
            for j, rel in enumerate(relevance_scores):
                if rel > 0:
                    dcg += (2**rel - 1) / math.log2(j + 2)  # j+2 because j starts from 0
            
            # Compute IDCG@K (ideal DCG)
            ideal_relevance = np.sort(targets[i])[::-1][:k]  # Top-K relevant items
            idcg = 0.0
            for j, rel in enumerate(ideal_relevance):
                if rel > 0:
                    idcg += (2**rel - 1) / math.log2(j + 2)
            
            # Compute NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        results[k] = np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    return results


def compute_hit_ratio(predictions: torch.Tensor,
                     targets: torch.Tensor,
                     k_values: List[int] = [5, 10, 20, 50]) -> Dict[int, float]:
    """
    Compute Hit Ratio@K (also known as Hit Rate@K).
    
    Definition: Hit Ratio@K measures the fraction of users for whom at least 
    one relevant item appears in the top-K recommendations.
    
    Intuition: This is a binary metric per user - either they get a "hit" 
    (at least one relevant item in top-K) or they don't. It measures the 
    percentage of users who receive at least one good recommendation.
    
    Formula: Hit Ratio@K = (Number of users with hits) / (Total number of users)
    where a hit occurs when |Relevant ∩ Retrieved@K| > 0.
    
    Args:
        predictions: Prediction scores [batch_size, n_items]
        targets: Ground truth binary labels [batch_size, n_items]
        k_values: List of K values to compute hit ratio for
        
    Returns:
        Dictionary mapping K to Hit Ratio@K values
    """
    results = {}
    
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    batch_size = predictions.shape[0]
    
    for k in k_values:
        hits = 0
        valid_users = 0
        
        for i in range(batch_size):
            # Get relevant items
            relevant_indices = np.where(targets[i] > 0)[0]
            
            if len(relevant_indices) == 0:
                # Skip users with no relevant items
                continue
            
            valid_users += 1
            
            # Get top-K predictions
            top_k_indices = np.argsort(predictions[i])[-k:]
            
            # Check if there's a hit
            intersection = np.intersect1d(top_k_indices, relevant_indices)
            if len(intersection) > 0:
                hits += 1
        
        results[k] = hits / valid_users if valid_users > 0 else 0.0
    
    return results


def compute_precision(predictions: torch.Tensor,
                     targets: torch.Tensor,
                     k_values: List[int] = [5, 10, 20, 50]) -> Dict[int, float]:
    """
    Compute Precision@K for basket prediction.
    
    Definition: Precision@K measures the fraction of top-K recommendations 
    that are relevant.
    
    Intuition: If a system recommends 20 items and 8 of them are relevant, 
    then Precision@20 = 8/20 = 0.4. This measures the "purity" of recommendations.
    
    Formula: Precision@K = |Relevant ∩ Retrieved@K| / K
    where K is the number of recommended items.
    
    Args:
        predictions: Prediction scores [batch_size, n_items]
        targets: Ground truth binary labels [batch_size, n_items]
        k_values: List of K values to compute precision for
        
    Returns:
        Dictionary mapping K to Precision@K values
    """
    results = {}
    
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    batch_size = predictions.shape[0]
    
    for k in k_values:
        precision_scores = []
        
        for i in range(batch_size):
            # Get relevant items
            relevant_indices = np.where(targets[i] > 0)[0]
            
            if len(relevant_indices) == 0:
                # Skip users with no relevant items
                continue
            
            # Get top-K predictions
            top_k_indices = np.argsort(predictions[i])[-k:]
            
            # Compute precision
            intersection = np.intersect1d(top_k_indices, relevant_indices)
            precision = len(intersection) / k
            precision_scores.append(precision)
        
        results[k] = np.mean(precision_scores) if precision_scores else 0.0
    
    return results


def compute_auc(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Area Under the ROC Curve (AUC).
    
    Definition: AUC measures the probability that a randomly chosen relevant 
    item is ranked higher than a randomly chosen non-relevant item.
    
    Intuition: AUC evaluates the overall ranking quality across all thresholds.
    A perfect ranker has AUC = 1.0, while random ranking gives AUC = 0.5.
    
    Formula: AUC is the area under the ROC curve, which plots True Positive 
    Rate vs False Positive Rate at various classification thresholds.
    
    Args:
        predictions: Prediction scores [batch_size, n_items]
        targets: Ground truth binary labels [batch_size, n_items]
        
    Returns:
        AUC score
    """
    # Convert to numpy and flatten
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(predictions_flat) | np.isnan(targets_flat))
    predictions_flat = predictions_flat[valid_mask]
    targets_flat = targets_flat[valid_mask]
    
    if len(np.unique(targets_flat)) < 2:
        # Need at least one positive and one negative sample
        return 0.0
    
    try:
        auc_score = roc_auc_score(targets_flat, predictions_flat)
        return float(auc_score)
    except ValueError as e:
        logger.warning(f"AUC computation failed: {e}")
        return 0.0


def compute_average_precision(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Average Precision (AP).
    
    Definition: AP is the average of precision values at each relevant item 
    retrieved, providing a single-number summary of precision-recall performance.
    
    Intuition: AP gives more weight to retrieving relevant items early in the 
    ranking. It's the area under the precision-recall curve.
    
    Formula: AP = Σ(k=1 to n) P(k) × rel(k) / |Relevant|
    where P(k) is precision at rank k, rel(k) = 1 if item at rank k is relevant.
    
    Args:
        predictions: Prediction scores [batch_size, n_items]
        targets: Ground truth binary labels [batch_size, n_items]
        
    Returns:
        Average Precision score
    """
    # Convert to numpy and flatten
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(predictions_flat) | np.isnan(targets_flat))
    predictions_flat = predictions_flat[valid_mask]
    targets_flat = targets_flat[valid_mask]
    
    if len(np.unique(targets_flat)) < 2:
        return 0.0
    
    try:
        ap_score = average_precision_score(targets_flat, predictions_flat)
        return float(ap_score)
    except ValueError as e:
        logger.warning(f"AP computation failed: {e}")
        return 0.0


def compute_coverage(predictions: torch.Tensor,
                    k: int = 20,
                    n_items: Optional[int] = None) -> float:
    """
    Compute Item Coverage@K.
    
    Definition: Coverage@K measures the percentage of unique items that appear 
    in top-K recommendations across all users.
    
    Intuition: Coverage measures diversity - a system with high coverage 
    recommends a wide variety of items rather than just popular ones.
    
    Formula: Coverage@K = |Unique items in all top-K lists| / |Total items|
    
    Args:
        predictions: Prediction scores [batch_size, n_items]
        k: Number of top items to consider
        n_items: Total number of items (if None, inferred from predictions)
        
    Returns:
        Coverage score
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    if n_items is None:
        n_items = predictions.shape[1]
    
    # Collect all recommended items
    recommended_items = set()
    
    for i in range(predictions.shape[0]):
        top_k_indices = np.argsort(predictions[i])[-k:]
        recommended_items.update(top_k_indices)
    
    coverage = len(recommended_items) / n_items
    return float(coverage)


def compute_novelty(predictions: torch.Tensor,
                   item_popularity: np.ndarray,
                   k: int = 20) -> float:
    """
    Compute Novelty@K.
    
    Definition: Novelty measures how "surprising" or "unexpected" the 
    recommendations are based on item popularity.
    
    Intuition: Recommending popular items is easy but not novel. Novelty 
    rewards recommending less popular (but still relevant) items.
    
    Formula: Novelty@K = -Σ(i in top-K) log2(popularity(i)) / K
    where popularity(i) is the fraction of users who interacted with item i.
    
    Args:
        predictions: Prediction scores [batch_size, n_items]
        item_popularity: Item popularity scores [n_items]
        k: Number of top items to consider
        
    Returns:
        Novelty score
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    novelty_scores = []
    
    for i in range(predictions.shape[0]):
        top_k_indices = np.argsort(predictions[i])[-k:]
        
        # Compute novelty for this user
        user_novelty = 0.0
        for item_idx in top_k_indices:
            if item_idx < len(item_popularity):
                pop = max(item_popularity[item_idx], 1e-8)  # Avoid log(0)
                user_novelty += -math.log2(pop)
        
        user_novelty /= k
        novelty_scores.append(user_novelty)
    
    return float(np.mean(novelty_scores))


if __name__ == "__main__":
    # Minimal sanity check with synthetic data
    import torch
    
    # Create synthetic test data
    batch_size, n_items = 10, 100
    predictions = torch.randn(batch_size, n_items)
    targets = torch.zeros(batch_size, n_items)
    
    # Add some positive targets
    for i in range(batch_size):
        n_pos = np.random.randint(1, 6)  # 1-5 positive items per user
        pos_indices = np.random.choice(n_items, n_pos, replace=False)
        targets[i, pos_indices] = 1
    
    # Test metrics
    try:
        recall = compute_recall(predictions, targets, [5, 10, 20])
        ndcg = compute_ndcg(predictions, targets, [5, 10, 20])
        hit_ratio = compute_hit_ratio(predictions, targets, [5, 10, 20])
        precision = compute_precision(predictions, targets, [5, 10, 20])
        auc = compute_auc(predictions, targets)
        ap = compute_average_precision(predictions, targets)
        coverage = compute_coverage(predictions, k=20)
        
        print("✓ All metrics computed successfully")
        print(f"Sample results:")
        print(f"  Recall@20: {recall[20]:.4f}")
        print(f"  NDCG@20: {ndcg[20]:.4f}")
        print(f"  Hit Ratio@20: {hit_ratio[20]:.4f}")
        print(f"  Precision@20: {precision[20]:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  AP: {ap:.4f}")
        print(f"  Coverage@20: {coverage:.4f}")
        
    except Exception as e:
        print(f"✗ Metrics test failed: {e}") 