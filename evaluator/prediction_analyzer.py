"""
Prediction Analysis Module for Basket Prediction Pipeline.

This module provides visibility into model behavior by capturing and analyzing
input data, model predictions, and ground truth during evaluation.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from collections import defaultdict

import torch
import numpy as np
import pandas as pd

from utils.config import Config
from utils.logger import TimedLogger
from pathlib import Path
from datetime import datetime
import re, logging


logger = logging.getLogger(__name__)


class PredictionAnalyzer:
    """
    Analyzes and visualizes model predictions for basket prediction tasks.
    
    This analyzer captures input data, model predictions, and ground truth
    to provide insights into model behavior for both next-basket and 
    within-basket prediction tasks.
    """
    
    def __init__(self, config: Config, data_module):
        """
        Initialize prediction analyzer.
        
        Args:
            config: Global configuration object
            data_module: Data module containing item mappings
        """
        self.config = config
        self.data_module = data_module
        if getattr(config, "paths", None) and getattr(config.paths, "output_dir", None):
            base_out = Path(config.paths.output_dir)
        elif hasattr(config, "get_output_directory"):
            base_out = Path(config.get_output_directory())
        else:
            base_out = Path("./outputs").resolve()
        base_out.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(base_out)
        
        # Analysis configuration
        analysis_config = getattr(config.evaluation, 'prediction_analysis', {})
        if analysis_config is None:
            analysis_config = {}
        
        self.enabled = analysis_config.get('enabled', True)
        self.max_examples = analysis_config.get('max_examples', 100)
        self.top_k_show = analysis_config.get('top_k_show', [5, 10, 20])
        self.save_to_file = analysis_config.get('save_to_file', True)
        self.show_item_names = analysis_config.get('show_item_names', True)
        self.save_raw_scores = analysis_config.get('save_raw_scores', True)  # Save raw prediction scores
        self.save_loss_per_example = analysis_config.get('save_loss_per_example', True)  # Save loss per example
        self.save_history = analysis_config.get("save_history", True)
        
        # Storage for analysis data
        self.prediction_data = []
        
        # Item mapping for readable output
        self.item_mapping = self._load_item_mapping()
        
        logger.info(f"PredictionAnalyzer initialized (enabled: {self.enabled})")
        if self.enabled:
            logger.info(f"Will analyze up to {self.max_examples} examples")
            logger.info(f"Will show top-K for K={self.top_k_show}")

        # save history
        model_name = getattr(config.model, "variant", None) or getattr(config.model, "name", None) or "model"
        datagroup = getattr(config.data, "datagroup", "")
        m = re.search(r"(split[^/]+)", str(datagroup))  
        split_tag = m.group(1) if m else None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag = "_".join([p for p in (model_name, split_tag, ts) if p])
        self.run_tag = run_tag
        
        if hasattr(config, "paths") and hasattr(config.paths, "output_dir") and config.paths.output_dir:
            base_out = Path(config.paths.output_dir)
        elif hasattr(config, "get_output_directory"):
            base_out = Path(config.get_output_directory())
        else:
            base_out = Path.cwd()
            
        hist_dir = Path(self.output_dir) / "prediction_analysis"
        hist_dir.mkdir(parents=True, exist_ok=True)
        self.history_csv = str(hist_dir / f"epoch_history_{self.run_tag}.csv")
        self._history_header = None

        base_out.mkdir(parents=True, exist_ok=True)
        logging.getLogger(__name__).info(f"[history] epoch CSV will be written to: {self.history_csv}")
            
    def _load_item_mapping(self) -> Optional[Dict[int, str]]:
        """Load item ID to name mapping if available."""
        try:
            # Try to load item mapping from data module
            if hasattr(self.data_module, 'item_index_mapping'):
                return self.data_module.item_index_mapping
            
            # Try to load from item mapping file
            data_path = Path(self.config.data.data_path)
            mapping_file = data_path / self.config.data.datagroup / self.config.data.dataset / "item_index_mapping.txt"
            
            if mapping_file.exists():
                item_mapping = {}
                with open(mapping_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            item_id = int(parts[0])
                            item_name = parts[1]
                            item_mapping[item_id] = item_name
                logger.info(f"Loaded item mapping with {len(item_mapping)} items")
                return item_mapping
        except Exception as e:
            logger.warning(f"Could not load item mapping: {e}")
        
        return None
    
    def _get_item_name(self, item_id: int) -> str:
        """Get readable item name from ID."""
        if self.item_mapping and item_id in self.item_mapping:
            return f"{item_id}:{self.item_mapping[item_id]}"
        return str(item_id)
    
    def analyze_predictions(self, 
                          rankings: List[np.ndarray], 
                          ground_truths: List[List[int]],
                          user_basket_pairs: List[Tuple[int, int]],
                          mode: str,
                          epoch: Optional[int] = None,
                          raw_scores: Optional[List[np.ndarray]] = None,
                          losses_per_example: Optional[List[float]] = None) -> None:
        """
        Analyze model predictions and store results.
        
        Args:
            rankings: List of item rankings for each user-basket pair
            ground_truths: List of ground truth item lists
            user_basket_pairs: List of (user_id, basket_id) tuples
            mode: Evaluation mode ("test", "val", "train")
            epoch: Current training epoch (optional)
            raw_scores: Raw prediction scores for each user-basket pair (optional)
            losses_per_example: Loss values per example (optional)
        """
        if not self.enabled:
            return
        
        logger.info(f"Analyzing predictions for {len(rankings)} examples ({mode} mode)")
        
        # Limit number of examples to analyze
        num_examples = min(len(rankings), self.max_examples)
        
        with TimedLogger(logger, f"Prediction analysis ({mode})"):
            for i in range(num_examples):
                ranking = rankings[i]
                ground_truth = ground_truths[i]
                user_id, basket_id = user_basket_pairs[i]
                
                # Get additional data if available
                raw_score = raw_scores[i] if raw_scores and i < len(raw_scores) else None
                loss_value = losses_per_example[i] if losses_per_example and i < len(losses_per_example) else None
                
                # Analyze this prediction
                analysis = self._analyze_single_prediction(
                    ranking, ground_truth, user_id, basket_id, mode, epoch, raw_score, loss_value
                )
                self.prediction_data.append(analysis)
        
        # Log summary and save results
        self._log_analysis_summary(mode)
        if self.save_to_file:
            self._save_analysis_to_file(mode)
    
    def _analyze_single_prediction(self, 
                                 ranking: np.ndarray,
                                 ground_truth: List[int], 
                                 user_id: int,
                                 basket_id: int,
                                 mode: str,
                                 epoch: Optional[int] = None,
                                 raw_scores: Optional[np.ndarray] = None,
                                 loss_value: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze a single prediction example.
        
        Args:
            ranking: Item ranking by model [n_items]
            ground_truth: Ground truth item list
            user_id: User ID
            basket_id: Basket ID
            mode: Evaluation mode
            epoch: Training epoch (optional)
            raw_scores: Raw prediction scores (optional)
            loss_value: Loss value for this example (optional)
            
        Returns:
            Analysis dictionary
        """
        analysis = {
            'mode': mode,
            'user_id': user_id,
            'basket_id': basket_id,
            'epoch': epoch,
            'loss_value': loss_value,
            'ground_truth': ground_truth,
            'ground_truth_names': [self._get_item_name(item) for item in ground_truth] if self.show_item_names else None,
            'ground_truth_count': len(ground_truth),
            'ranking_analysis': {}
        }
        
        # Add raw scores if requested and available
        if self.save_raw_scores and raw_scores is not None:
            # Save top K raw scores to reduce file size
            max_scores_to_save = max(self.top_k_show) * 2  # Save 2x top K for analysis
            top_indices = ranking[:max_scores_to_save]
            analysis['raw_scores'] = {
                'top_item_ids': top_indices.tolist(),
                'top_scores': raw_scores[top_indices].tolist() if isinstance(raw_scores, np.ndarray) else [raw_scores[i] for i in top_indices]
            }
        
        # Analyze different top-K levels
        for k in self.top_k_show:
            if k <= len(ranking):
                top_k_items = ranking[:k].tolist()
                top_k_names = [self._get_item_name(item) for item in top_k_items] if self.show_item_names else None
                
                # Find hits (items in both prediction and ground truth)
                hits = [item for item in top_k_items if item in ground_truth]
                hit_names = [self._get_item_name(item) for item in hits] if self.show_item_names else None
                
                # Calculate metrics for this example
                recall_k = len(hits) / len(ground_truth) if ground_truth else 0.0
                precision_k = len(hits) / k
                hit_ratio_k = 1.0 if hits else 0.0
                
                analysis['ranking_analysis'][f'top_{k}'] = {
                    'predicted_items': top_k_items,
                    'predicted_names': top_k_names,
                    'hits': hits,
                    'hit_names': hit_names,
                    'recall': recall_k,
                    'precision': precision_k,
                    'hit_ratio': hit_ratio_k,
                    'num_hits': len(hits)
                }
        
        return analysis
    
    def _log_analysis_summary(self, mode: str) -> None:
        """Log summary of prediction analysis."""
        if not self.prediction_data:
            return
        
        logger.info(f"=== PREDICTION ANALYSIS SUMMARY ({mode.upper()}) ===")
        
        # Calculate overall statistics
        mode_data = [d for d in self.prediction_data if d['mode'] == mode]
        if not mode_data:
            return
        
        # Ground truth statistics
        gt_counts = [d['ground_truth_count'] for d in mode_data]
        logger.info(f"Ground truth items per example: avg={np.mean(gt_counts):.1f}, "
                   f"min={min(gt_counts)}, max={max(gt_counts)}")
        
        # Epoch and loss statistics if available
        epochs = [d['epoch'] for d in mode_data if d.get('epoch') is not None]
        if epochs:
            logger.info(f"Epoch range: {min(epochs)} - {max(epochs)}")
        
        losses = [d['loss_value'] for d in mode_data if d.get('loss_value') is not None]
        if losses:
            logger.info(f"Loss per example: avg={np.mean(losses):.4f}, "
                       f"min={min(losses):.4f}, max={max(losses):.4f}")
        
        # Raw scores statistics if available
        raw_scores_examples = [d for d in mode_data if d.get('raw_scores') is not None]
        if raw_scores_examples:
            logger.info(f"Raw scores saved for {len(raw_scores_examples)}/{len(mode_data)} examples")
        
        # Top-K hit statistics
        for k in self.top_k_show:
            hits = [d['ranking_analysis'].get(f'top_{k}', {}).get('num_hits', 0) for d in mode_data]
            recalls = [d['ranking_analysis'].get(f'top_{k}', {}).get('recall', 0) for d in mode_data]
            
            if hits:
                logger.info(f"Top-{k}: avg_hits={np.mean(hits):.1f}, "
                           f"avg_recall={np.mean(recalls):.3f}, "
                           f"examples_with_hits={sum(1 for h in hits if h > 0)}/{len(hits)}")
    
    def show_detailed_examples(self, mode: str, num_examples: int = 5) -> None:
        """
        Show detailed prediction examples.
        
        Args:
            mode: Evaluation mode to show examples for
            num_examples: Number of examples to show in detail
        """
        if not self.enabled or not self.prediction_data:
            logger.info("Prediction analysis not enabled or no data available")
            return
        
        mode_data = [d for d in self.prediction_data if d['mode'] == mode]
        if not mode_data:
            logger.info(f"No prediction data available for mode: {mode}")
            return
        
        logger.info(f"=== DETAILED PREDICTION EXAMPLES ({mode.upper()}) ===")
        
        # Show first few examples
        for i, example in enumerate(mode_data[:num_examples]):
            self._show_single_example(example, i + 1)
        
        logger.info("=" * 60)
    
    def _show_single_example(self, example: Dict[str, Any], example_num: int) -> None:
        """Show detailed analysis for a single example."""
        task_type = "within-basket" if self.config.data.within_basket else "next-basket"
        
        logger.info(f"\n--- Example {example_num} ({task_type} prediction) ---")
        logger.info(f"User ID: {example['user_id']}, Basket ID: {example['basket_id']}")
        
        # Show ground truth
        if self.show_item_names and example['ground_truth_names']:
            logger.info(f"Ground Truth ({len(example['ground_truth'])} items): {example['ground_truth_names'][:10]}")
            if len(example['ground_truth']) > 10:
                logger.info(f"  ... and {len(example['ground_truth']) - 10} more items")
        else:
            logger.info(f"Ground Truth ({len(example['ground_truth'])} items): {example['ground_truth'][:10]}")
            if len(example['ground_truth']) > 10:
                logger.info(f"  ... and {len(example['ground_truth']) - 10} more items")
        
        # Show predictions for each top-K
        for k in self.top_k_show:
            k_key = f'top_{k}'
            if k_key in example['ranking_analysis']:
                k_data = example['ranking_analysis'][k_key]
                
                logger.info(f"\nTop-{k} Predictions:")
                if self.show_item_names and k_data['predicted_names']:
                    logger.info(f"  Predicted: {k_data['predicted_names']}")
                else:
                    logger.info(f"  Predicted: {k_data['predicted_items']}")
                
                if k_data['hits']:
                    if self.show_item_names and k_data['hit_names']:
                        logger.info(f"  ✓ Hits ({k_data['num_hits']}): {k_data['hit_names']}")
                    else:
                        logger.info(f"  ✓ Hits ({k_data['num_hits']}): {k_data['hits']}")
                    logger.info(f"  Metrics: Recall={k_data['recall']:.3f}, Precision={k_data['precision']:.3f}")
                else:
                    logger.info(f"  ✗ No hits in top-{k}")
    
    def _save_analysis_to_file(self, mode: str) -> None:
        """Save analysis results to JSON file."""
        try:
            output_dir = Path(self.config.get_output_directory()) / "prediction_analysis"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed analysis
            # analysis_file = output_dir / f"predictions_{mode}.json"
            tag = getattr(self, "run_tag", datetime.now().strftime("%Y%m%d_%H%M%S"))
            analysis_file = output_dir / f"predictions_{mode}_{tag}.json"
            mode_data = [d for d in self.prediction_data if d['mode'] == mode]
            
            with open(analysis_file, 'w') as f:
                json.dump(mode_data, f, indent=2, default=str)
            
            logger.info(f"Saved prediction analysis to: {analysis_file}")
            
            # Save summary statistics
            self._save_summary_statistics(output_dir, mode)
            
        except Exception as e:
            logger.error(f"Failed to save prediction analysis: {e}")
    
    def _save_summary_statistics(self, output_dir: Path, mode: str) -> None:
        """Save summary statistics to CSV file."""
        try:
            mode_data = [d for d in self.prediction_data if d['mode'] == mode]
            if not mode_data:
                return
            
            # Create summary dataframe
            summary_rows = []
            for example in mode_data:
                base_row = {
                    'user_id': example['user_id'],
                    'basket_id': example['basket_id'],
                    'ground_truth_count': example['ground_truth_count'],
                    'epoch': example.get('epoch'),
                    'loss_value': example.get('loss_value')
                }
                
                # Add metrics for each top-K
                for k in self.top_k_show:
                    k_key = f'top_{k}'
                    if k_key in example['ranking_analysis']:
                        k_data = example['ranking_analysis'][k_key]
                        base_row.update({
                            f'recall@{k}': k_data['recall'],
                            f'precision@{k}': k_data['precision'],
                            f'hit_ratio@{k}': k_data['hit_ratio'],
                            f'num_hits@{k}': k_data['num_hits']
                        })
                
                summary_rows.append(base_row)
            
            # Save to CSV
            df = pd.DataFrame(summary_rows)
            # summary_file = output_dir / f"summary_{mode}.csv"
            tag = getattr(self, "run_tag", datetime.now().strftime("%Y%m%d_%H%M%S"))
            summary_file = output_dir / f"summary_{mode}_{tag}.csv"
            df.to_csv(summary_file, index=False)
            
            logger.info(f"Saved summary statistics to: {summary_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save summary statistics: {e}")
    
    def clear_data(self) -> None:
        """Clear stored prediction data."""
        self.prediction_data.clear()
        logger.debug("Cleared prediction analysis data")

    def log_epoch_metrics(self, epoch, train_metrics=None, val_metrics=None, test_metrics=None, train_loss=None):

        if not getattr(self, "save_history", True):
            return
    
        from pathlib import Path
        import csv, os, logging
        logger = logging.getLogger(__name__)
    
        def _flatten(metrics_dict):
            if not isinstance(metrics_dict, dict):
                return {}
            base = metrics_dict.get("metrics", metrics_dict) if isinstance(metrics_dict.get("metrics"), dict) else metrics_dict
            out = {}
            for k, v in base.items():
                if k == "loss":
                    continue
                out[str(k)] = float(v) if isinstance(v, (int, float)) else v
            if "loss" in metrics_dict and metrics_dict["loss"] is not None:
                out["loss"] = float(metrics_dict["loss"]) if isinstance(metrics_dict["loss"], (int, float)) else metrics_dict["loss"]
            return out
    
        train_flat = _flatten(train_metrics)
        val_flat   = _flatten(val_metrics)
        test_flat  = _flatten(test_metrics)
    
        if isinstance(train_loss, (int, float)):
            train_flat["loss"] = float(train_loss)
    
        hist_path = Path(self.history_csv)
        hist_path.parent.mkdir(parents=True, exist_ok=True)
    
        if self._history_header is None or not hist_path.exists():
            def _keys_no_loss(d): return [k for k in d.keys() if k != "loss"]
            def _norm_key(k):
                if   k.startswith("train_"): return k[6:]
                elif k.startswith("val_"):   return k[4:]
                elif k.startswith("test_"):  return k[5:]
                return k
    
            header = ["epoch", "train_loss", "val_loss", "test_loss"]
            header += [f"train_{_norm_key(k)}" for k in _keys_no_loss(train_flat)]
            header += [f"val_{_norm_key(k)}"   for k in _keys_no_loss(val_flat)]
            header += [f"test_{_norm_key(k)}"  for k in _keys_no_loss(test_flat)]

            header = list(dict.fromkeys(header))
            with open(hist_path, "w", newline="") as f:
                csv.writer(f).writerow(header)
            self._history_header = header
            logger.info(f"[history] header created at {hist_path.name}")
    
        row = {h: "" for h in self._history_header}
        row["epoch"] = int(epoch)
    
        def _fill(prefix, flat):
            if not flat: return
            if f"{prefix}_loss" in row:
                row[f"{prefix}_loss"] = flat.get("loss", "")
            for k, v in flat.items():
                if k == "loss": continue
                # 去掉可能已有的前缀
                if   k.startswith("train_"): k_norm = k[6:]
                elif k.startswith("val_"):   k_norm = k[4:]
                elif k.startswith("test_"):  k_norm = k[5:]
                else:                        k_norm = k
                col = f"{prefix}_{k_norm}"
                if col in row:
                    row[col] = v
    
        _fill("train", train_flat)
        _fill("val",   val_flat)
        _fill("test",  test_flat)
    
        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(h, "") for h in self._history_header])


if __name__ == "__main__":
    # Basic testing
    print("✓ PredictionAnalyzer module created successfully")
    print("Note: Full testing requires integration with pipeline") 
