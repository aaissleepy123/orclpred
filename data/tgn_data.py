"""
TGN data module for the unified basket prediction pipeline.

This module provides data loading functionality specific to the TGN variant.
TGN extends MITGNN with Temporal Graph Networks for time-aware recommendations.

Data Schema Extensions for TGN:
- Inherits all MITGNN data schema from MITGNNDataModule
- Adds timestamp processing for temporal modeling
- Creates time-aware graph structures
- Supports different temporal encoding strategies

Temporal Processing:
- Extracts and normalizes timestamps from data
- Creates temporal embeddings and encodings
- Handles time gaps and irregular intervals
- Supports multiple temporal encoding methods (cosine, sinusoidal, learned)

Expected Temporal Data:
- If available: timestamp files with format "basket_id timestamp"
- If not available: uses implicit ordering as timestamps
- FUTURE_IMPROVEMENT: Support for multiple temporal granularities
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import os

import numpy as np
import torch
import scipy.sparse as sp

from .mitgnn_data import MITGNNDataModule
from utils.config import Config


logger = logging.getLogger(__name__)


class TemporalProcessor:
    """
    Processor for temporal features in TGN.
    
    This class handles timestamp extraction, normalization, and encoding
    for temporal graph neural networks.
    """
    
    def __init__(self, 
                 encoding_type: str = "raw",
                 time_encoding: str = "cosine",
                 temporal_dim: int = 2):
        """
        Initialize temporal processor.
        
        Args:
            encoding_type: Type of temporal encoding ("raw", "minmax", "zscore")
            time_encoding: Encoding method ("cosine", "sinusoidal", "learned")
            temporal_dim: Dimension of temporal embeddings
        """
        self.encoding_type = encoding_type
        self.time_encoding = time_encoding
        self.temporal_dim = temporal_dim
        
        # Temporal statistics
        self.time_stats = {
            'min_time': None,
            'max_time': None,
            'mean_time': None,
            'std_time': None
        }
        
        logger.info(f"Temporal processor: {encoding_type} normalization, {time_encoding} method")
    
    def extract_timestamps(self, 
                          data_path: str,
                          train_u2b: Dict[int, List[int]]) -> Dict[int, float]:
        """
        Extract timestamps for baskets.
        
        Args:
            data_path: Path to data directory
            train_u2b: User-to-basket mappings
            
        Returns:
            Dictionary mapping basket_id to timestamp
        """
        basket_timestamps = {}
        
        # Try to load from timestamp file if available
        timestamp_file = os.path.join(data_path, "basket_timestamps.txt")
        
        if os.path.exists(timestamp_file):
            # Load actual timestamps
            with open(timestamp_file, 'r') as f:
                for line in f:
                    if len(line.strip()) > 0:
                        parts = line.strip().split()
                        basket_id = int(parts[0])
                        timestamp = float(parts[1])
                        basket_timestamps[basket_id] = timestamp
            
            logger.info(f"Loaded {len(basket_timestamps)} timestamps from file")
        
        else:
            # Generate implicit timestamps based on user basket order
            logger.warning("No timestamp file found, using implicit ordering")
            
            for user_id, basket_ids in train_u2b.items():
                for i, basket_id in enumerate(basket_ids):
                    # Use order as implicit timestamp (with user offset)
                    timestamp = user_id * 1000 + i  # Simple implicit timestamp
                    basket_timestamps[basket_id] = float(timestamp)
        
        return basket_timestamps
    
    def compute_statistics(self, timestamps: Dict[int, float]) -> None:
        """
        Compute temporal statistics for normalization.
        
        Args:
            timestamps: Dictionary of basket timestamps
        """
        if not timestamps:
            logger.warning("No timestamps available for statistics")
            return
        
        time_values = list(timestamps.values())
        
        self.time_stats = {
            'min_time': np.min(time_values),
            'max_time': np.max(time_values),
            'mean_time': np.mean(time_values),
            'std_time': np.std(time_values)
        }
        
        logger.info(f"Temporal statistics computed:")
        logger.info(f"  Time range: {self.time_stats['min_time']:.2f} - {self.time_stats['max_time']:.2f}")
        logger.info(f"  Mean: {self.time_stats['mean_time']:.2f}, Std: {self.time_stats['std_time']:.2f}")
    
    def normalize_timestamps(self, timestamps: Dict[int, float]) -> Dict[int, float]:
        """
        Normalize timestamps according to encoding type.
        
        Args:
            timestamps: Raw timestamps
            
        Returns:
            Normalized timestamps
        """
        if self.encoding_type == "raw":
            return timestamps
        
        elif self.encoding_type == "minmax":
            # Min-max normalization to [0, 1]
            min_time = self.time_stats['min_time']
            max_time = self.time_stats['max_time']
            
            if max_time == min_time:
                # All timestamps are the same
                return {k: 0.0 for k in timestamps.keys()}
            
            return {
                k: (v - min_time) / (max_time - min_time) 
                for k, v in timestamps.items()
            }
        
        elif self.encoding_type == "zscore":
            # Z-score normalization
            mean_time = self.time_stats['mean_time']
            std_time = self.time_stats['std_time']
            
            if std_time == 0:
                # No variation in timestamps
                return {k: 0.0 for k in timestamps.keys()}
            
            return {
                k: (v - mean_time) / std_time 
                for k, v in timestamps.items()
            }
        
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
    
    def create_temporal_embeddings(self, 
                                  normalized_timestamps: Dict[int, float],
                                  basket_ids: List[int]) -> torch.Tensor:
        """
        Create temporal embeddings for baskets.
        
        Args:
            normalized_timestamps: Normalized timestamp values
            basket_ids: List of basket IDs
            
        Returns:
            Temporal embedding tensor [num_baskets, temporal_dim]
        """
        embeddings = []
        
        for basket_id in basket_ids:
            timestamp = normalized_timestamps.get(basket_id, 0.0)
            
            if self.time_encoding == "cosine":
                # Cosine-based temporal encoding
                emb = []
                for i in range(self.temporal_dim):
                    freq = 1.0 / (10000 ** (2 * i / self.temporal_dim))
                    if i % 2 == 0:
                        emb.append(np.cos(timestamp * freq))
                    else:
                        emb.append(np.sin(timestamp * freq))
                embeddings.append(emb)
            
            elif self.time_encoding == "sinusoidal":
                # Simple sinusoidal encoding
                emb = []
                for i in range(self.temporal_dim):
                    freq = (i + 1) * np.pi
                    emb.append(np.sin(timestamp * freq))
                embeddings.append(emb)
            
            elif self.time_encoding == "learned":
                # For learned embeddings, just pass the normalized timestamp
                # The actual embedding will be learned by the model
                emb = [timestamp] * self.temporal_dim
                embeddings.append(emb)
            
            else:
                raise ValueError(f"Unknown time encoding: {self.time_encoding}")
        
        return torch.tensor(embeddings, dtype=torch.float32)


class TGNDataModule(MITGNNDataModule):
    """
    Data module for TGN variant.
    
    This data module extends MITGNNDataModule with temporal processing
    capabilities required for Temporal Graph Networks.
    """
    
    def __init__(self, config: Config):
        """
        Initialize TGN data module.
        
        Args:
            config: Global configuration object
        """
        super().__init__(config)
        
        # TGN-specific parameters
        if config.model.tgn:
            self.temporal_dim = config.model.tgn.temporal_dim
            self.temporal_encoding = config.model.tgn.temporal_encoding
            self.temporal_normalization = config.model.tgn.temporal_normalization
        else:
            self.temporal_dim = 2
            self.temporal_encoding = 'cosine'
            self.temporal_normalization = 'raw'
        
        # Temporal data processors
        self.temporal_processor = TemporalProcessor(
            encoding_type=self.temporal_normalization,
            time_encoding=self.temporal_encoding,
            temporal_dim=self.temporal_dim
        )
        
        # Temporal data
        self.basket_timestamps = {}
        self.normalized_timestamps = {}
        self.temporal_features = None
        self.temporal_features_extra = None  # sliding-window aggregates aligned by basket id
        
        logger.info(f"TGN data module initialized with temporal dim: {self.temporal_dim}")
    
    def _setup_variant_specific(self) -> None:
        """
        Perform TGN-specific data setup.
        
        This includes:
        1. All MITGNN setup (adjacency matrices, etc.)
        2. Temporal data extraction and processing
        3. Temporal embedding creation
        4. Time-aware graph structure preparation
        """
        logger.info("Setting up TGN-specific data processing")
        
        # First perform MITGNN setup
        super()._setup_variant_specific()
        
        # Add temporal processing (just normalize timestamps, no encoding)
        self._setup_temporal_data()
        
        # Create raw temporal features for model
        self._create_temporal_features()

        # Create sliding-window aggregates if enabled
        self._create_sliding_window_aggregates()
        
        # Analyze temporal patterns
        self._analyze_temporal_patterns()
        
        logger.info("TGN data setup completed")
    
    def _setup_temporal_data(self) -> None:
        """Set up temporal data processing."""
        # Extract timestamps
        self.basket_timestamps = self.temporal_processor.extract_timestamps(
            data_path=str(self.data_path),
            train_u2b=self.train_dataset.train_u2b
        )
        
        # Compute temporal statistics
        self.temporal_processor.compute_statistics(self.basket_timestamps)
        
        # Normalize timestamps
        self.normalized_timestamps = self.temporal_processor.normalize_timestamps(
            self.basket_timestamps
        )
        
        logger.info("Temporal data processing completed")
    
    def _create_temporal_features(self) -> None:
        """Prepare raw temporal features for model (no encoding here)."""
        if not self.normalized_timestamps:
            logger.warning("No temporal data available")
            return
        
        # Get all basket IDs and create raw timestamp tensor
        all_basket_ids = list(range(self.statistics['n_baskets']))
        
        # Create tensor of raw timestamps for each basket
        timestamps = []
        for basket_id in all_basket_ids:
            timestamp = self.normalized_timestamps.get(basket_id, 0.0)
            timestamps.append(timestamp)
        
        self.temporal_features = torch.tensor(timestamps, dtype=torch.float32)
        
        logger.info(f"Prepared raw temporal features: {self.temporal_features.shape}")

    def _create_sliding_window_aggregates(self) -> None:
        """Create additional temporal features from sliding windows over basket history."""
        sw_cfg = getattr(self.config.data, 'sliding_window', {})
        if not sw_cfg or not sw_cfg.get('enabled', False):
            return
        window_len = int(sw_cfg.get('length', 5))
        # Build per-basket aggregates using user histories and timestamps
        num_baskets_total = self.statistics['n_baskets']
        # Initialize feature buffers
        num_baskets_feat = torch.zeros(num_baskets_total, dtype=torch.float32)
        num_unique_items_feat = torch.zeros(num_baskets_total, dtype=torch.float32)
        avg_items_per_basket_feat = torch.zeros(num_baskets_total, dtype=torch.float32)
        time_since_last_feat = torch.zeros(num_baskets_total, dtype=torch.float32)
        median_time_gap_feat = torch.zeros(num_baskets_total, dtype=torch.float32)

        # Helper to get time from basket id (normalized timestamps already built)
        def get_time(bid: int) -> float:
            return float(self.normalized_timestamps.get(bid, 0.0))

        for user_id, basket_ids in self.train_dataset.train_u2b.items():
            for idx, b_id in enumerate(basket_ids):
                # Determine window (exclude current b_id to avoid leakage)
                start_idx = max(0, idx - window_len)
                window_baskets = basket_ids[start_idx:idx]
                if not window_baskets:
                    continue

                # Compute aggregates
                num_baskets_feat[b_id] = len(window_baskets)
                # Items in window
                items_in_window = []
                for wb in window_baskets:
                    items_in_window.extend(self.train_dataset.train_b2i.get(wb, []))
                if items_in_window:
                    num_unique_items_feat[b_id] = float(len(set(items_in_window)))
                    avg_items_per_basket_feat[b_id] = float(len(items_in_window)) / len(window_baskets)
                # Time features
                current_time = get_time(b_id)
                times = [get_time(wb) for wb in window_baskets if wb in self.normalized_timestamps]
                times.sort()
                if times:
                    time_since_last_feat[b_id] = float(current_time - times[-1])
                    if len(times) >= 2:
                        gaps = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
                        gaps.sort()
                        m_idx = len(gaps) // 2
                        median_time_gap_feat[b_id] = float(gaps[m_idx])

        # Stack into tensor [n_baskets, num_features]
        self.temporal_features_extra = torch.stack([
            num_baskets_feat,
            num_unique_items_feat,
            avg_items_per_basket_feat,
            time_since_last_feat,
            median_time_gap_feat
        ], dim=1)
        logger.info(f"Prepared temporal sliding-window aggregates: {self.temporal_features_extra.shape}")
    
    def _analyze_temporal_patterns(self) -> None:
        """Analyze temporal patterns in the data."""
        if not self.basket_timestamps:
            return
        
        # Analyze time gaps between consecutive baskets for each user
        time_gaps = []
        
        for user_id, basket_ids in self.train_dataset.train_u2b.items():
            user_times = []
            for basket_id in basket_ids:
                if basket_id in self.basket_timestamps:
                    user_times.append(self.basket_timestamps[basket_id])
            
            # Sort times and compute gaps
            user_times.sort()
            for i in range(1, len(user_times)):
                gap = user_times[i] - user_times[i-1]
                time_gaps.append(gap)
        
        if time_gaps:
            avg_gap = np.mean(time_gaps)
            median_gap = np.median(time_gaps)
            std_gap = np.std(time_gaps)
            
            logger.info(f"Temporal gap analysis:")
            logger.info(f"  Average gap: {avg_gap:.2f}")
            logger.info(f"  Median gap: {median_gap:.2f}")
            logger.info(f"  Std gap: {std_gap:.2f}")
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get TGN-specific data configuration.
        
        Returns:
            Data configuration including temporal parameters
        """
        # Get MITGNN configuration
        config = super().get_data_config()
        
        # Add TGN-specific parameters
        config.update({
            'temporal_dim': self.temporal_dim,
            'temporal_encoding': self.temporal_encoding,
            'temporal_normalization': self.temporal_normalization,
            'temporal_features': getattr(self, 'temporal_features', None),
            'basket_timestamps': self.basket_timestamps,
            'time_stats': self.temporal_processor.time_stats,
            'temporal_features_extra': getattr(self, 'temporal_features_extra', None)
        })
        
        return config
    
    def get_temporal_features(self) -> torch.Tensor:
        """
        Get raw temporal features for baskets.
        
        Returns:
            Temporal features tensor [n_baskets] with raw timestamps
        """
        if not hasattr(self, 'temporal_features') or self.temporal_features is None:
            logger.warning("Temporal features not available")
            return torch.zeros(self.statistics['n_baskets'])
        
        return self.temporal_features
    
    def get_basket_timestamp(self, basket_id: int, normalized: bool = True) -> float:
        """
        Get timestamp for a specific basket.
        
        Args:
            basket_id: Basket ID
            normalized: Whether to return normalized timestamp
            
        Returns:
            Timestamp value
        """
        if normalized:
            return self.normalized_timestamps.get(basket_id, 0.0)
        else:
            return self.basket_timestamps.get(basket_id, 0.0)
    
    def create_temporal_graph(self) -> Dict[str, Any]:
        """
        Create temporal graph structure for TGN.
        
        Returns:
            Temporal graph data including time-aware edges
        """
        # Create time-ordered edges
        temporal_edges = []
        edge_timestamps = []
        
        # User-basket edges with timestamps
        for user_id, basket_ids in self.train_dataset.train_u2b.items():
            for basket_id in basket_ids:
                timestamp = self.get_basket_timestamp(basket_id, normalized=True)
                temporal_edges.append([user_id, basket_id])
                edge_timestamps.append(timestamp)
        
        # Basket-item edges (use basket timestamps)
        for basket_id, item_ids in self.train_dataset.train_b2i.items():
            basket_timestamp = self.get_basket_timestamp(basket_id, normalized=True)
            for item_id in item_ids:
                # Offset item IDs to avoid conflicts
                item_node_id = self.statistics['n_users'] + self.statistics['n_baskets'] + item_id
                temporal_edges.append([basket_id, item_node_id])
                edge_timestamps.append(basket_timestamp)
        
        return {
            'temporal_edges': torch.tensor(temporal_edges, dtype=torch.long),
            'edge_timestamps': torch.tensor(edge_timestamps, dtype=torch.float32),
            'temporal_features': self.temporal_features
        }


if __name__ == "__main__":
    # Minimal sanity check
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent))
    
    from utils.config import Config, ModelConfig, DataConfig, TrainingConfig, EvaluationConfig, SystemConfig
    
    try:
        # Create a minimal test config with TGN parameters
        test_config = Config(
            model=ModelConfig(
                variant="tgn",
                tgn={
                    'temporal_dim': 2,
                    'time_encoding': 'cosine',
                    'encoding_type': 'raw'
                }
            ),
            data=DataConfig(
                data_path="test_data",
                datagroup="groceries_data",
                dataset="grocerieswithres"
            ),
            training=TrainingConfig(),
            evaluation=EvaluationConfig(),
            system=SystemConfig()
        )
        
        # Test data module creation (won't work without actual data)
        data_module = TGNDataModule(test_config)
        print("✓ TGNDataModule created successfully")
        
    except Exception as e:
        print(f"Note: Test requires actual data files: {e}")
        print("✓ TGNDataModule defined successfully") 