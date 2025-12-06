"""
MITGNN data module for the unified basket prediction pipeline.

This module provides data loading functionality specific to the MITGNN variant.
MITGNN focuses on multi-intent graph neural networks for basket recommendation.

Data Schema Extensions for MITGNN:
- Inherits all base data schema from BaseDataModule
- No additional data files required
- Processes standard user-basket-item interactions

Intent Modeling:
- MITGNN models multiple user intents through graph convolution
- Intent separation is learned during training, not pre-defined in data
- Graph structure captures collaborative filtering patterns
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import scipy.sparse as sp

from .base_data import BaseDataModule, BasketDataset
from utils.config import Config


logger = logging.getLogger(__name__)


class MITGNNDataModule(BaseDataModule):
    """
    Data module for MITGNN variant.
    
    This data module handles the standard collaborative filtering data
    and prepares it for MITGNN's multi-intent graph neural network.
    """
    
    def __init__(self, config: Config):
        """
        Initialize MITGNN data module.
        
        Args:
            config: Global configuration object
        """
        super().__init__(config)
        
        # MITGNN-specific parameters
        self.num_intent = config.model.num_intent
        self.sigma = config.model.sigma
        
        # Adjacency matrix processing
        self.adj_type = config.model.adj_type
        
        # Loss type for data loading (now guaranteed to exist in config schema)
        self.loss_type = config.training.loss_type
        
        # Processed adjacency matrices for GNN
        self.norm_adj_u2b = None
        self.norm_adj_b2i = None
        self.norm_adj_combined = None
        
        logger.info(f"MITGNN data module initialized with {self.num_intent} intents, loss_type: {self.loss_type}")
    
    def _setup_variant_specific(self) -> None:
        """
        Perform MITGNN-specific data setup.
        
        This includes:
        1. Normalizing adjacency matrices for graph convolution
        2. Creating combined user-basket-item adjacency matrix
        3. Preparing for multi-intent learning
        """
        logger.info("Setting up MITGNN-specific data processing")
        
        # Process adjacency matrices
        self._process_adjacency_matrices()
        
        # Create combined adjacency matrix for joint learning
        self._create_combined_adjacency()
        
        # Build user-item mapping for BPR negative sampling
        if self.loss_type == 'bpr':
            self._build_user_item_mapping()
        
        logger.info("MITGNN data setup completed")
    
    def _build_user_item_mapping(self) -> None:
        """Build user-item mapping for efficient negative sampling in BPR."""
        logger.info("Building user-item mapping for BPR negative sampling...")
        
        self.user_items = {}
        
        # For each user, collect all items they have interacted with
        for user_id, basket_ids in self.train_dataset.train_u2b.items():
            user_items = set()
            for basket_id in basket_ids:
                if basket_id in self.train_dataset.train_b2i:
                    user_items.update(self.train_dataset.train_b2i[basket_id])
            self.user_items[user_id] = user_items
        
        logger.info(f"Built user-item mapping for {len(self.user_items)} users")
    
    def sample_negative_item(self, user_id: int) -> int:
        """
        Sample a negative item for given user (for BPR loss).
        
        Args:
            user_id: User ID
            
        Returns:
            Negative item ID
        """
        if not hasattr(self, 'user_items'):
            # Fallback: random sampling
            return np.random.randint(0, self.statistics['n_items'])
        
        user_pos_items = self.user_items.get(user_id, set())
        
        # Simple random sampling with rejection
        max_attempts = 50
        for _ in range(max_attempts):
            neg_item = np.random.randint(0, self.statistics['n_items'])
            if neg_item not in user_pos_items:
                return neg_item
        
        # Fallback: return random item (rare case)
        return np.random.randint(0, self.statistics['n_items'])
    
    def _create_train_dataset(self) -> BasketDataset:
        """Create training dataset."""
        # For BPR, we don't need pre-generated negative samples
        # Negative sampling will be handled in the trainer
        negative_sampling = (self.loss_type == 'bce')
        
        dataset = BasketDataset(
            data_path=self.data_path,
            mode="train",
            negative_sampling=negative_sampling
        )
        return dataset
    
    def _create_test_dataset(self) -> BasketDataset:
        """Create test dataset."""
        return BasketDataset(
            data_path=self.data_path,
            mode="test",
            negative_sampling=False
        )
    
    def _process_adjacency_matrices(self) -> None:
        """Process and normalize adjacency matrices for GNN operations."""
        # Get raw adjacency matrices
        R_u2b = self.train_dataset.R_u2b
        R_b2i = self.train_dataset.R_b2i
        
        # Try to load cached normalized matrices first
        cache_dir = self.data_path.parent / "cache" 
        cache_dir.mkdir(exist_ok=True)
        
        cache_prefix = f"{self.datagroup}_{self.dataset}_{self.adj_type}"
        u2b_cache_file = cache_dir / f"{cache_prefix}_norm_adj_u2b.npz"
        b2i_cache_file = cache_dir / f"{cache_prefix}_norm_adj_b2i.npz"
        
        # Check if cached files exist and are newer than data files
        use_cache = self._should_use_cache([u2b_cache_file, b2i_cache_file])
        
        if use_cache:
            logger.info("Loading cached normalized adjacency matrices...")
            try:
                self.norm_adj_u2b = sp.load_npz(str(u2b_cache_file))
                self.norm_adj_b2i = sp.load_npz(str(b2i_cache_file))
                logger.info("Successfully loaded cached adjacency matrices")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Computing from scratch...")
        
        # Normalize adjacency matrices based on adj_type
        logger.info(f"Computing normalized adjacency matrices with type: {self.adj_type}")
        
        if self.adj_type == "plain":
            # No normalization
            self.norm_adj_u2b = R_u2b
            self.norm_adj_b2i = R_b2i
            
        elif self.adj_type == "norm":
            # Row normalization: D^(-1) * A (matches original normalized_adj_single)
            logger.info("Normalizing user-basket matrix...")
            self.norm_adj_u2b = self._normalize_adj_single(R_u2b)
            logger.info("Normalizing basket-item matrix...")
            self.norm_adj_b2i = self._normalize_adj_single(R_b2i)
            
        elif self.adj_type == "lap":
            # Bipartite normalization: D_row^(-1/2) * A * D_col^(-1/2) (matches original normalized_adj_lap)
            logger.info("Applying bipartite normalization to user-basket matrix...")
            self.norm_adj_u2b = self._normalize_adj_lap(R_u2b)
            logger.info("Applying bipartite normalization to basket-item matrix...")
            self.norm_adj_b2i = self._normalize_adj_lap(R_b2i)
            
        else:
            raise ValueError(f"Unknown adjacency type: {self.adj_type}")
        
        # Cache the normalized matrices for future use
        if self.adj_type != "plain":
            try:
                logger.info("Caching normalized adjacency matrices...")
                sp.save_npz(str(u2b_cache_file), self.norm_adj_u2b)
                sp.save_npz(str(b2i_cache_file), self.norm_adj_b2i)
                logger.info("Successfully cached adjacency matrices")
            except Exception as e:
                logger.warning(f"Failed to cache matrices: {e}")
        
        logger.info(f"Processed adjacency matrices with type: {self.adj_type}")
    
    def _normalize_adj_single(self, adj: sp.csr_matrix) -> sp.csr_matrix:
        """
        Single normalization: D^(-1) * A (row normalization)
        This exactly matches the original TensorFlow normalized_adj_single function.
        
        Args:
            adj: Adjacency matrix to normalize
            
        Returns:
            Row-normalized adjacency matrix
        """
        t1 = time.time()
        
        # Ensure CSR format for efficient row operations
        if not sp.isspmatrix_csr(adj):
            adj = adj.tocsr()
        
        # Compute row sums (degrees) efficiently
        rowsum = np.array(adj.sum(1)).flatten()
        
        # Handle zero degrees to avoid division by zero
        nonzero_mask = rowsum > 0
        d_inv = np.zeros_like(rowsum)
        d_inv[nonzero_mask] = 1.0 / rowsum[nonzero_mask]
        
        # Create diagonal matrix and multiply
        d_mat_inv = sp.diags(d_inv, format='csr')
        norm_adj = d_mat_inv.dot(adj)
        
        elapsed = time.time() - t1
        logger.debug(f'Generated single-normalized adjacency matrix in {elapsed:.2f}s.')
        return norm_adj.tocsr()
    
    def _normalize_adj_lap(self, adj: sp.csr_matrix) -> sp.csr_matrix:
        """
        Laplacian normalization: D_row^(-1/2) * A * D_col^(-1/2) (bipartite normalization)
        This exactly matches the original TensorFlow normalized_adj_lap function.
        
        Args:
            adj: Adjacency matrix to normalize
            
        Returns:
            Bipartite-normalized adjacency matrix
        """
        t1 = time.time()
        
        # Ensure CSR format for efficient operations
        if not sp.isspmatrix_csr(adj):
            adj = adj.tocsr()
        
        # Row normalization: D_row^(-1/2)
        rowsum = np.array(adj.sum(1)).flatten()
        row_nonzero_mask = rowsum > 0
        d_inv_row = np.zeros_like(rowsum)
        d_inv_row[row_nonzero_mask] = 1.0 / np.sqrt(rowsum[row_nonzero_mask])
        
        d_mat_inv_row = sp.diags(d_inv_row, format='csr')
        norm_adj = d_mat_inv_row.dot(adj)
        
        # Column normalization: D_col^(-1/2)
        colsum = np.array(adj.sum(0)).flatten()
        col_nonzero_mask = colsum > 0
        d_inv_col = np.zeros_like(colsum)
        d_inv_col[col_nonzero_mask] = 1.0 / np.sqrt(colsum[col_nonzero_mask])
        
        d_mat_inv_col = sp.diags(d_inv_col, format='csr')
        norm_adj = norm_adj.dot(d_mat_inv_col)
        
        elapsed = time.time() - t1
        logger.debug(f'Generated laplacian norm adjacency matrix in {elapsed:.2f}s.')
        return norm_adj.tocsr()
    
    def _should_use_cache(self, cache_files: List[Path]) -> bool:
        """
        Check if cached adjacency matrices should be used.
        
        Args:
            cache_files: List of cache file paths to check
            
        Returns:
            True if cache should be used, False otherwise
        """
        # Check if all cache files exist
        for cache_file in cache_files:
            if not cache_file.exists():
                return False
        
        # For now, always use cache if files exist
        # In the future, could add timestamp checking against data files
        return True
    
    def _create_combined_adjacency(self) -> None:
        """
        Create combined user-basket-item adjacency matrix for joint learning.
        
        This creates a heterogeneous graph with user, basket, and item nodes:
        [  0    R_u2b   0   ]
        [R_u2b^T  0   R_b2i ]
        [  0   R_b2i^T  0   ]
        """
        t1 = time.time()
        
        n_users = self.statistics['n_users']
        n_baskets = self.statistics['n_baskets']
        n_items = self.statistics['n_items']
        total_nodes = n_users + n_baskets + n_items
        
        logger.info(f"Creating combined adjacency matrix: {total_nodes} total nodes "
                   f"({n_users} users + {n_baskets} baskets + {n_items} items)")
        
        # Get interaction matrices
        R_u2b = self.train_dataset.R_u2b
        R_b2i = self.train_dataset.R_b2i
        
        # Create combined adjacency matrix using LIL format for efficient construction
        combined_adj = sp.lil_matrix((total_nodes, total_nodes), dtype=np.float32)
        
        logger.info("Adding user-basket connections...")
        # User-Basket connections
        combined_adj[0:n_users, n_users:n_users+n_baskets] = R_u2b
        combined_adj[n_users:n_users+n_baskets, 0:n_users] = R_u2b.T
        
        logger.info("Adding basket-item connections...")
        # Basket-Item connections
        combined_adj[n_users:n_users+n_baskets, n_users+n_baskets:] = R_b2i
        combined_adj[n_users+n_baskets:, n_users:n_users+n_baskets] = R_b2i.T
        
        # Convert to CSR for efficient operations
        logger.info("Converting to CSR format...")
        combined_adj = combined_adj.tocsr()
        
        # For combined matrices, original TensorFlow always:
        # 1. Adds self-loops: adj_mat + sp.eye(adj_mat.shape[0])
        # 2. Applies single normalization regardless of adj_type
        if self.adj_type in ["norm", "lap"]:
            logger.info("Adding self-loops and normalizing combined matrix...")
            # Add self-loops to combined square matrix (matches original)
            combined_adj_with_loops = combined_adj + sp.eye(combined_adj.shape[0])
            self.norm_adj_combined = self._normalize_adj_single(combined_adj_with_loops)
        else:
            self.norm_adj_combined = combined_adj
        
        elapsed = time.time() - t1
        logger.info(f"Created combined adjacency matrix: {total_nodes} total nodes in {elapsed:.2f}s")
    
    def get_adjacency_matrices(self) -> Dict[str, sp.csr_matrix]:
        """
        Get processed adjacency matrices for model.
        
        Returns:
            Dictionary containing normalized adjacency matrices
        """
        return {
            'norm_adj_u2b': self.norm_adj_u2b,
            'norm_adj_b2i': self.norm_adj_b2i,
            'norm_adj_combined': self.norm_adj_combined
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get MITGNN-specific data configuration.
        
        Returns:
            Data configuration including adjacency matrices
        """
        # Get base configuration
        config = super().get_data_config()
        
        # Add MITGNN-specific adjacency matrices
        config['adj_matrices'] = self.get_adjacency_matrices()
        config['num_intent'] = self.num_intent
        config['sigma'] = self.sigma
        config['adj_type'] = self.adj_type
        
        return config
    
    def create_negative_samples(self, 
                              user_id: int, 
                              basket_id: int, 
                              n_neg: int = 1) -> list:
        """
        Create negative samples for training.
        
        Args:
            user_id: User ID
            basket_id: Basket ID
            n_neg: Number of negative samples to generate
            
        Returns:
            List of negative item IDs
        """
        # Get positive items for this basket
        positive_items = set(self.train_dataset.train_b2i.get(basket_id, []))
        
        # Get all items user has interacted with
        user_items = set()
        user_baskets = self.train_dataset.train_u2b.get(user_id, [])
        for bid in user_baskets:
            user_items.update(self.train_dataset.train_b2i.get(bid, []))
        
        # Sample negative items
        negative_items = []
        max_attempts = n_neg * 10  # Avoid infinite loop
        attempts = 0
        
        while len(negative_items) < n_neg and attempts < max_attempts:
            neg_item = np.random.randint(0, self.statistics['n_items'])
            
            # Ensure it's truly negative (not in positive items or user history)
            if neg_item not in positive_items and neg_item not in user_items:
                negative_items.append(neg_item)
            
            attempts += 1
        
        # If we couldn't find enough negatives, fill with random items
        while len(negative_items) < n_neg:
            neg_item = np.random.randint(0, self.statistics['n_items'])
            if neg_item not in negative_items:
                negative_items.append(neg_item)
        
        return negative_items[:n_neg]


if __name__ == "__main__":
    # Minimal sanity check
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent))
    
    from utils.config import Config, ModelConfig, DataConfig, TrainingConfig, EvaluationConfig, SystemConfig
    
    try:
        # Create a minimal test config
        test_config = Config(
            model=ModelConfig(variant="mitgnn"),
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
        data_module = MITGNNDataModule(test_config)
        print("✓ MITGNNDataModule created successfully")
        
    except Exception as e:
        print(f"Note: Test requires actual data files: {e}")
        print("✓ MITGNNDataModule defined successfully") 