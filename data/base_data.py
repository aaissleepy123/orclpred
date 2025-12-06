"""
Base data module for the unified basket prediction pipeline.

This module provides the common data loading functionality that is shared
across all model variants (MITGNN, GRNN, TGN).

Expected Data Schema:
- train_u2b.txt: User-to-basket mappings for training
  Format: user_id basket_id1 basket_id2 ... basket_idN
  Each line represents one user and their associated baskets
  
- train_b2i.txt: Basket-to-item mappings for training  
  Format: basket_id item_id1 item_id2 ... item_idM
  Each line represents one basket and its contained items
  
- test_b2i.txt: Basket-to-item mappings for testing
  Format: basket_id item_id1 item_id2 ... item_idM
  Each line represents one test basket and its contained items

Data Types and Shapes:
- User IDs: int32, range [0, n_users)
- Basket IDs: int32, range [0, n_baskets) 
- Item IDs: int32, range [0, n_items)
- Adjacency matrices: scipy.sparse matrices (dok_matrix -> csr_matrix)
- Embeddings: float32 tensors
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp

from utils.config import Config
from utils.reproducibility import create_reproducible_dataloader


logger = logging.getLogger(__name__)


class BasketDataset(Dataset):
    """
    PyTorch Dataset for basket prediction.
    
    This dataset handles the loading and preprocessing of user-basket-item
    interaction data for training and evaluation.
    """
    
    def __init__(self, 
                 data_path: Path,
                 mode: str = "train",
                 negative_sampling: bool = True,
                 filter_training_items: bool = True,
                 multiple_test_baskets: bool = True,
                 aggregation_method: str = "user_level",
                 variant: str = "mitgnn",
                 sliding_window_config: Optional[Dict[str, Any]] = None,
                 within_basket_task: bool = True):
        """
        Initialize basket dataset.
        
        Args:
            data_path: Path to the data directory containing txt files
            mode: Dataset mode ("train" or "test")
            negative_sampling: Whether to perform negative sampling
            filter_training_items: Whether to exclude training items from test evaluation
            multiple_test_baskets: Whether to support multiple test baskets per user  
            aggregation_method: How to aggregate evaluation ("user_level" or "user_basket_level")
            variant: Model variant ("mitgnn", "grnn", "tgn") - affects sequence data inclusion
        """
        self.data_path = Path(data_path)
        self.mode = mode
        self.negative_sampling = negative_sampling
        self.filter_training_items = filter_training_items
        self.multiple_test_baskets = multiple_test_baskets
        self.aggregation_method = aggregation_method
        self.variant = variant.lower()
        self.sliding_window_config = sliding_window_config or {}
        self.within_basket_task = within_basket_task
        
        # Data containers
        self.n_users = 0
        self.n_baskets = 0 
        self.n_items = 0
        self.train_u2b = {}  # user -> list of basket IDs (chronological order)
        self.train_b2i = {}  # basket -> list of item IDs
        self.test_set = {}   # basket -> list of item IDs
        self.test_u2b = {}   # user -> test basket ID (for next-basket prediction)
        
        # Sequence data for GRNN (only used when variant == "grnn")
        self.user_sequences = {}     # user -> chronological basket sequence
        self.user_seq_lengths = {}   # user -> actual sequence length
        # Optional sliding window sequences: (user_id, basket_id) -> window list of basket IDs
        self.window_sequences = {}
        
        # Interaction matrices
        self.R_u2b = None  # User-basket interaction matrix
        self.R_b2i = None  # Basket-item interaction matrix
        self.R_u2i = None  # User-item interaction matrix (derived)
        
        # Load and process data
        self._load_data()
        self._build_interaction_matrices()
        self._prepare_dataset()
        
        logger.info(f"Dataset initialized: {self.n_users} users, {self.n_baskets} baskets, {self.n_items} items")
    
    def _load_data(self) -> None:
        """Load data from text files."""
        # Load user-to-basket mappings
        u2b_file = self.data_path / "train_u2b.txt"
        b2i_file = self.data_path / "train_b2i.txt"
        test_file = self.data_path / "test_b2i.txt"
        
        # Validate files exist
        for file_path in [u2b_file, b2i_file, test_file]:
            if not file_path.exists():
                raise FileNotFoundError(f"Required data file not found: {file_path}")
        
        # Load train_u2b.txt
        with open(u2b_file, 'r') as f:
            for line in f:
                if len(line.strip()) > 0:
                    parts = line.strip().split()
                    user_id = int(parts[0])
                    basket_ids = [int(float(bid)) for bid in parts[1:]]
                    self.train_u2b[user_id] = basket_ids
                    
                    # Update counters
                    self.n_users = max(self.n_users, user_id)
                    self.n_baskets = max(self.n_baskets, max(basket_ids))
        
        # Check if we have timestamp format (TGN) by looking at first line of b2i file
        has_timestamps = self._check_for_timestamps(b2i_file)
        
        # Load train_b2i.txt
        with open(b2i_file, 'r') as f:
            for line in f:
                if len(line.strip()) > 0:
                    parts = line.strip().split()
                    basket_id = int(parts[0])
                    
                    if has_timestamps and len(parts) > 1:
                        # For TGN format: basket_id item1 item2 ... timestamp
                        # Last part is timestamp, everything else are items
                        item_ids = [int(iid) for iid in parts[1:-1]]  # Exclude last part (timestamp)
                        # Capture timestamps for TGN (will be exported to basket_timestamps.txt)
                        try:
                            timestamp_token = parts[-1]
                            if not hasattr(self, '_basket_timestamps'):
                                self._basket_timestamps = {}
                            self._basket_timestamps[basket_id] = float(timestamp_token)
                        except Exception:
                            pass
                    else:
                        # Standard format: basket_id item1 item2 ...
                        item_ids = [int(iid) for iid in parts[1:]]
                    
                    self.train_b2i[basket_id] = item_ids
                    
                    # Update counters
                    if len(item_ids) > 0:
                        self.n_items = max(self.n_items, max(item_ids))
        
        # Load test_b2i.txt
        with open(test_file, 'r') as f:
            for line in f:
                if len(line.strip()) > 0:
                    parts = line.strip().split()
                    basket_id = int(parts[0])
                    
                    if has_timestamps and len(parts) > 1:
                        # For TGN format: basket_id item1 item2 ... timestamp
                        item_ids = [int(iid) for iid in parts[1:-1]]  # Exclude last part (timestamp)
                        # Capture timestamps for TGN export
                        try:
                            timestamp_token = parts[-1]
                            if not hasattr(self, '_basket_timestamps'):
                                self._basket_timestamps = {}
                            self._basket_timestamps[basket_id] = float(timestamp_token)
                        except Exception:
                            pass
                    else:
                        # Standard format: basket_id item1 item2 ...
                        item_ids = [int(iid) for iid in parts[1:]]
                    
                    self.test_set[basket_id] = item_ids
                    
                    # Update counters (IMPORTANT: Include test basket IDs for next-basket prediction)
                    self.n_baskets = max(self.n_baskets, basket_id)
                    if len(item_ids) > 0:
                        self.n_items = max(self.n_items, max(item_ids))
        
        # Load test_u2b.txt (user -> test basket mapping for next-basket prediction)
        test_u2b_file = self.data_path / "test_u2b.txt"
        if test_u2b_file.exists():
            with open(test_u2b_file, 'r') as f:
                for line in f:
                    if len(line.strip()) > 0:
                        parts = line.strip().split()
                        user_id = int(parts[0])
                        
                        if self.multiple_test_baskets:
                            # TensorFlow mode: support multiple test baskets per user
                            test_basket_ids = [int(bid) for bid in parts[1:]]
                            if user_id not in self.test_u2b:
                                self.test_u2b[user_id] = []
                            self.test_u2b[user_id].extend(test_basket_ids)
                        else:
                            # PyTorch mode: single test basket per user
                            test_basket_id = int(parts[1])
                            self.test_u2b[user_id] = test_basket_id
            
            if self.multiple_test_baskets:
                total_baskets = sum(len(baskets) for baskets in self.test_u2b.values())
                logger.info(f"Loaded test user-basket mapping: {len(self.test_u2b)} users, {total_baskets} test baskets")
            else:
                logger.info(f"Loaded test user-basket mapping: {len(self.test_u2b)} users")
        else:
            logger.warning("test_u2b.txt not found - assuming within-basket prediction task")
        
        # If inline timestamps were detected, export to a canonical file for TGN
        try:
            if has_timestamps and hasattr(self, '_basket_timestamps'):
                ts_path = self.data_path / "basket_timestamps.txt"
                if not ts_path.exists():
                    with open(ts_path, 'w') as f:
                        for bid, ts in sorted(self._basket_timestamps.items()):
                            f.write(f"{bid} {ts}\n")
                    logger.info(f"Wrote basket timestamps to {ts_path}")
        except Exception as e:
            logger.warning(f"Could not write basket_timestamps.txt: {e}")

        # Attempt to load optional price vector for multimodal datasets
        self._load_price_vector_if_exists()

        # Add 1 to account for 0-indexing
        self.n_users += 1
        self.n_baskets += 1
        self.n_items += 1
        
        logger.info(f"Loaded data: {len(self.train_u2b)} users, {len(self.train_b2i)} train baskets, {len(self.test_set)} test baskets")
    
    def _load_price_vector_if_exists(self) -> None:
        """Load item price mapping if available and store as z-scored vector.

        Expects a tab-separated file named 'item_price_mapping.txt' with two columns:
        item index and price. If present, creates a numpy vector of length n_items
        with z-score normalized prices and stores as self.item_price_vector.
        """
        try:
            price_file = self.data_path / "item_price_mapping.txt"
            if not price_file.exists():
                return
            # Read as tab-separated without headers or with default headers
            df = pd.read_csv(price_file, sep='\t')
            # Heuristics to find columns
            if {'item_no', 'itemPrice'}.issubset(set(df.columns)):
                key_col, val_col = 'item_no', 'itemPrice'
            else:
                # Fallback: use the first two columns
                key_col, val_col = df.columns[0], df.columns[1]
            mapping = dict(zip(df[key_col].astype(int), df[val_col].astype(float)))
            # Build dense vector
            size = int(self.n_items) if isinstance(self.n_items, (int, np.integer)) else int(self.n_items)
            vec = np.zeros((size,), dtype=np.float32)
            for idx, price in mapping.items():
                if 0 <= idx < size:
                    vec[idx] = float(price)
            # z-score normalize (as in TF pipeline)
            mean = vec.mean()
            std = vec.std()
            if std > 0:
                vec = (vec - mean) / std
            else:
                vec = vec * 0.0
            self.item_price_vector = vec
            logger.info(f"Loaded item price vector from {price_file}")
        except Exception as e:
            logger.warning(f"Failed to load item price mapping: {e}")
    
    def _check_for_timestamps(self, b2i_file: Path) -> bool:
        """
        Check if the b2i file has timestamp format by examining the last token.
        
        Args:
            b2i_file: Path to basket-to-item file
            
        Returns:
            True if file appears to have timestamps, False otherwise
        """
        try:
            with open(b2i_file, 'r') as f:
                # Check first few non-empty lines
                for _ in range(min(10, sum(1 for _ in open(b2i_file)))):
                    line = f.readline().strip()
                    if len(line) > 0:
                        parts = line.split()
                        if len(parts) >= 3:  # basket_id + at least 1 item + timestamp
                            last_token = parts[-1]
                            # Check if last token looks like a timestamp (8 digits starting with 201x or 202x)
                            if (len(last_token) == 8 and 
                                last_token.isdigit() and 
                                last_token.startswith(('2012','2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025'))):
                                logger.info("Detected timestamp format in b2i files")
                                return True
                        break
                        
            logger.info("No timestamp format detected, using standard format")
            return False
            
        except Exception as e:
            logger.warning(f"Error checking timestamp format: {e}. Assuming standard format.")
            return False
    
    def _build_interaction_matrices(self) -> None:
        """Build sparse interaction matrices."""
        # Initialize sparse matrices
        self.R_u2b = sp.dok_matrix((self.n_users, self.n_baskets), dtype=np.float32)
        self.R_b2i = sp.dok_matrix((self.n_baskets, self.n_items), dtype=np.float32)
        self.R_u2i = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        
        # Fill user-basket matrix
        for user_id, basket_ids in self.train_u2b.items():
            for basket_id in basket_ids:
                self.R_u2b[user_id, basket_id] = 1.0
        
        # Fill basket-item matrix
        for basket_id, item_ids in self.train_b2i.items():
            for item_id in item_ids:
                self.R_b2i[basket_id, item_id] = 1.0
        
        # Compute user-item matrix (derived from user-basket and basket-item)
        for user_id, basket_ids in self.train_u2b.items():
            user_items = set()
            for basket_id in basket_ids:
                if basket_id in self.train_b2i:
                    user_items.update(self.train_b2i[basket_id])
            
            for item_id in user_items:
                self.R_u2i[user_id, item_id] = 1.0
        
        # Convert to CSR format for efficiency
        self.R_u2b = self.R_u2b.tocsr()
        self.R_b2i = self.R_b2i.tocsr()
        self.R_u2i = self.R_u2i.tocsr()
        
        logger.info("Built interaction matrices")
        
        # Build sequence data for GRNN variant
        if self.variant == "grnn":
            self._build_sequence_data()
    
    def _build_sequence_data(self) -> None:
        """Build sequence data for GRNN from chronological basket sequences."""
        logger.info("Building sequence data for GRNN variant")
        
        for user_id, basket_ids in self.train_u2b.items():
            if len(basket_ids) >= 2:  # Need at least 2 baskets for sequence
                # Store chronological sequence (already in order from train_u2b.txt)
                self.user_sequences[user_id] = basket_ids
                self.user_seq_lengths[user_id] = len(basket_ids)

                # If sliding window enabled, precompute per-basket windows
                if self.sliding_window_config.get('enabled', False):
                    window_len = int(self.sliding_window_config.get('length', 5))
                    include_current = bool(self.sliding_window_config.get('include_current_basket', False))
                    # For within-basket tasks we must not include the current basket in the window
                    if self.within_basket_task:
                        include_current = False
                    for pos, b_id in enumerate(basket_ids):
                        # Determine window end index
                        end_idx = pos + (1 if include_current else 0)
                        start_idx = max(0, end_idx - window_len)
                        window = basket_ids[start_idx:end_idx]
                        self.window_sequences[(user_id, b_id)] = window
        
        logger.info(f"Built sequences for {len(self.user_sequences)} users")
        if self.user_sequences:
            avg_length = sum(self.user_seq_lengths.values()) / len(self.user_seq_lengths)
            max_length = max(self.user_seq_lengths.values())
            min_length = min(self.user_seq_lengths.values())
            logger.info(f"Sequence stats: avg={avg_length:.2f}, min={min_length}, max={max_length}")
    
    def _prepare_dataset(self) -> None:
        """Prepare the dataset for training/testing."""
        self.samples = []
        
        if self.mode == "train":
            # Create positive samples from training data
            for basket_id, item_ids in self.train_b2i.items():
                # Find corresponding user for this basket
                user_id = None
                for uid, bids in self.train_u2b.items():
                    if basket_id in bids:
                        user_id = uid
                        break
                
                if user_id is not None:
                    for item_id in item_ids:
                        sample = {
                            'user_id': user_id,
                            'basket_id': basket_id,
                            'item_id': item_id,
                            'label': 1.0
                        }
                        
                        # Add sequence information for GRNN variant
                        if self.variant == "grnn" and user_id in self.user_sequences:
                            if self.sliding_window_config.get('enabled', False):
                                # Use precomputed sliding window for this (user, basket)
                                window = self.window_sequences.get((user_id, basket_id), [])
                                sample['user_sequence'] = window
                                sample['sequence_length'] = len(window)
                                # Position within user's full sequence is still informative
                                if basket_id in self.user_sequences[user_id]:
                                    sample['basket_position'] = self.user_sequences[user_id].index(basket_id)
                                else:
                                    sample['basket_position'] = -1
                            else:
                                # Fallback: full user sequence
                                sample['user_sequence'] = self.user_sequences[user_id]
                                sample['sequence_length'] = self.user_seq_lengths[user_id]
                                if basket_id in self.user_sequences[user_id]:
                                    sample['basket_position'] = self.user_sequences[user_id].index(basket_id)
                                else:
                                    sample['basket_position'] = -1  # Not in sequence
                        
                        self.samples.append(sample)
                        
                        # Add negative samples if enabled
                        if self.negative_sampling:
                            # Simple random negative sampling
                            neg_item = np.random.randint(0, self.n_items)
                            while neg_item in item_ids:
                                neg_item = np.random.randint(0, self.n_items)
                            
                            neg_sample = {
                                'user_id': user_id,
                                'basket_id': basket_id,
                                'item_id': neg_item,
                                'label': 0.0
                            }
                            
                            # Add same sequence information for negative samples in GRNN
                            if self.variant == "grnn" and user_id in self.user_sequences:
                                if self.sliding_window_config.get('enabled', False):
                                    window = self.window_sequences.get((user_id, basket_id), [])
                                    neg_sample['user_sequence'] = window
                                    neg_sample['sequence_length'] = len(window)
                                    if basket_id in self.user_sequences[user_id]:
                                        neg_sample['basket_position'] = self.user_sequences[user_id].index(basket_id)
                                    else:
                                        neg_sample['basket_position'] = -1
                                else:
                                    neg_sample['user_sequence'] = self.user_sequences[user_id]
                                    neg_sample['sequence_length'] = self.user_seq_lengths[user_id]
                                    if basket_id in self.user_sequences[user_id]:
                                        neg_sample['basket_position'] = self.user_sequences[user_id].index(basket_id)
                                    else:
                                        neg_sample['basket_position'] = -1
                            
                            self.samples.append(neg_sample)
        
        else:  # test mode
            # Create samples from test data - one sample per item for consistency
            for basket_id, item_ids in self.test_set.items():
                # Find corresponding user using test user-basket mapping (for next-basket)
                # or fallback to training mapping (for within-basket)
                user_id = None
                
                if self.test_u2b:
                    # Next-basket prediction: use test_u2b.txt mapping
                    for uid, test_baskets in self.test_u2b.items():
                        if self.multiple_test_baskets:
                            # TensorFlow mode: test_baskets is a list
                            if basket_id in test_baskets:
                                user_id = uid
                                break
                        else:
                            # PyTorch mode: test_baskets is a single basket ID
                            if test_baskets == basket_id:
                                user_id = uid
                                break
                else:
                    # Within-basket prediction: search in training mapping
                    for uid, bids in self.train_u2b.items():
                        if basket_id in bids:
                            user_id = uid
                            break
                
                if user_id is not None:
                    # Create individual samples for each item (consistent with training)
                    for item_id in item_ids:
                        sample = {
                            'user_id': user_id,
                            'basket_id': basket_id,
                            'item_id': item_id,  # Single item like training
                            'label': 1.0,       # Ground truth items have label 1
                            'is_test': True     # Mark as test sample
                        }

                        # Attach GRNN sequence window if enabled
                        if self.variant == "grnn" and self.sliding_window_config.get('enabled', False):
                            # Prefer a window based on this basket if available, else last W from user history
                            window = self.window_sequences.get((user_id, basket_id))
                            if window is None:
                                user_hist = self.train_u2b.get(user_id, [])
                                window_len = int(self.sliding_window_config.get('length', 5))
                                window = user_hist[-window_len:]
                            sample['user_sequence'] = window
                            sample['sequence_length'] = len(window)
                            sample['basket_position'] = -1

                        self.samples.append(sample)
                else:
                    # Track unmappable baskets for debugging
                    if not hasattr(self, '_unmappable_baskets'):
                        self._unmappable_baskets = []
                    self._unmappable_baskets.append(basket_id)
                    logger.warning(f"Could not find user for test basket {basket_id} - basket will be excluded from evaluation")
        
        # Report unmappable baskets summary
        if self.mode == "test" and hasattr(self, '_unmappable_baskets'):
            n_unmappable = len(self._unmappable_baskets)
            n_total_test = len(self.test_set)
            if n_unmappable > 0:
                logger.warning(f"Test dataset: {n_unmappable}/{n_total_test} baskets ({n_unmappable/n_total_test*100:.1f}%) could not be mapped to users and will be excluded from evaluation")
                logger.info(f"Unmappable basket IDs: {sorted(self._unmappable_baskets)[:10]}{'...' if n_unmappable > 10 else ''}")
        
        logger.info(f"Prepared {len(self.samples)} samples for {self.mode}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        return self.samples[idx]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        n_train_u2b = sum(len(baskets) for baskets in self.train_u2b.values())
        n_train_b2i = sum(len(items) for items in self.train_b2i.values())
        n_test_b2i = sum(len(items) for items in self.test_set.values())
        
        stats = {
            'n_users': self.n_users,
            'n_baskets': self.n_baskets,
            'n_items': self.n_items,
            'n_train_u2b': n_train_u2b,
            'n_train_b2i': n_train_b2i,
            'n_test_b2i': n_test_b2i,
            'sparsity_u2b': 1.0 - (n_train_u2b / (self.n_users * self.n_baskets)),
            'sparsity_b2i': 1.0 - (n_train_b2i / (self.n_baskets * self.n_items)),
            'sparsity_u2i': 1.0 - (self.R_u2i.nnz / (self.n_users * self.n_items))
        }
        
        # Add unmappable basket statistics if available
        if hasattr(self, '_unmappable_baskets'):
            stats['n_unmappable_baskets'] = len(self._unmappable_baskets)
            stats['unmappable_rate'] = len(self._unmappable_baskets) / len(self.test_set) if self.test_set else 0.0
        
        return stats


class BaseDataModule(ABC):
    """
    Abstract base class for data modules.
    
    This class defines the interface that all data modules must implement.
    Subclasses should override the variant-specific methods while reusing
    the common functionality.
    """
    
    def __init__(self, config: Config):
        """
        Initialize base data module.
        
        Args:
            config: Global configuration object
        """
        self.config = config
        self.data_path = config.get_data_directory()

        self.datagroup = config.data.datagroup
        self.dataset = config.data.dataset
        self.within_basket = config.data.within_basket
        
        # Dataset instances
        self.train_dataset = None
        self.test_dataset = None
        
        # Data loaders
        self.train_loader = None
        self.test_loader = None
        
        # Dataset statistics
        self.statistics = None
        
        logger.info(f"Initializing data module for variant: {config.model.variant}")
        logger.info(f"Data path: {self.data_path}")
    
    def setup(self) -> None:
        """Set up data loaders and preprocessing."""
        # Validate data path
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
        # Create datasets
        self.train_dataset = self._create_train_dataset()
        self.test_dataset = self._create_test_dataset()
        
        # Get statistics from training dataset
        self.statistics = self.train_dataset.get_statistics()
        
        # Create data loaders
        self.train_loader = self._create_train_loader()
        self.test_loader = self._create_test_loader()
        
        # Variant-specific setup
        self._setup_variant_specific()
        
        logger.info("Data module setup completed")
    
    def _create_train_dataset(self) -> BasketDataset:
        """Create training dataset.""" 
        return BasketDataset(
            data_path=self.data_path,
            mode="train",
            negative_sampling=True,
            filter_training_items=getattr(self.config.evaluation, 'filter_training_items', True),
            multiple_test_baskets=getattr(self.config.evaluation, 'multiple_test_baskets', True),
            aggregation_method=getattr(self.config.evaluation, 'aggregation_method', 'user_level'),
            variant=self.config.model.variant  # Pass variant for sequence data
        )
    
    def _create_test_dataset(self) -> BasketDataset:
        """Create test dataset.""" 
        return BasketDataset(
            data_path=self.data_path,
            mode="test",
            negative_sampling=False,
            filter_training_items=getattr(self.config.evaluation, 'filter_training_items', True),
            multiple_test_baskets=getattr(self.config.evaluation, 'multiple_test_baskets', True),
            aggregation_method=getattr(self.config.evaluation, 'aggregation_method', 'user_level'),
            variant=self.config.model.variant  # Pass variant for sequence data
        )
    
    def _create_train_loader(self) -> DataLoader:
        """Create training data loader."""
        return create_reproducible_dataloader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=True
        )
    
    def _create_test_loader(self) -> DataLoader:
        """Create test data loader."""
        return create_reproducible_dataloader(
            self.test_dataset,
            batch_size=self.config.evaluation.test_batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=False
        )
    
    @abstractmethod
    def _setup_variant_specific(self) -> None:
        """
        Perform variant-specific setup.
        
        This method should be implemented by subclasses to add any
        variant-specific data preprocessing or setup.
        """
        pass
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data configuration for model initialization.
        
        Returns:
            Dictionary containing data configuration parameters
        """
        if self.statistics is None:
            raise RuntimeError("Data module not set up. Call setup() first.")
        
        # Create interaction matrices dictionary
        inter_mat = {
            'R_u2b': self.train_dataset.R_u2b,
            'R_b2i': self.train_dataset.R_b2i,
            'R_u2i': self.train_dataset.R_u2i
        }
        
        return {
            'n_users': self.statistics['n_users'],
            'n_baskets': self.statistics['n_baskets'],
            'n_items': self.statistics['n_items'],
            'inter_mat': inter_mat
        }
    
    def log_statistics(self) -> None:
        """Log dataset statistics."""
        if self.statistics is None:
            logger.warning("No statistics available. Call setup() first.")
            return
        
        logger.info("=== DATASET STATISTICS ===")
        logger.info(f"Users: {self.statistics['n_users']:,}")
        logger.info(f"Baskets: {self.statistics['n_baskets']:,}")
        logger.info(f"Items: {self.statistics['n_items']:,}")
        logger.info(f"Training U2B interactions: {self.statistics['n_train_u2b']:,}")
        logger.info(f"Training B2I interactions: {self.statistics['n_train_b2i']:,}")
        logger.info(f"Test B2I interactions: {self.statistics['n_test_b2i']:,}")
        logger.info(f"U2B sparsity: {self.statistics['sparsity_u2b']:.4f}")
        logger.info(f"B2I sparsity: {self.statistics['sparsity_b2i']:.4f}")
        logger.info(f"U2I sparsity: {self.statistics['sparsity_u2i']:.4f}")
        logger.info("=" * 27)


def grnn_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for GRNN that handles variable-length sequences.
    
    This function properly handles the sequence data added by the enhanced
    BasketDataset when variant="grnn".
    
    Args:
        batch: List of samples from BasketDataset
        
    Returns:
        Dictionary with batched data including properly padded sequences
    """
    # Standard fields that all samples have
    collated = {
        'user_id': torch.tensor([sample['user_id'] for sample in batch], dtype=torch.long),
        'basket_id': torch.tensor([sample['basket_id'] for sample in batch], dtype=torch.long),
        'item_id': torch.tensor([sample['item_id'] for sample in batch], dtype=torch.long),
        'label': torch.tensor([sample['label'] for sample in batch], dtype=torch.float),
    }
    
    # Handle sequence data if present (GRNN variant)
    # Check if any samples have sequence data
    has_sequence_data = any('user_sequence' in sample for sample in batch)
    
    if has_sequence_data:
        # Extract sequence data, handling samples that may not have it
        sequences = []
        seq_lengths = []
        basket_positions = []
        
        for sample in batch:
            if 'user_sequence' in sample:
                user_seq = sample['user_sequence']
                seq_len = sample['sequence_length']
                basket_pos = sample.get('basket_position', -1)
            else:
                # Fallback for samples without sequence data
                user_seq = [0]  # Dummy sequence
                seq_len = 1
                basket_pos = -1
            
            sequences.append(user_seq)
            seq_lengths.append(seq_len)
            basket_positions.append(basket_pos)
        
        # Find maximum sequence length for padding
        max_seq_len = max(seq_lengths)
        
        # Pad sequences to same length
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_seq_len:
                # Pad with -1 (will be handled by embedding layer)
                padded_seq = seq + [-1] * (max_seq_len - len(seq))
            else:
                padded_seq = seq[:max_seq_len]  # Truncate if needed
            padded_sequences.append(padded_seq)
        
        # Add sequence data to collated batch
        collated.update({
            'user_sequence': torch.tensor(padded_sequences, dtype=torch.long),
            'sequence_length': torch.tensor(seq_lengths, dtype=torch.long),
            'basket_position': torch.tensor(basket_positions, dtype=torch.long),
            'max_sequence_length': max_seq_len
        })
    
    return collated


if __name__ == "__main__":
    # Minimal sanity check
    from pathlib import Path
    import tempfile
    
    # Create a simple test implementation
    class TestDataModule(BaseDataModule):
        def _setup_variant_specific(self):
            pass
    
    # This would require actual data files to test fully
    print("✓ BaseDataModule defined successfully")
    print("Note: Full testing requires actual data files") 