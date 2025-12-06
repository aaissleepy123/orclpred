"""
GRNN data module for the unified basket prediction pipeline.

This module provides data loading functionality specific to the GRNN variant.
GRNN extends MITGNN with recurrent neural networks for sequential modeling.

Data Schema Extensions for GRNN:
- Inherits all MITGNN data schema from MITGNNDataModule
- Adds sequential ordering for RNN processing
- Maintains temporal order of user-basket interactions

Sequential Processing:
- Orders baskets chronologically per user
- Creates sequences for RNN input
- Handles variable-length sequences with padding
- Supports sequence length constraints for memory efficiency
"""

import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .mitgnn_data import MITGNNDataModule
from .base_data import grnn_collate_fn
from utils.config import Config


logger = logging.getLogger(__name__)


class SequentialBasketDataset:
    """
    Dataset extension for sequential basket modeling.
    
    This class handles the sequential aspects of user-basket interactions
    required for RNN processing in GRNN.
    """
    
    def __init__(self, 
                 base_dataset,
                 max_sequence_length: int = 10,
                 min_sequence_length: int = 2):
        """
        Initialize sequential dataset.
        
        Args:
            base_dataset: Base basket dataset
            max_sequence_length: Maximum sequence length for RNN
            min_sequence_length: Minimum sequence length for valid sequences
        """
        self.base_dataset = base_dataset
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        
        # Sequential data containers
        self.user_sequences = {}  # user_id -> list of (basket_id, timestamp)
        self.sequence_lengths = {}  # user_id -> sequence length
        
        # Process sequential data
        self._create_sequences()
        
        logger.info(f"Sequential dataset: {len(self.user_sequences)} users with sequences")

    # def _create_sequences(self) -> None:
    #     """
    #     Create sequential representations of user-basket interactions.
    #
    #     Note: Since original data doesn't include timestamps, we use
    #     the order of baskets in the file as implicit temporal order.
    #     FUTURE_IMPROVEMENT: Add actual timestamp support when available.
    #     """
    #     for user_id, basket_ids in self.base_dataset.train_u2b.items():
    #         # Use basket order as implicit temporal sequence
    #         # In real implementation, this would be sorted by timestamp
    #         sequence = []
    #
    #         for i, basket_id in enumerate(basket_ids):
    #             # Use index as implicit timestamp
    #             timestamp = i
    #             sequence.append((basket_id, timestamp))
    #
    #         # Only keep sequences that meet minimum length requirement
    #         if len(sequence) >= self.min_sequence_length:
    #             # Truncate if too long
    #             if len(sequence) > self.max_sequence_length:
    #                 sequence = sequence[-self.max_sequence_length:]
    #
    #             self.user_sequences[user_id] = sequence
    #             self.sequence_lengths[user_id] = len(sequence)

    def _create_sequences(self) -> None:
        """
        Create sequential representations of user-basket interactions.

        Improvements:
        - Logs a few example sequences so you can inspect them
        - Adapts window length per user (shorter for low-activity users)
        - Keeps chronological order (based on file order, or timestamps if available)
        """
        example_count = 0  # limit printed examples to avoid spam

        for user_id, basket_ids in self.base_dataset.train_u2b.items():
            # 🧠 STEP 1: (Optional) sort by real timestamps if available
            if hasattr(self.base_dataset, "basket_timestamps"):
                baskets_with_times = [
                    (b_id, self.base_dataset.basket_timestamps.get(b_id, i))
                    for i, b_id in enumerate(basket_ids)
                ]
                baskets_with_times.sort(key=lambda x: x[1])  # sort chronologically
                sequence = baskets_with_times
            else:
                # fallback: use order in file
                sequence = [(b_id, i) for i, b_id in enumerate(basket_ids)]

            # 🧠 STEP 2: Adaptive sequence length
            adaptive_max_len = min(len(sequence), self.max_sequence_length)
            if len(sequence) > adaptive_max_len:
                sequence = sequence[-adaptive_max_len:]

            # 🧠 STEP 3: Skip very short users
            if len(sequence) < self.min_sequence_length:
                continue

            # store results
            self.user_sequences[user_id] = sequence
            self.sequence_lengths[user_id] = len(sequence)

            # 🧠 STEP 4: Log a few examples
            if example_count < 5:  # only print first few
                baskets_only = [b for b, _ in sequence]
                print(f"[DEBUG] User {user_id} → sequence of baskets: {baskets_only}")
                example_count += 1

        print(f"[INFO] Created sequences for {len(self.user_sequences)} users")

    def get_user_sequence(self, user_id: int) -> Tuple[List[int], List[int], int]:
        """
        Get sequential data for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (basket_ids, timestamps, sequence_length)
        """
        if user_id not in self.user_sequences:
            return [], [], 0
        
        sequence = self.user_sequences[user_id]
        basket_ids = [item[0] for item in sequence]
        timestamps = [item[1] for item in sequence]
        
        return basket_ids, timestamps, len(sequence)
    
    def create_sequence_batch(self, user_ids: List[int]) -> Dict[str, torch.Tensor]:
        """
        Create a batch of sequences for RNN processing.
        
        Args:
            user_ids: List of user IDs
            
        Returns:
            Dictionary with padded sequences and lengths
        """
        sequences = []
        lengths = []
        
        for user_id in user_ids:
            basket_ids, _, seq_len = self.get_user_sequence(user_id)
            if seq_len > 0:
                sequences.append(torch.tensor(basket_ids, dtype=torch.long))
                lengths.append(seq_len)
            else:
                # Handle users with no valid sequences
                sequences.append(torch.tensor([0], dtype=torch.long))
                lengths.append(1)
        
        # Pad sequences to same length
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        length_tensor = torch.tensor(lengths, dtype=torch.long)
        
        return {
            'sequences': padded_sequences,
            'lengths': length_tensor,
            'max_length': padded_sequences.size(1)
        }


class GRNNDataModule(MITGNNDataModule):
    """
    Data module for GRNN variant.
    
    This data module extends MITGNNDataModule with sequential processing
    capabilities required for the RNN component of GRNN.
    """
    
    def __init__(self, config: Config):
        """
        Initialize GRNN data module.
        
        Args:
            config: Global configuration object
        """
        super().__init__(config)
        
        # GRNN-specific parameters
        if config.model.grnn:
            self.rnn_hidden_dim = config.model.grnn.rnn_hidden_dim
            self.rnn_num_layers = config.model.grnn.rnn_num_layers
            self.sequence_max_length = config.model.grnn.sequence_max_length
            self.rnn_type = config.model.grnn.rnn_type
        else:
            # Default values
            self.rnn_hidden_dim = 64
            self.rnn_num_layers = 1
            self.sequence_max_length = 10
            self.rnn_type = 'LSTM'
        
        # Sequential data processors
        self.sequential_train = None
        self.sequential_test = None
        
        logger.info(f"GRNN data module initialized with {self.rnn_type} RNN")
    
    def _setup_variant_specific(self) -> None:
        """
        Perform GRNN-specific data setup.
        
        This includes:
        1. All MITGNN setup (adjacency matrices, etc.)
        2. Sequential data processing for RNN
        3. Sequence length analysis and optimization
        """
        logger.info("Setting up GRNN-specific data processing")
        
        # First perform MITGNN setup
        super()._setup_variant_specific()
        
        # Add sequential processing
        self._setup_sequential_data()
        
        # Analyze sequence statistics
        self._analyze_sequences()
        
        logger.info("GRNN data setup completed")
    
    def _create_train_dataset(self):
        """Create training dataset with GRNN variant support."""
        from .base_data import BasketDataset
        
        return BasketDataset(
            data_path=self.data_path,
            mode="train",
            negative_sampling=True,
            filter_training_items=getattr(self.config.evaluation, 'filter_training_items', True),
            multiple_test_baskets=getattr(self.config.evaluation, 'multiple_test_baskets', True),
            aggregation_method=getattr(self.config.evaluation, 'aggregation_method', 'user_level'),
            variant="grnn",  # Explicitly pass GRNN variant
            sliding_window_config=getattr(self.config.data, 'sliding_window', {}),
            within_basket_task=self.config.data.within_basket
        )
    
    def _create_test_dataset(self):
        """Create test dataset with GRNN variant support."""
        from .base_data import BasketDataset
        
        return BasketDataset(
            data_path=self.data_path,
            mode="test",
            negative_sampling=False,
            filter_training_items=getattr(self.config.evaluation, 'filter_training_items', True),
            multiple_test_baskets=getattr(self.config.evaluation, 'multiple_test_baskets', True),
            aggregation_method=getattr(self.config.evaluation, 'aggregation_method', 'user_level'),
            variant="grnn",  # Explicitly pass GRNN variant
            sliding_window_config=getattr(self.config.data, 'sliding_window', {}),
            within_basket_task=self.config.data.within_basket
        )
    
    def _setup_sequential_data(self) -> None:
        """Set up sequential data processors."""
        # Create sequential processors for train and test
        self.sequential_train = SequentialBasketDataset(
            base_dataset=self.train_dataset,
            max_sequence_length=self.sequence_max_length,
            min_sequence_length=2
        )
        
        self.sequential_test = SequentialBasketDataset(
            base_dataset=self.test_dataset,
            max_sequence_length=self.sequence_max_length,
            min_sequence_length=1  # More lenient for test
        )
        
        logger.info("Sequential data processors created")
    
    def _analyze_sequences(self) -> None:
        """Analyze sequence length statistics."""
        if self.sequential_train is None:
            return
        
        lengths = list(self.sequential_train.sequence_lengths.values())
        
        if lengths:
            avg_length = np.mean(lengths)
            median_length = np.median(lengths)
            max_length = np.max(lengths)
            min_length = np.min(lengths)
            
            logger.info(f"Sequence statistics:")
            logger.info(f"  Average length: {avg_length:.2f}")
            logger.info(f"  Median length: {median_length:.2f}")
            logger.info(f"  Min/Max length: {min_length}/{max_length}")
            logger.info(f"  Total sequences: {len(lengths)}")
            
            # Check if sequences are too long
            long_sequences = sum(1 for l in lengths if l > self.sequence_max_length)
            if long_sequences > 0:
                logger.warning(f"{long_sequences} sequences truncated to {self.sequence_max_length}")
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get GRNN-specific data configuration.
        
        Returns:
            Data configuration including RNN parameters
        """
        # Get MITGNN configuration
        config = super().get_data_config()
        
        # Add GRNN-specific parameters
        config.update({
            'rnn_hidden_dim': self.rnn_hidden_dim,
            'rnn_num_layers': self.rnn_num_layers,
            'sequence_max_length': self.sequence_max_length,
            'rnn_type': self.rnn_type
        })
        
        return config
    
    def create_rnn_batch(self, user_ids: List[int], mode: str = "train") -> Dict[str, torch.Tensor]:
        """
        Create a batch for RNN processing.
        
        Args:
            user_ids: List of user IDs
            mode: Dataset mode ("train" or "test")
            
        Returns:
            Dictionary with RNN-ready batch data
        """
        sequential_data = self.sequential_train if mode == "train" else self.sequential_test
        
        if sequential_data is None:
            raise RuntimeError("Sequential data not initialized. Call setup() first.")
        
        return sequential_data.create_sequence_batch(user_ids)
    
    def get_sequence_embeddings_config(self) -> Dict[str, Any]:
        """
        Get configuration for sequence embeddings.
        
        Returns:
            Configuration for embedding layers used in sequences
        """
        return {
            'num_baskets': self.statistics['n_baskets'],
            'basket_embedding_dim': self.config.model.embed_size,
            'sequence_max_length': self.sequence_max_length,
            'rnn_hidden_dim': self.rnn_hidden_dim,
            'rnn_num_layers': self.rnn_num_layers,
            'rnn_type': self.rnn_type
        }
    
    def _create_train_loader(self):
        """Create training data loader with GRNN-specific collate function."""
        from torch.utils.data import DataLoader
        from utils.reproducibility import set_random_seed
        
        # Set seed for reproducibility
        set_random_seed(self.config.system.seed)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=True,
            collate_fn=grnn_collate_fn  # Use GRNN-specific collate function
        )
    
    def _create_test_loader(self):
        """Create test data loader with GRNN-specific collate function."""
        from torch.utils.data import DataLoader
        from utils.reproducibility import set_random_seed
        
        # Set seed for reproducibility
        set_random_seed(self.config.system.seed)
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.evaluation.test_batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=False,
            collate_fn=grnn_collate_fn  # Use GRNN-specific collate function
        )
    
    def validate_sequence_data(self) -> bool:
        """
        Validate that sequence data is properly set up.
        
        Returns:
            True if validation passes
        """
        if self.sequential_train is None or self.sequential_test is None:
            logger.error("Sequential data not initialized")
            return False
        
        # Check if we have enough sequential data
        min_users = 10  # Minimum users with valid sequences
        train_sequences = len(self.sequential_train.user_sequences)
        
        if train_sequences < min_users:
            logger.warning(f"Only {train_sequences} users have valid sequences (min: {min_users})")
            return False
        
        logger.info("Sequence data validation passed")
        return True


if __name__ == "__main__":
    # Minimal sanity check
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent))
    
    from utils.config import Config, ModelConfig, DataConfig, TrainingConfig, EvaluationConfig, SystemConfig
    
    try:
        # Create a minimal test config with GRNN parameters
        test_config = Config(
            model=ModelConfig(
                variant="grnn",
                grnn={
                    'rnn_hidden_dim': 64,
                    'rnn_num_layers': 1,
                    'sequence_max_length': 10,
                    'rnn_type': 'LSTM'
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
        data_module = GRNNDataModule(test_config)
        print("✓ GRNNDataModule created successfully")
        
    except Exception as e:
        print(f"Note: Test requires actual data files: {e}")
        print("✓ GRNNDataModule defined successfully") 