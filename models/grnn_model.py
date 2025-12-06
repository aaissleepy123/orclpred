"""
GRNN model implementation for basket prediction.

This module implements the Graph Recurrent Neural Network (GRNN) which extends
MITGNN with sequential modeling capabilities using LSTM/GRU components for
temporal basket prediction.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .mitgnn_model import MITGNNModel
from .components import SequentialEncoder
from utils.config import Config


logger = logging.getLogger(__name__)


class GRNNModel(MITGNNModel):
    """
    Graph Recurrent Neural Network for basket prediction.
    
    GRNN extends MITGNN by adding sequential modeling through RNN components,
    allowing the model to capture temporal patterns in user shopping behavior.
    """
    
    def __init__(self, config: Config, data_config: Dict[str, Any]):
        """
        Initialize GRNN model.
        
        Args:
            config: Global configuration object
            data_config: Data configuration from data module
        """
        # Set GRNN-specific parameters BEFORE calling parent constructor
        # This is needed because parent calls _init_variant_components()
        if config.model.grnn:
            self.rnn_hidden_dim = config.model.grnn.rnn_hidden_dim
            self.rnn_num_layers = config.model.grnn.rnn_num_layers
            self.sequence_max_length = config.model.grnn.sequence_max_length
            self.rnn_type = config.model.grnn.rnn_type
            self.sequential_weight = config.model.grnn.sequential_weight
        else:
            # Default values if no GRNN config provided
            self.rnn_hidden_dim = 64
            self.rnn_num_layers = 1
            self.sequence_max_length = 10
            self.rnn_type = 'LSTM'
            self.sequential_weight = 1.0
        
        # Initialize MITGNN after setting GRNN attributes
        super(GRNNModel, self).__init__(config, data_config)
        
        logger.info(f"GRNN initialized with {self.rnn_type} RNN, hidden_dim={self.rnn_hidden_dim}")
        # Throttle empty-sequence warnings to avoid log spam under sliding windows
        self._empty_seq_warning_count = 0
    
    def _init_variant_components(self) -> None:
        """Initialize GRNN-specific components."""
        # First initialize MITGNN components
        super()._init_variant_components()
        
        # Add sequential components
        self._init_sequential_components()
        
        # Modify prediction layers to incorporate sequential information
        self._init_grnn_prediction_layers()
        
        logger.info("GRNN components initialized")
    
    def _init_sequential_components(self) -> None:
        """Initialize sequential processing components."""
        final_dim = self.layer_sizes[-1] if self.layer_sizes else self.embed_size
        
        # Sequential encoder for basket sequences
        self.basket_sequential_encoder = SequentialEncoder(
            input_dim=final_dim,
            hidden_dim=self.rnn_hidden_dim,
            num_layers=self.rnn_num_layers,
            rnn_type=self.rnn_type,
            bidirectional=False,
            dropout=self.dropout
        )
        
        # Sequential encoder for user-basket interaction sequences
        self.user_sequential_encoder = SequentialEncoder(
            input_dim=final_dim * 2,  # user + basket concatenated
            hidden_dim=self.rnn_hidden_dim,
            num_layers=self.rnn_num_layers,
            rnn_type=self.rnn_type,
            bidirectional=False,
            dropout=self.dropout
        )
        
        # Sequence fusion layer - FIXED: SequentialEncoder output dimension varies by hidden_dim
        # For the actual SequentialEncoder implementation, we need to check the real output dimension
        # Empirically: with hidden_dim=64, output is [batch, 64], not [batch, 128]
        actual_rnn_output_dim = self.rnn_hidden_dim  # SequentialEncoder returns hidden_dim (not 2*hidden_dim)
        self.sequence_fusion = nn.Sequential(
            nn.Linear(final_dim + actual_rnn_output_dim, final_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        logger.info("Sequential components initialized")
    
    def _init_grnn_prediction_layers(self) -> None:
        """Initialize GRNN-specific prediction layers."""
        final_dim = self.layer_sizes[-1] if self.layer_sizes else self.embed_size
        
        # Enhanced prediction layers that incorporate sequential information
        self.grnn_prediction_layers = nn.Sequential(
            nn.Linear(final_dim * 4, final_dim * 2),  # user + basket + item + sequential
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_dim * 2, final_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_dim, 1)
        )
        
        # Attention layer for combining graph and sequential representations
        self.graph_seq_attention = nn.Sequential(
            nn.Linear(final_dim * 2, final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        logger.info("GRNN prediction layers initialized")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of GRNN model.
        
        Unlike MITGNN, GRNN uses RNN-processed embeddings for basket attention.
        This matches the original TensorFlow implementation where GRNN uses
        rnn_outputs_pooled instead of raw user embeddings.
        
        Args:
            batch: Input batch containing user_id, basket_id, item_id, etc.
            
        Returns:
            Dictionary containing predictions and auxiliary outputs
        """
        # Get enhanced embeddings from graph convolution (inherited from MITGNN)
        enhanced_embeddings = self._apply_graph_convolution()
        
        # Extract user, basket, and item embeddings
        user_ids = batch['user_id']
        basket_ids = batch['basket_id'] 
        item_ids = batch['item_id']
        neg_item_ids = batch.get('neg_item_id')  # For BPR loss
        
        user_emb = enhanced_embeddings['user'][user_ids]
        basket_emb = enhanced_embeddings['basket'][basket_ids]
        item_emb = enhanced_embeddings['item'][item_ids]
        
        # Apply intent-aware basket representation (from MITGNN)
        intent_basket_emb = self._compute_intent_basket_representation(basket_emb, user_emb)
        
        # CRITICAL FIX: Extract sequential information and create RNN-processed basket attention
        sequential_features = self._compute_sequential_features(batch)
        
        if sequential_features is not None:
            # Combine graph and sequential information for basket representation
            enhanced_basket_emb = self.sequence_fusion(
                torch.cat([intent_basket_emb, sequential_features], dim=-1)
            )
            
            # Compute attention weights for graph vs sequential representations
            combined_repr = torch.cat([intent_basket_emb, enhanced_basket_emb], dim=-1)
            attention_weights = self.graph_seq_attention(combined_repr)
            
            # Weighted combination (this becomes our basket attention embeddings)
            graph_weight = attention_weights[:, 0:1]
            seq_weight = attention_weights[:, 1:2]
            
            # GRNN basket attention: RNN-enhanced embeddings (not raw user embeddings!)
            basket_att_emb = (graph_weight * intent_basket_emb + seq_weight * enhanced_basket_emb)
        else:
            # Fallback to intent-aware basket embeddings
            basket_att_emb = intent_basket_emb
            # FIXED: Match the actual SequentialEncoder output dimension
            actual_rnn_output_dim = self.rnn_hidden_dim  # SequentialEncoder returns hidden_dim (not 2*hidden_dim)
            sequential_features = torch.zeros(user_emb.size(0), actual_rnn_output_dim, device=user_emb.device)
        
        # Use dot product prediction with RNN-processed basket attention
        # This matches the TensorFlow implementation: batch_ratings = matmul(b_at_embeddings, pos_i_g_embeddings^T)
        predictions = torch.sum(basket_att_emb * item_emb, dim=-1)
        
        # Prepare output
        output = {
            'predictions': predictions,
            'logits': predictions,
            'user_embeddings': user_emb,
            'basket_embeddings': basket_emb,
            'item_embeddings': item_emb,
            'enhanced_embeddings': enhanced_embeddings,
            'sequential_features': sequential_features,
            'basket_att_embeddings': basket_att_emb  # GRNN-specific: RNN-processed basket attention
        }
        
        # For BPR loss, also compute negative scores if negative items are provided
        if neg_item_ids is not None:
            enhanced_neg_item_emb = enhanced_embeddings['item'][neg_item_ids]
            neg_predictions = torch.sum(basket_att_emb * enhanced_neg_item_emb, dim=-1)
            output['pos_scores'] = predictions
            output['neg_scores'] = neg_predictions
            output['neg_item_embeddings'] = enhanced_neg_item_emb
        
        return output
    
    def _compute_sequential_features(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Compute sequential features from batch data.
        
        ENHANCED: Now prioritizes real sequence data from the data pipeline.
        Falls back to synthetic sequences only when real data is unavailable.
        
        Args:
            batch: Input batch
            
        Returns:
            Sequential features tensor or None if no sequential data
        """
        # PRIORITY 1: Check for real sequence data from enhanced data pipeline
        if 'user_sequence' in batch and 'sequence_length' in batch:
            logger.debug("Using real sequence data from data pipeline")
            return self._process_real_sequences(batch)
        
        # PRIORITY 2: Check for pre-processed sequences (from data module)
        if 'sequences' in batch:
            sequences = batch['sequences']  # [batch_size, max_seq_len]
            seq_lengths = batch.get('sequence_lengths', torch.full((sequences.size(0),), sequences.size(1)))
            return self._process_sequences(sequences, seq_lengths)
        
        # FALLBACK: Generate synthetic sequence data (current implementation)
        logger.debug("Falling back to synthetic sequence generation")
        return self._generate_synthetic_sequences(batch)
    
    def _generate_synthetic_sequences(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate meaningful synthetic sequential data for users in the batch.
        
        FIXED: Instead of identical timesteps, create varied temporal patterns
        that provide actual sequential information to the RNN.
        
        Args:
            batch: Input batch
            
        Returns:
            Sequential features tensor
        """
        user_ids = batch['user_id']
        basket_ids = batch['basket_id']
        batch_size = user_ids.size(0)
        
        # Get enhanced embeddings from graph convolution
        enhanced_embeddings = self._apply_graph_convolution()
        
        # Create meaningful synthetic sequences
        user_embeddings = enhanced_embeddings['user'][user_ids]  # [batch_size, embed_size]
        basket_embeddings = enhanced_embeddings['basket'][basket_ids]  # [batch_size, embed_size]
        
        # Create temporal variations instead of identical sequences
        sequences = []
        seq_length = 3
        
        for i in range(batch_size):
            user_emb = user_embeddings[i]  # [embed_size]
            basket_emb = basket_embeddings[i]  # [embed_size]
            
            # Create a sequence with temporal progression
            # t=0: Pure user preference (distant past)
            # t=1: Mix of user and basket (recent past) 
            # t=2: Current basket-user interaction (present)
            timestep_0 = user_emb  # Historical user preference
            timestep_1 = 0.7 * user_emb + 0.3 * basket_emb  # Recent transition
            timestep_2 = 0.4 * user_emb + 0.6 * basket_emb  # Current context
            
            # Add small random variations to break symmetry
            noise_scale = 0.1
            device = user_emb.device
            timestep_0 = timestep_0 + noise_scale * torch.randn_like(timestep_0)
            timestep_1 = timestep_1 + noise_scale * torch.randn_like(timestep_1)  
            timestep_2 = timestep_2 + noise_scale * torch.randn_like(timestep_2)
            
            sequence = torch.stack([timestep_0, timestep_1, timestep_2], dim=0)  # [3, embed_size]
            sequences.append(sequence)
        
        synthetic_sequences = torch.stack(sequences, dim=0)  # [batch_size, 3, embed_size]
        seq_lengths = torch.full((batch_size,), seq_length, device=user_embeddings.device)
        
        return self._process_sequences(synthetic_sequences, seq_lengths)
    
    def _process_real_sequences(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process real sequence data from the data pipeline.
        
        This method handles real chronological basket sequences from the data files,
        creating meaningful temporal embeddings for RNN processing.
        
        Args:
            batch: Batch containing user_sequence and sequence_length data
            
        Returns:
            Sequential features tensor [batch_size, hidden_dim]
        """
        batch_size = batch['user_id'].size(0)
        device = batch['user_id'].device
        
        # Get enhanced embeddings for conversion
        enhanced_embeddings = self._apply_graph_convolution()
        
        # Process each user's sequence
        sequences = []
        seq_lengths = []
        
        for i in range(batch_size):
            user_id = batch['user_id'][i].item()
            user_sequence = batch['user_sequence'][i]  # This should be a list/tensor of basket IDs
            seq_length = batch['sequence_length'][i].item()
            
            # Create embeddings for each basket in the user's chronological sequence
            if isinstance(user_sequence, list):
                # Convert list to tensor if needed
                basket_sequence = torch.tensor(user_sequence[:seq_length], device=device)
            else:
                # Already a tensor, slice to actual length  
                basket_sequence = user_sequence[:seq_length]
            
            # Filter out padding values (-1) and ensure valid basket IDs
            valid_mask = (basket_sequence >= 0) & (basket_sequence < enhanced_embeddings['basket'].size(0))
            valid_baskets = basket_sequence[valid_mask]
            
            if len(valid_baskets) == 0:
                # Fallback to user embedding if no valid baskets (expected for first-basket windows)
                if self._empty_seq_warning_count < 10:
                    logger.warning(f"No valid baskets for user {user_id}, using user embedding")
                elif self._empty_seq_warning_count == 10:
                    logger.warning("Many users have no prior baskets in window; suppressing further warnings.")
                self._empty_seq_warning_count += 1
                user_emb = enhanced_embeddings['user'][user_id]
                temporal_sequence = user_emb.unsqueeze(0)  # [1, embed_size]
                actual_length = 1
            else:
                # Get basket embeddings for this user's historical sequence
                basket_embeddings = enhanced_embeddings['basket'][valid_baskets]  # [valid_len, embed_size]
                
                # Create temporal progression similar to TensorFlow approach
                user_emb = enhanced_embeddings['user'][user_id]  # [embed_size]
                user_emb_expanded = user_emb.unsqueeze(0).expand(basket_embeddings.size(0), -1)  # [valid_len, embed_size]
                
                # Combine user and basket embeddings to represent user state at each timestep
                temporal_sequence = 0.6 * user_emb_expanded + 0.4 * basket_embeddings  # [valid_len, embed_size]
                actual_length = len(valid_baskets)
            
            sequences.append(temporal_sequence)
            seq_lengths.append(actual_length)
        
        # Pad sequences to same length for batch processing
        max_seq_len = max(seq_lengths)
        padded_sequences = []
        
        for seq in sequences:
            if seq.size(0) < max_seq_len:
                # Pad with zeros
                padding = torch.zeros(max_seq_len - seq.size(0), seq.size(1), device=device)
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)
        
        # Stack into batch tensor
        sequence_tensor = torch.stack(padded_sequences, dim=0)  # [batch_size, max_seq_len, embed_size]
        seq_lengths_tensor = torch.tensor(seq_lengths, device=device)
        
        return self._process_sequences(sequence_tensor, seq_lengths_tensor)
    
    def _process_sequences(self, sequences: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        Process sequences through RNN.
        
        Args:
            sequences: Sequence data [batch_size, max_seq_len, feature_dim]
            seq_lengths: Actual sequence lengths [batch_size]
            
        Returns:
            Sequential features [batch_size, hidden_dim]
        """
        # Process sequences through basket sequential encoder
        seq_outputs, final_hidden = self.basket_sequential_encoder(sequences, seq_lengths)
        
        return final_hidden  # [batch_size, hidden_dim]
    
    def get_all_item_scores(self, user_ids: torch.Tensor, basket_ids: torch.Tensor) -> torch.Tensor:
        """
        Get prediction scores for all items with sequential modeling.
        
        FIXED: Uses the same basket attention mechanism as forward pass.
        This matches TensorFlow: batch_ratings = matmul(b_at_embeddings, pos_i_g_embeddings^T)
        
        Args:
            user_ids: User IDs [batch_size]
            basket_ids: Basket IDs [batch_size]
            
        Returns:
            Scores for all items [batch_size, n_items]
        """
        # Create a mock batch for sequential processing
        batch = {
            'user_id': user_ids,
            'basket_id': basket_ids,
            'item_id': torch.zeros_like(user_ids)  # Dummy item IDs
        }
        
        # Get enhanced embeddings
        enhanced_embeddings = self._apply_graph_convolution()
        
        # Get user and basket embeddings
        user_emb = enhanced_embeddings['user'][user_ids]
        basket_emb = enhanced_embeddings['basket'][basket_ids]
        
        # Apply intent-aware basket representation (from MITGNN)
        intent_basket_emb = self._compute_intent_basket_representation(basket_emb, user_emb)
        
        # Compute sequential features (same as forward pass)
        sequential_features = self._compute_sequential_features(batch)
        
        if sequential_features is not None:
            # Enhance basket representation with sequential information
            enhanced_basket_emb = self.sequence_fusion(
                torch.cat([intent_basket_emb, sequential_features], dim=-1)
            )
            
            # Combine representations with attention
            combined_repr = torch.cat([intent_basket_emb, enhanced_basket_emb], dim=-1)
            attention_weights = self.graph_seq_attention(combined_repr)
            
            graph_weight = attention_weights[:, 0:1]
            seq_weight = attention_weights[:, 1:2]
            
            # GRNN basket attention: RNN-enhanced embeddings (same as forward pass)
            basket_att_emb = (graph_weight * intent_basket_emb + seq_weight * enhanced_basket_emb)
        else:
            # Fallback to intent-aware basket embeddings
            basket_att_emb = intent_basket_emb
        
        # Get all item embeddings
        all_item_emb = enhanced_embeddings['item']  # [n_items, embed_size]
        
        # FIXED: Use dot product like TensorFlow implementation
        # batch_ratings = tf.matmul(b_at_embeddings, pos_i_g_embeddings, transpose_b=True)
        # [batch_size, embed_size] @ [embed_size, n_items] = [batch_size, n_items]
        scores = torch.matmul(basket_att_emb, all_item_emb.T)
        
        return scores
    
    def get_sequence_representation(self, user_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get sequential representation for users.
        
        Args:
            user_ids: User IDs [batch_size]
            
        Returns:
            Sequential representations [batch_size, hidden_dim] or None
        """
        batch = {'user_id': user_ids}
        return self._compute_sequential_features(batch)


if __name__ == "__main__":
    # Minimal sanity check
    import sys
    from pathlib import Path
    
    # Add parent directories to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from utils.config import Config, ModelConfig, DataConfig, TrainingConfig, EvaluationConfig, SystemConfig
        
        # Create test config with GRNN parameters
        test_config = Config(
            model=ModelConfig(
                variant="grnn",
                num_intent=5,
                grnn={
                    'rnn_hidden_dim': 32,
                    'rnn_num_layers': 1,
                    'sequence_max_length': 5,
                    'rnn_type': 'LSTM'
                }
            ),
            data=DataConfig(data_path="test", datagroup="test", dataset="test"),
            training=TrainingConfig(),
            evaluation=EvaluationConfig(),
            system=SystemConfig()
        )
        
        # Create test data config
        test_data_config = {
            'n_users': 100,
            'n_baskets': 200,
            'n_items': 1000,
            'adj_matrices': {}
        }
        
        # Create model
        model = GRNNModel(test_config, test_data_config)
        
        # Test forward pass
        batch = {
            'user_id': torch.randint(0, 100, (16,)),
            'basket_id': torch.randint(0, 200, (16,)),
            'item_id': torch.randint(0, 1000, (16,))
        }
        
        output = model(batch)
        
        print("✓ GRNNModel created and tested successfully")
        print(f"  Output keys: {list(output.keys())}")
        print(f"  Predictions shape: {output['predictions'].shape}")
        
        # Test sequence representation
        seq_repr = model.get_sequence_representation(batch['user_id'])
        if seq_repr is not None:
            print(f"  Sequential features shape: {seq_repr.shape}")
        
    except Exception as e:
        print(f"✗ GRNNModel test failed: {e}")
        print("Note: Full testing requires proper imports and data") 