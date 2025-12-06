"""
TGN model implementation for basket prediction.

This module implements the Temporal Graph Network (TGN) which extends MITGNN
with temporal modeling capabilities for time-aware basket recommendation.

Note: Time encoding pipeline 

Raw Timestamps → [encoding_type] → Normalized Timestamps → [time_encoding] → Temporal Embeddings
     ↓                                        ↓                                      ↓
[1609459200,        →     [raw]     →     [1609459200,     →   [cosine]   →    [[0.877, 0.479],
 1609545600,                                1609545600,                           [0.123, 0.991],
 1609632000]                                1609632000]                           [0.654, 0.756]]
"""

import logging
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .mitgnn_model import MITGNNModel
from .components import TemporalEncoder, AttentionLayer, LearnedTemporalEncoder
from utils.config import Config


logger = logging.getLogger(__name__)


class TGNModel(MITGNNModel):
    """
    Temporal Graph Network for basket prediction.
    
    TGN extends MITGNN by adding temporal modeling through time-aware embeddings
    and temporal graph convolution, enabling time-sensitive recommendations.
    """
    
    def __init__(self, config: Config, data_config: Dict[str, Any]):
        """
        Initialize TGN model.
        
        Args:
            config: Global configuration object
            data_config: Data configuration from data module
        """
        # Set TGN-specific parameters BEFORE calling parent constructor
        # This is needed because parent calls _init_variant_components()
        if config.model.tgn:
            self.temporal_dim = config.model.tgn.temporal_dim
            self.temporal_encoding = config.model.tgn.temporal_encoding
            self.temporal_normalization = config.model.tgn.temporal_normalization
            self.temporal_weight = config.model.tgn.temporal_weight
        else:
            # Default values if no TGN config provided
            self.temporal_dim = 2
            self.temporal_encoding = 'cosine'
            self.temporal_normalization = 'raw'
            self.temporal_weight = 1.0
        
        # Get temporal data from data config
        self.temporal_features = data_config.get('temporal_features')
        self.temporal_features_extra = data_config.get('temporal_features_extra')
        self.basket_timestamps = data_config.get('basket_timestamps', {})
        self.time_stats = data_config.get('time_stats', {})
        
        # Initialize MITGNN after setting TGN attributes
        super(TGNModel, self).__init__(config, data_config)
        
        logger.info(f"TGN initialized with temporal_dim={self.temporal_dim}, encoding={self.temporal_encoding}, normalization={self.temporal_normalization}")
    
    def _init_variant_components(self) -> None:
        """Initialize TGN-specific components."""
        # First initialize MITGNN components
        super()._init_variant_components()
        
        # Add temporal components
        self._init_temporal_components()
        
        # Modify prediction layers to incorporate temporal information
        self._init_tgn_prediction_layers()
        
        # Process temporal features
        self._process_temporal_features()
        
        logger.info("TGN components initialized")
    
    def _init_temporal_components(self) -> None:
        """Initialize temporal processing components."""
        final_dim = self.layer_sizes[-1] if self.layer_sizes else self.embed_size
        
        # Temporal encoder for basket embeddings
        if self.temporal_encoding == "learned":
            self.basket_temporal_encoder = LearnedTemporalEncoder(
                temporal_dim=self.temporal_dim,
                embed_dim=final_dim,
            )
        else:
            self.basket_temporal_encoder = TemporalEncoder(
                embed_dim=final_dim,
                temporal_dim=self.temporal_dim,
                temporal_encoding=self.temporal_encoding,
                temporal_normalization=self.temporal_normalization
            )
        
        # Temporal encoder for user-basket interactions
        if self.temporal_encoding == "learned":
            self.interaction_temporal_encoder = LearnedTemporalEncoder(
                temporal_dim=self.temporal_dim,
                embed_dim=final_dim * 2,
            )
        else:
            self.interaction_temporal_encoder = TemporalEncoder(
                embed_dim=final_dim * 2,  # user + basket concatenated
                temporal_dim=self.temporal_dim,
                temporal_encoding=self.temporal_encoding,
                temporal_normalization=self.temporal_normalization
            )
        
        # Temporal attention for time-aware representations  
        # Note: Handles interaction_temporal_encoder output (final_dim * 2 = 128)
        self.temporal_attention = AttentionLayer(
            embed_dim=final_dim * 2,  # 128-dim to match interaction encoder output
            num_heads=4,
            dropout=self.dropout
        )
        
        # Time decay layer: Raw timestamps → simple recency weighting
        # This learns time-based decay like: decay = sigmoid(w * timestamp + b)
        self.time_decay = nn.Sequential(
            nn.Linear(1, 1),  # Raw timestamp [batch_size, 1] → decay weight [batch_size, 1]
            nn.Sigmoid()      # Ensures decay weight ∈ [0, 1]
        )
        
        # Temporal fusion layer
        self.temporal_fusion = nn.Sequential(
            nn.Linear(final_dim * 2, final_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Additional MLP to embed extra temporal aggregates if provided
        self.extra_temporal_embed = nn.Sequential(
            nn.Linear(5, final_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        logger.info("Temporal components initialized")
    
    def _init_tgn_prediction_layers(self) -> None:
        """Initialize TGN-specific prediction layers."""
        final_dim = self.layer_sizes[-1] if self.layer_sizes else self.embed_size
        
        # Enhanced prediction layers that incorporate temporal information (MLP path)
        self.tgn_prediction_layers = nn.Sequential(
            nn.Linear(final_dim * 4, final_dim * 2),  # user + basket + item + temporal
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_dim * 2, final_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_dim, 1)
        )
        
        # Attention layer for combining graph and temporal representations
        self.graph_temporal_attention = nn.Sequential(
            nn.Linear(final_dim * 2, final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        logger.info("TGN prediction layers initialized")
    
    def _process_temporal_features(self) -> None:
        """Process and store raw temporal features."""
        if self.temporal_features is not None:
            # Move temporal features to same device as model
            if isinstance(self.temporal_features, torch.Tensor):
                self.register_buffer('temporal_timestamps', self.temporal_features)
            else:
                # Convert numpy array to tensor
                temporal_tensor = torch.from_numpy(self.temporal_features).float()
                self.register_buffer('temporal_timestamps', temporal_tensor)
            
            logger.info(f"Processed temporal features: {self.temporal_timestamps.shape}")
        else:
            # Create default temporal features (simple incremental timestamps)
            default_timestamps = torch.arange(self.n_baskets, dtype=torch.float32)
            self.register_buffer('temporal_timestamps', default_timestamps)
            logger.warning("No temporal features provided, using default incremental timestamps")
        
        # Process extra temporal features (sliding-window aggregates)
        if self.temporal_features_extra is not None:
            if isinstance(self.temporal_features_extra, torch.Tensor):
                self.register_buffer('temporal_features_extra_buf', self.temporal_features_extra)
            else:
                extra_tensor = torch.from_numpy(self.temporal_features_extra).float()
                self.register_buffer('temporal_features_extra_buf', extra_tensor)
            logger.info(f"Processed extra temporal aggregates: {self.temporal_features_extra_buf.shape}")
        else:
            # Placeholder zeros [n_baskets, 5]
            self.register_buffer('temporal_features_extra_buf', torch.zeros(self.n_baskets, 5))
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of TGN model.
        
        Args:
            batch: Input batch containing user_id, basket_id, item_id, etc.
            
        Returns:
            Dictionary containing predictions and auxiliary outputs
        """
        # First get MITGNN forward pass results
        mitgnn_output = super().forward(batch)
        
        # Extract temporal information
        temporal_features = self._compute_temporal_features(batch)
        
        # Compute extra temporal aggregates embedding
        extra_temporal_emb = self._compute_extra_temporal_features(batch['basket_id'])
        
        # Combine graph and temporal representations
        enhanced_predictions = self._combine_graph_temporal_representations(
            mitgnn_output, temporal_features, batch, extra_temporal_emb
        )
        
        # Update predictions
        mitgnn_output['predictions'] = enhanced_predictions
        mitgnn_output['logits'] = enhanced_predictions
        mitgnn_output['temporal_features'] = temporal_features
        
        return mitgnn_output
    
    def _compute_temporal_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute temporal features for the batch.
        
        Args:
            batch: Input batch
            
        Returns:
            Temporal features tensor
        """
        basket_ids = batch['basket_id']
        
        # Get raw temporal timestamps for baskets
        if hasattr(self, 'temporal_timestamps') and self.temporal_timestamps.size(0) > 0:
            # Ensure basket_ids are within valid range
            valid_basket_ids = torch.clamp(basket_ids, 0, self.temporal_timestamps.size(0) - 1)
            temporal_features = self.temporal_timestamps[valid_basket_ids]  # [batch_size] raw timestamps
        else:
            # Create default temporal features
            batch_size = basket_ids.size(0)
            temporal_features = torch.arange(batch_size, dtype=torch.float32, device=basket_ids.device)
        
        return temporal_features
    
    def _compute_extra_temporal_features(self, basket_ids: torch.Tensor) -> torch.Tensor:
        """Lookup extra temporal aggregates for baskets and embed to model dimension."""
        valid_basket_ids = torch.clamp(basket_ids, 0, self.temporal_features_extra_buf.size(0) - 1)
        raw_extra = self.temporal_features_extra_buf[valid_basket_ids]  # [batch, 5]
        embedded_extra = self.extra_temporal_embed(raw_extra)
        return embedded_extra
    
    def _combine_graph_temporal_representations(self,
                                              mitgnn_output: Dict[str, torch.Tensor],
                                              temporal_features: torch.Tensor,
                                              batch: Dict[str, torch.Tensor],
                                              extra_temporal_emb: torch.Tensor) -> torch.Tensor:
        """
        Combine graph-based and temporal representations.
        
        Args:
            mitgnn_output: Output from MITGNN forward pass
            temporal_features: Temporal features
            batch: Input batch
            extra_temporal_emb: Embedded sliding-window aggregates
            
        Returns:
            Enhanced prediction scores
        """
        # Get embeddings from MITGNN output
        user_emb = mitgnn_output['user_embeddings']  # [batch_size, embed_size]
        basket_emb = mitgnn_output['basket_embeddings']  # [batch_size, embed_size]
        item_emb = mitgnn_output['item_embeddings']  # [batch_size, embed_size]
        
        # CLEAN ARCHITECTURE - Option 1:
        # 1. TemporalEncoder: raw timestamps → cosine encoding → rich temporal features
        temporal_basket_emb = self.basket_temporal_encoder(basket_emb, temporal_features)
        
        # 2. TimeDecay: raw timestamps → simple recency weighting  
        if temporal_features.dim() == 1:
            temporal_features_for_decay = temporal_features.unsqueeze(1)  # [batch_size, 1]
        elif temporal_features.size(1) == 1:
            temporal_features_for_decay = temporal_features  # Already [batch_size, 1]
        else:
            # If temporal_features is multi-dimensional, take the mean as a simple aggregation
            temporal_features_for_decay = temporal_features.mean(dim=1, keepdim=True)  # [batch_size, 1]
            
        time_decay_weights = self.time_decay(temporal_features_for_decay)  # [batch_size, 1]
        
        # 3. Apply simple decay to complex temporal features
        decayed_temporal_basket = temporal_basket_emb * time_decay_weights
        
        # Combine graph and temporal representations with attention
        combined_repr = torch.cat([basket_emb, decayed_temporal_basket], dim=-1)
        attention_weights = self.graph_temporal_attention(combined_repr)  # [batch_size, 2]
        
        graph_weight = attention_weights[:, 0:1]  # [batch_size, 1]
        temporal_weight = attention_weights[:, 1:2]  # [batch_size, 1]
        
        # Weighted combination
        final_basket_emb = (graph_weight * basket_emb + 
                          temporal_weight * decayed_temporal_basket)
        
        # Apply temporal attention to user-item interactions
        temporal_context = self._compute_temporal_context(
            user_emb, final_basket_emb, item_emb, temporal_features
        )
        
        # Compute predictions: dot-product or MLP based on config
        if getattr(self.config.model, 'prediction_method', 'mlp') == 'dot_product':
            # Follow TF: use user-attended basket for scoring; here we combine as in MITGNN
            basket_att_emb = user_emb  # Align with TF's b_at_embeddings = u_c_embeddings
            predictions = torch.sum(basket_att_emb * item_emb, dim=-1)
            return predictions
        else:
            combined_features = torch.cat([
                user_emb,
                final_basket_emb + extra_temporal_emb,
                item_emb,
                temporal_context
            ], dim=-1)
            enhanced_predictions = self.tgn_prediction_layers(combined_features).squeeze(-1)
            return enhanced_predictions
    
    def _compute_temporal_context(self,
                                user_emb: torch.Tensor,
                                basket_emb: torch.Tensor,
                                item_emb: torch.Tensor,
                                temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal context for interactions.
        
        Args:
            user_emb: User embeddings [batch_size, embed_size]
            basket_emb: Basket embeddings [batch_size, embed_size]
            item_emb: Item embeddings [batch_size, embed_size]
            temporal_features: Temporal features [batch_size, temporal_dim]
            
        Returns:
            Temporal context [batch_size, embed_size]
        """
        # Create interaction representations
        user_basket_interaction = torch.cat([user_emb, basket_emb], dim=-1)
        
        # Apply temporal encoding to interactions
        temporal_interaction = self.interaction_temporal_encoder(
            user_basket_interaction, temporal_features
        )
        
        # Use temporal attention to weight interactions
        # Note: temporal_interaction is 128-dim, item_emb is 64-dim
        # Expand item_emb to match temporal_interaction dimension for attention
        item_emb_expanded = torch.cat([item_emb, item_emb], dim=-1)  # [batch_size, 128] 
        
        query = item_emb_expanded.unsqueeze(1)  # [batch_size, 1, 128]
        key = temporal_interaction.unsqueeze(1)  # [batch_size, 1, 128]
        value = temporal_interaction.unsqueeze(1)  # [batch_size, 1, 128]
        
        attended_context = self.temporal_attention(query, key, value)  # [batch_size, 1, 128]
        
        # Project back to original embedding dimension for consistency
        temporal_context_128 = attended_context.squeeze(1)  # [batch_size, 128]
        temporal_context = temporal_context_128[:, :self.embed_size]  # [batch_size, 64] - take first half
        
        return temporal_context
    
    def get_all_item_scores(self, user_ids: torch.Tensor, basket_ids: torch.Tensor) -> torch.Tensor:
        """
        Get prediction scores for all items with temporal modeling.
        
        Args:
            user_ids: User IDs [batch_size]
            basket_ids: Basket IDs [batch_size]
            
        Returns:
            Scores for all items [batch_size, n_items]
        """
        # Create a batch for temporal processing
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
        
        # Compute temporal features
        temporal_features = self._compute_temporal_features(batch)
        
        # Apply temporal encoding to basket embeddings
        temporal_basket_emb = self.basket_temporal_encoder(intent_basket_emb, temporal_features)
        
        # Compute time decay weights (ensure proper shape for raw timestamps)
        if temporal_features.dim() == 1:
            temporal_features_for_decay = temporal_features.unsqueeze(1)  # [batch_size, 1]
        elif temporal_features.size(1) == 1:
            temporal_features_for_decay = temporal_features  # Already [batch_size, 1]
        else:
            # If temporal_features is multi-dimensional, take the mean as a simple aggregation
            temporal_features_for_decay = temporal_features.mean(dim=1, keepdim=True)  # [batch_size, 1]
            
        time_decay_weights = self.time_decay(temporal_features_for_decay)
        decayed_temporal_basket = temporal_basket_emb * time_decay_weights
        
        # Combine representations with attention
        combined_repr = torch.cat([intent_basket_emb, decayed_temporal_basket], dim=-1)
        attention_weights = self.graph_temporal_attention(combined_repr)
        
        graph_weight = attention_weights[:, 0:1]
        temporal_weight = attention_weights[:, 1:2]
        
        final_basket_emb = (graph_weight * intent_basket_emb + 
                          temporal_weight * decayed_temporal_basket)
        
        # Get all item embeddings
        all_item_emb = enhanced_embeddings['item']  # [n_items, embed_size]
        
        # Dot-product path (fast all-item scoring)
        if getattr(self.config.model, 'prediction_method', 'mlp') == 'dot_product':
            # TF uses user embeddings as basket attention
            basket_att_emb = user_emb  # [batch, embed]
            # [batch, embed] @ [embed, n_items] => [batch, n_items]
            return torch.matmul(basket_att_emb, all_item_emb.T)
        
        # Compute scores for all items (MLP path)
        batch_size = user_emb.size(0)
        all_scores = []
        
        for i in range(batch_size):
            user_emb_i = user_emb[i:i+1].expand(self.n_items, -1)
            basket_emb_i = final_basket_emb[i:i+1].expand(self.n_items, -1)
            temporal_features_i = temporal_features[i:i+1].expand(self.n_items, -1)
            
            # Compute temporal context for each item
            temporal_context_i = []
            for j in range(0, self.n_items, 256):  # Process in chunks to avoid memory issues
                end_idx = min(j + 256, self.n_items)
                item_chunk = all_item_emb[j:end_idx]
                
                context_chunk = self._compute_temporal_context(
                    user_emb_i[j:end_idx],
                    basket_emb_i[j:end_idx],
                    item_chunk,
                    temporal_features_i[j:end_idx]
                )
                temporal_context_i.append(context_chunk)
            
            temporal_context_full = torch.cat(temporal_context_i, dim=0)
            
            # Compute scores
            combined_features = torch.cat([
                user_emb_i,
                basket_emb_i,
                all_item_emb,
                temporal_context_full
            ], dim=-1)
            
            scores = self.tgn_prediction_layers(combined_features).squeeze(-1)
            all_scores.append(scores)
        
        return torch.stack(all_scores, dim=0)
    
    def get_temporal_representation(self, basket_ids: torch.Tensor) -> torch.Tensor:
        """
        Get temporal representation for baskets.
        
        Args:
            basket_ids: Basket IDs [batch_size]
            
        Returns:
            Temporal representations [batch_size, temporal_dim]
        """
        batch = {'basket_id': basket_ids}
        return self._compute_temporal_features(batch)
    
    def update_temporal_features(self, new_temporal_timestamps: torch.Tensor) -> None:
        """
        Update temporal features with new timestamp data.
        
        Args:
            new_temporal_timestamps: New temporal timestamps [n_baskets]
        """
        if new_temporal_timestamps.size(0) <= self.n_baskets:
            self.temporal_timestamps[:new_temporal_timestamps.size(0)] = new_temporal_timestamps
            logger.info(f"Updated temporal features for {new_temporal_timestamps.size(0)} baskets")
        else:
            logger.warning("New temporal features size exceeds model capacity")


if __name__ == "__main__":
    # Minimal sanity check
    import sys
    from pathlib import Path
    
    # Add parent directories to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from utils.config import Config, ModelConfig, DataConfig, TrainingConfig, EvaluationConfig, SystemConfig
        
        # Create test config with TGN parameters
        test_config = Config(
            model=ModelConfig(
                variant="tgn",
                num_intent=5,
                tgn={
                    'temporal_dim': 4,
                    'time_encoding': 'cosine',
                    'encoding_type': 'raw'
                }
            ),
            data=DataConfig(data_path="test", datagroup="test", dataset="test"),
            training=TrainingConfig(),
            evaluation=EvaluationConfig(),
            system=SystemConfig()
        )
        
        # Create test data config with temporal embeddings
        test_data_config = {
            'n_users': 100,
            'n_baskets': 200,
            'n_items': 1000,
            'adj_matrices': {},
            'temporal_embeddings': torch.randn(200, 4),  # Random temporal embeddings
            'basket_timestamps': {},
            'time_stats': {},
            'temporal_features_extra': torch.randn(200, 5) # Random extra temporal aggregates
        }
        
        # Create model
        model = TGNModel(test_config, test_data_config)
        
        # Test forward pass
        batch = {
            'user_id': torch.randint(0, 100, (16,)),
            'basket_id': torch.randint(0, 200, (16,)),
            'item_id': torch.randint(0, 1000, (16,))
        }
        
        output = model(batch)
        
        print("✓ TGNModel created and tested successfully")
        print(f"  Output keys: {list(output.keys())}")
        print(f"  Predictions shape: {output['predictions'].shape}")
        
        # Test temporal representation
        temporal_repr = model.get_temporal_representation(batch['basket_id'])
        print(f"  Temporal features shape: {temporal_repr.shape}")
        
    except Exception as e:
        print(f"✗ TGNModel test failed: {e}")
        print("Note: Full testing requires proper imports and data") 