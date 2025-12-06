"""
Enhanced temporal encoder for optimized TGN models.

This module implements advanced temporal encoding strategies specifically optimized
for within-basket and next-basket prediction tasks with different time scales.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


logger = logging.getLogger(__name__)


class MultiScaleTemporalEncoder(nn.Module):
    """
    Multi-scale temporal encoder for next-basket prediction.
    
    Captures temporal patterns at different scales (daily, weekly, monthly, yearly)
    with appropriate encoding strategies for each scale.
    """
    
    def __init__(self,
                 embed_dim: int,
                 temporal_dim: int = 8,
                 temporal_scales: List[int] = [1, 7, 30, 365],
                 scale_weights: List[float] = [0.1, 0.3, 0.4, 0.2]):
        """
        Initialize multi-scale temporal encoder.
        
        Args:
            embed_dim: Embedding dimension
            temporal_dim: Temporal encoding dimension
            temporal_scales: Time scales in days [daily, weekly, monthly, yearly]
            scale_weights: Importance weights for each scale
        """
        super(MultiScaleTemporalEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim
        self.temporal_scales = temporal_scales
        self.scale_weights = nn.Parameter(torch.tensor(scale_weights), requires_grad=False)
        
        # Individual encoders for each scale
        self.scale_encoders = nn.ModuleList([
            nn.Linear(temporal_dim // len(temporal_scales), embed_dim)
            for _ in temporal_scales
        ])
        
        # Fusion layer for combining multi-scale representations
        self.scale_fusion = nn.Sequential(
            nn.Linear(embed_dim * len(temporal_scales), embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Final fusion with original embeddings
        self.final_fusion = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, embeddings: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale temporal encoding.
        
        Args:
            embeddings: Input embeddings [batch_size, embed_dim]
            timestamps: Normalized timestamps [batch_size, 1]
            
        Returns:
            Multi-scale temporally encoded embeddings [batch_size, embed_dim]
        """
        batch_size = embeddings.size(0)
        scale_representations = []
        
        for i, (scale, encoder) in enumerate(zip(self.temporal_scales, self.scale_encoders)):
            # Create temporal encoding for this scale
            scale_temporal_features = self._create_scale_encoding(timestamps, scale, 
                                                                self.temporal_dim // len(self.temporal_scales))
            
            # Project to embedding space
            scale_repr = encoder(scale_temporal_features)
            scale_representations.append(scale_repr)
        
        # Combine multi-scale representations
        combined_scales = torch.cat(scale_representations, dim=-1)
        fused_temporal = self.scale_fusion(combined_scales)
        
        # Final fusion with original embeddings
        combined_final = torch.cat([embeddings, fused_temporal], dim=-1)
        output = self.final_fusion(combined_final)
        
        return output
    
    def _create_scale_encoding(self, timestamps: torch.Tensor, scale: int, dim: int) -> torch.Tensor:
        """Create temporal encoding for a specific time scale."""
        # Normalize timestamps to the specific scale
        scaled_time = timestamps / scale
        
        # Create positional encoding for this scale
        encodings = []
        for i in range(dim):
            freq = 1.0 / (10000 ** (2 * i / dim))
            if i % 2 == 0:
                encodings.append(torch.cos(scaled_time * freq))
            else:
                encodings.append(torch.sin(scaled_time * freq))
        
        return torch.cat(encodings, dim=1)


class AdaptiveTemporalEncoder(nn.Module):
    """
    Adaptive temporal encoder that chooses encoding strategy based on time gap magnitude.
    Optimized for next-basket prediction with varying time gaps.
    """
    
    def __init__(self,
                 embed_dim: int,
                 temporal_dim: int = 8,
                 gap_thresholds: List[int] = [1, 7, 30],
                 gap_encodings: List[str] = ["linear", "cosine", "exponential"]):
        """
        Initialize adaptive temporal encoder.
        
        Args:
            embed_dim: Embedding dimension
            temporal_dim: Temporal encoding dimension
            gap_thresholds: Time gap thresholds in days
            gap_encodings: Encoding types for each gap range
        """
        super(AdaptiveTemporalEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim
        self.gap_thresholds = gap_thresholds
        self.gap_encodings = gap_encodings
        
        # Encoders for different gap types
        self.gap_encoders = nn.ModuleDict({
            encoding: nn.Sequential(
                nn.Linear(temporal_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for encoding in set(gap_encodings)
        })
        
        # Gating mechanism to choose appropriate encoder
        self.gap_gate = nn.Sequential(
            nn.Linear(1, len(gap_encodings)),
            nn.Softmax(dim=-1)
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, embeddings: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive temporal encoding based on time gaps.
        
        Args:
            embeddings: Input embeddings [batch_size, embed_dim]
            timestamps: Time gaps [batch_size, 1]
            
        Returns:
            Adaptively encoded embeddings [batch_size, embed_dim]
        """
        batch_size = embeddings.size(0)
        
        # Generate encoding for each type
        encoding_outputs = {}
        for encoding_type in self.gap_encoders.keys():
            temporal_features = self._create_encoding(timestamps, encoding_type)
            encoded = self.gap_encoders[encoding_type](temporal_features)
            encoding_outputs[encoding_type] = encoded
        
        # Determine gating weights based on time gaps
        gate_weights = self.gap_gate(timestamps)  # [batch_size, num_encodings]
        
        # Weighted combination of encodings
        weighted_encoding = torch.zeros_like(list(encoding_outputs.values())[0])
        for i, encoding_type in enumerate(self.gap_encodings):
            if encoding_type in encoding_outputs:
                weight = gate_weights[:, i:i+1]  # [batch_size, 1]
                weighted_encoding += weight * encoding_outputs[encoding_type]
        
        # Fuse with original embeddings
        combined = torch.cat([embeddings, weighted_encoding], dim=-1)
        output = self.fusion_layer(combined)
        
        return output
    
    def _create_encoding(self, timestamps: torch.Tensor, encoding_type: str) -> torch.Tensor:
        """Create temporal encoding of specified type."""
        if encoding_type == "linear":
            # Simple linear transformation
            features = []
            for i in range(self.temporal_dim):
                features.append(timestamps * (i + 1))
            return torch.cat(features, dim=1)
        
        elif encoding_type == "cosine":
            # Cosine-based positional encoding
            features = []
            for i in range(self.temporal_dim):
                freq = 1.0 / (10000 ** (2 * i / self.temporal_dim))
                if i % 2 == 0:
                    features.append(torch.cos(timestamps * freq))
                else:
                    features.append(torch.sin(timestamps * freq))
            return torch.cat(features, dim=1)
        
        elif encoding_type == "exponential":
            # Exponential decay encoding
            features = []
            for i in range(self.temporal_dim):
                decay_rate = 0.1 * (i + 1)
                features.append(torch.exp(-decay_rate * timestamps))
            return torch.cat(features, dim=1)
        
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")


class ShoppingFlowEncoder(nn.Module):
    """
    Shopping flow encoder for within-basket prediction.
    Models item ordering and shopping cart flow patterns.
    """
    
    def __init__(self,
                 embed_dim: int,
                 temporal_dim: int = 4,
                 flow_attention_heads: int = 2):
        """
        Initialize shopping flow encoder.
        
        Args:
            embed_dim: Embedding dimension
            temporal_dim: Temporal encoding dimension for item ordering
            flow_attention_heads: Number of attention heads for flow modeling
        """
        super(ShoppingFlowEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim
        self.flow_attention_heads = flow_attention_heads
        
        # Item ordering encoder
        self.ordering_encoder = nn.Sequential(
            nn.Linear(temporal_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Shopping flow attention
        self.flow_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=flow_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Flow pattern encoder
        self.flow_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Final fusion
        self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, embeddings: torch.Tensor, item_order: torch.Tensor) -> torch.Tensor:
        """
        Apply shopping flow encoding.
        
        Args:
            embeddings: Input embeddings [batch_size, embed_dim]
            item_order: Item ordering within basket [batch_size, 1]
            
        Returns:
            Shopping flow encoded embeddings [batch_size, embed_dim]
        """
        batch_size = embeddings.size(0)
        
        # Create item ordering features
        ordering_features = self._create_ordering_encoding(item_order)
        ordering_repr = self.ordering_encoder(ordering_features)
        
        # Expand for attention
        embeddings_exp = embeddings.unsqueeze(1)  # [batch_size, 1, embed_dim]
        ordering_exp = ordering_repr.unsqueeze(1)  # [batch_size, 1, embed_dim//2]
        
        # Pad ordering representation to match embedding dimension
        padding_size = self.embed_dim - ordering_repr.size(-1)
        ordering_padded = F.pad(ordering_exp, (0, padding_size))
        
        # Apply flow attention
        attended_flow, _ = self.flow_attention(
            query=embeddings_exp,
            key=ordering_padded,
            value=ordering_padded
        )
        attended_flow = attended_flow.squeeze(1)  # [batch_size, embed_dim]
        
        # Encode flow patterns
        flow_repr = self.flow_encoder(attended_flow)
        
        # Fuse with original embeddings
        combined = torch.cat([embeddings, flow_repr], dim=-1)
        output = self.fusion_layer(combined)
        
        return output
    
    def _create_ordering_encoding(self, item_order: torch.Tensor) -> torch.Tensor:
        """Create encoding for item ordering within basket."""
        # Normalize item order to [0, 1]
        normalized_order = torch.sigmoid(item_order)
        
        # Create positional encoding for item order
        features = []
        for i in range(self.temporal_dim):
            freq = (i + 1) * math.pi
            features.append(torch.sin(normalized_order * freq))
        
        return torch.cat(features, dim=1)


class TimeDecayEncoder(nn.Module):
    """
    Time decay encoder for weighting recent vs distant temporal information.
    """
    
    def __init__(self,
                 embed_dim: int,
                 decay_factor: float = 0.05,
                 min_decay_weight: float = 0.1):
        """
        Initialize time decay encoder.
        
        Args:
            embed_dim: Embedding dimension
            decay_factor: Exponential decay rate
            min_decay_weight: Minimum weight for very old information
        """
        super(TimeDecayEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.decay_factor = decay_factor
        self.min_decay_weight = min_decay_weight
        
        # Learnable decay parameters
        self.decay_projection = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, embeddings: torch.Tensor, time_gaps: torch.Tensor) -> torch.Tensor:
        """
        Apply time decay weighting to embeddings.
        
        Args:
            embeddings: Input embeddings [batch_size, embed_dim]
            time_gaps: Time gaps [batch_size, 1]
            
        Returns:
            Time-weighted embeddings [batch_size, embed_dim]
        """
        # Compute exponential decay weights
        decay_weights = torch.exp(-self.decay_factor * time_gaps)
        decay_weights = torch.clamp(decay_weights, min=self.min_decay_weight)
        
        # Project decay weights to embedding space
        decay_features = self.decay_projection(decay_weights)
        
        # Apply element-wise weighting
        weighted_embeddings = embeddings * decay_features
        
        return weighted_embeddings


# Factory function for creating appropriate temporal encoder
def create_temporal_encoder(task_type: str, config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create the appropriate temporal encoder for the task.
    
    Args:
        task_type: "within_basket" or "next_basket"
        config: Configuration dictionary
        
    Returns:
        Appropriate temporal encoder instance
    """
    embed_dim = config.get('embed_dim', 64)
    temporal_dim = config.get('temporal_dim', 4)
    
    if task_type == "within_basket":
        return ShoppingFlowEncoder(
            embed_dim=embed_dim,
            temporal_dim=temporal_dim,
            flow_attention_heads=config.get('flow_attention_heads', 2)
        )
    
    elif task_type == "next_basket":
        # Choose between multi-scale or adaptive based on config
        if config.get('multi_scale_encoding', False):
            return MultiScaleTemporalEncoder(
                embed_dim=embed_dim,
                temporal_dim=temporal_dim,
                temporal_scales=config.get('temporal_scales', [1, 7, 30, 365]),
                scale_weights=config.get('scale_weights', [0.1, 0.3, 0.4, 0.2])
            )
        elif config.get('adaptive_encoding', False):
            return AdaptiveTemporalEncoder(
                embed_dim=embed_dim,
                temporal_dim=temporal_dim,
                gap_thresholds=config.get('gap_thresholds', [1, 7, 30]),
                gap_encodings=config.get('gap_encodings', ["linear", "cosine", "exponential"])
            )
        else:
            # Fallback to multi-scale as default for next-basket
            return MultiScaleTemporalEncoder(embed_dim=embed_dim, temporal_dim=temporal_dim)
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")


if __name__ == "__main__":
    # Test the temporal encoders
    batch_size = 16
    embed_dim = 64
    
    # Test within-basket encoder
    within_config = {
        'embed_dim': embed_dim,
        'temporal_dim': 4,
        'flow_attention_heads': 2
    }
    
    within_encoder = create_temporal_encoder("within_basket", within_config)
    embeddings = torch.randn(batch_size, embed_dim)
    item_order = torch.randint(0, 10, (batch_size, 1)).float()
    
    within_output = within_encoder(embeddings, item_order)
    print(f"✓ Within-basket encoder: {within_output.shape}")
    
    # Test next-basket encoder
    next_config = {
        'embed_dim': embed_dim,
        'temporal_dim': 8,
        'multi_scale_encoding': True,
        'temporal_scales': [1, 7, 30, 365],
        'scale_weights': [0.1, 0.3, 0.4, 0.2]
    }
    
    next_encoder = create_temporal_encoder("next_basket", next_config)
    timestamps = torch.randint(1, 365, (batch_size, 1)).float()
    
    next_output = next_encoder(embeddings, timestamps)
    print(f"✓ Next-basket encoder: {next_output.shape}")
    
    print("✓ All temporal encoders tested successfully") 