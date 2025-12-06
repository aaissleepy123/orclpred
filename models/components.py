"""
Shared components for basket prediction models.

This module provides reusable components that can be used across different
model variants (MITGNN, GRNN, TGN) to avoid code duplication and ensure
consistency.
"""

import logging
import math
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


logger = logging.getLogger(__name__)


class GraphConvolution(nn.Module):
    """
    Graph convolution layer for processing adjacency matrices.
    
    This layer performs message passing on graphs by aggregating information
    from neighboring nodes according to the adjacency matrix structure.
    """
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 activation: Optional[str] = 'relu',
                 dropout: float = 0.1):
        """
        Initialize graph convolution layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias
            activation: Activation function ('relu', 'tanh', 'sigmoid', None)
            dropout: Dropout rate
        """
        super(GraphConvolution, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Linear transformation
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            self.activation = None
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of graph convolution.
        
        Args:
            input: Input features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Output features [num_nodes, out_features]
        """
        # Ensure adjacency lives on same device as input (important in multi-GPU)
        if adj.device != input.device:
            adj = adj.to(input.device)
        # Linear transformation: input * weight
        support = torch.mm(input, self.weight)
        
        # Graph convolution: adj * support
        output = torch.spmm(adj, support)
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        # Apply activation
        if self.activation is not None:
            output = self.activation(output)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output


class IntentConvolution(nn.Module):
    """
    Intent-aware graph convolution for MITGNN.
    
    This layer models multiple user intents by learning separate intent-specific
    representations and combining them appropriately.
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_intents: int,
                 intent_dim: Optional[int] = None):
        """
        Initialize intent convolution layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension  
            num_intents: Number of intents to model
            intent_dim: Dimension for intent representations (default: out_features)
        """
        super(IntentConvolution, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_intents = num_intents
        self.intent_dim = intent_dim or out_features
        
        # Intent-specific transformations
        self.intent_weights = nn.ModuleList([
            nn.Linear(in_features, self.intent_dim, bias=False)
            for _ in range(num_intents)
        ])
        
        # Intent combination weights
        self.intent_attention = nn.Linear(self.intent_dim, 1, bias=False)
        
        # Final projection
        self.output_projection = nn.Linear(self.intent_dim, out_features)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        logger.info(f"Intent convolution: {num_intents} intents, {in_features} -> {out_features}")
    
    def forward(self, 
                input: torch.Tensor,
                adj: torch.Tensor,
                user_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of intent convolution.
        
        Args:
            input: Input features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            user_context: Optional user context for intent weighting
            
        Returns:
            Output features [num_nodes, out_features]
        """
        # Align adjacency device with input to avoid device mismatch under DataParallel
        if adj.device != input.device:
            adj = adj.to(input.device)
        batch_size = input.size(0)
        
        # Compute intent-specific representations
        intent_outputs = []
        intent_weights = []
        
        for i, intent_layer in enumerate(self.intent_weights):
            # Transform input for this intent
            intent_features = intent_layer(input)  # [num_nodes, intent_dim]
            
            # Apply graph convolution
            intent_output = torch.spmm(adj, intent_features)  # [num_nodes, intent_dim]
            
            # Compute attention weight for this intent
            intent_weight = self.intent_attention(intent_output)  # [num_nodes, 1]
            
            intent_outputs.append(intent_output)
            intent_weights.append(intent_weight)
        
        # Stack intent outputs and weights
        intent_stack = torch.stack(intent_outputs, dim=2)  # [num_nodes, intent_dim, num_intents]
        weight_stack = torch.stack(intent_weights, dim=2)  # [num_nodes, 1, num_intents]
        
        # Apply softmax to attention weights
        attention_weights = F.softmax(weight_stack, dim=2)  # [num_nodes, 1, num_intents]
        
        # Weighted combination of intents
        combined_output = torch.sum(intent_stack * attention_weights, dim=2)  # [num_nodes, intent_dim]
        
        # Final projection
        output = self.output_projection(combined_output)  # [num_nodes, out_features]
        
        # Apply dropout
        output = self.dropout(output)
        
        return output


class AttentionLayer(nn.Module):
    """
    Multi-head attention layer for combining different representations.
    """
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize attention layer.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(AttentionLayer, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of attention layer.
        
        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim]
            value: Value tensor [batch_size, seq_len, embed_dim]
            mask: Optional attention mask
            
        Returns:
            Attended output [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = query.size()
        
        # Project to query, key, value
        Q = self.query_projection(query)  # [batch_size, seq_len, embed_dim]
        K = self.key_projection(key)      # [batch_size, seq_len, embed_dim]
        V = self.value_projection(value)  # [batch_size, seq_len, embed_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        # Shape: [batch_size, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        # Final projection
        output = self.output_projection(attended_values)
        
        return output


class TemporalEncoder(nn.Module):
    """
    Temporal encoding layer for TGN models.
    
    This layer creates time-aware representations by encoding temporal
    information into the embeddings.
    """
    
    def __init__(self,
                 embed_dim: int,
                 temporal_dim: int = 2,
                 temporal_encoding: str = "cosine",
                 temporal_normalization: str = "raw"):
        """
        Initialize temporal encoder.
        
        Args:
            embed_dim: Embedding dimension
            temporal_dim: Temporal encoding dimension
            temporal_encoding: How to encode time ("cosine", "learned", "sinusoidal")
            temporal_normalization: How to normalize timestamps ("raw", "minmax", "zscore")
        """
        super(TemporalEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim
        self.temporal_encoding = temporal_encoding
        self.temporal_normalization = temporal_normalization
        
        if temporal_encoding == "learned":
            self.temporal_projection = nn.Linear(temporal_dim, embed_dim)
        
        self.fusion_layer = nn.Linear(embed_dim + temporal_dim, embed_dim)
        
    def forward(self, 
                embeddings: torch.Tensor,
                timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of temporal encoder.
        
        Args:
            embeddings: Input embeddings [batch_size, embed_dim]
            timestamps: Raw timestamps [batch_size] or [batch_size, 1]
            
        Returns:
            Temporally-encoded embeddings [batch_size, embed_dim]
        """
        batch_size = embeddings.size(0)
        
        # Ensure timestamps have correct shape [batch_size, 1]
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(1)  # [batch_size, 1]
        
        # Normalize timestamps if needed
        normalized_timestamps = self._normalize_timestamps(timestamps)
        
        # Create temporal encodings
        if self.temporal_encoding == "cosine":
            # Cosine temporal encoding
            temporal_encodings = []
            for i in range(self.temporal_dim):
                freq = 1.0 / (10000 ** (2 * i / self.temporal_dim))
                if i % 2 == 0:
                    encoding = torch.cos(normalized_timestamps * freq)
                else:
                    encoding = torch.sin(normalized_timestamps * freq)
                temporal_encodings.append(encoding)
            
            temporal_features = torch.cat(temporal_encodings, dim=1)  # [batch_size, temporal_dim]
        
        elif self.temporal_encoding == "learned":
            # Learnable temporal encoding
            temporal_features = self.temporal_projection(normalized_timestamps)  # [batch_size, embed_dim]
            # For learned encoding, we return directly without concatenation
            combined = torch.cat([embeddings, temporal_features], dim=1)
            return self.fusion_layer(combined)
        
        elif self.temporal_encoding == "sinusoidal":
            # Simple sinusoidal encoding
            temporal_encodings = []
            for i in range(self.temporal_dim):
                freq = (i + 1) * torch.pi
                encoding = torch.sin(normalized_timestamps * freq)
                temporal_encodings.append(encoding)
            
            temporal_features = torch.cat(temporal_encodings, dim=1)  # [batch_size, temporal_dim]
        
        else:
            # Raw timestamps (no encoding, just repeat to match temporal_dim)
            temporal_features = normalized_timestamps.repeat(1, self.temporal_dim)
        
        # Concatenate embeddings with temporal features
        combined = torch.cat([embeddings, temporal_features], dim=1)
        
        # Fuse temporal and embedding information
        output = self.fusion_layer(combined)
        
        return output
    
    def _normalize_timestamps(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Normalize timestamps based on temporal_normalization setting.
        
        Args:
            timestamps: Raw timestamps [batch_size, 1]
            
        Returns:
            Normalized timestamps [batch_size, 1]
        """
        if self.temporal_normalization == "raw":
            return timestamps
        
        elif self.temporal_normalization == "minmax":
            # Min-max normalization to [0, 1]
            min_val = timestamps.min()
            max_val = timestamps.max()
            if max_val == min_val:
                return torch.zeros_like(timestamps)
            return (timestamps - min_val) / (max_val - min_val)
        
        elif self.temporal_normalization == "zscore":
            # Z-score normalization
            mean_val = timestamps.mean()
            std_val = timestamps.std()
            if std_val == 0:
                return torch.zeros_like(timestamps)
            return (timestamps - mean_val) / std_val
        
        else:
            # Default to raw
            return timestamps


class SequentialEncoder(nn.Module):
    """
    Sequential encoder using RNN for GRNN models.
    
    This layer processes sequential basket data using LSTM or GRU.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 1,
                 rnn_type: str = "LSTM",
                 bidirectional: bool = False,
                 dropout: float = 0.1):
        """
        Initialize sequential encoder.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of RNN layers
            rnn_type: Type of RNN ("LSTM", "GRU")
            bidirectional: Whether to use bidirectional RNN
            dropout: Dropout rate
        """
        super(SequentialEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        
        # RNN layer
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output dimension adjustment for bidirectional
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Output projection
        self.output_projection = nn.Linear(self.output_dim, input_dim)
        
        logger.info(f"Sequential encoder: {rnn_type}, hidden_dim={hidden_dim}, bidirectional={bidirectional}")
    
    def forward(self, 
                sequences: torch.Tensor,
                sequence_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of sequential encoder.
        
        Args:
            sequences: Input sequences [batch_size, max_seq_len, input_dim]
            sequence_lengths: Actual sequence lengths [batch_size]
            
        Returns:
            Tuple of (output_sequences, final_hidden_state)
        """
        batch_size, max_seq_len, _ = sequences.size()
        
        # Pack sequences for efficient processing
        packed_sequences = nn.utils.rnn.pack_padded_sequence(
            sequences, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Forward pass through RNN
        if self.rnn_type == "LSTM":
            packed_output, (hidden, cell) = self.rnn(packed_sequences)
            # Use final hidden state
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                final_hidden = hidden[-1]
        else:  # GRU
            packed_output, hidden = self.rnn(packed_sequences)
            if self.bidirectional:
                final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                final_hidden = hidden[-1]
        
        # Unpack sequences
        output_sequences, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        
        # Project final hidden state back to input dimension
        final_output = self.output_projection(final_hidden)
        
        return output_sequences, final_output


class LearnedTemporalEncoder(nn.Module):
    """Encodes temporal features using a learned linear projection."""
    def __init__(self, temporal_dim, embed_dim):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.embed_dim = embed_dim
        
        # Project timestamp into the temporal feature dimension
        self.feature_projection = nn.Linear(1, temporal_dim)
        
        # Projects temporal features to the main embedding dimension
        self.temporal_projection = nn.Linear(temporal_dim, embed_dim)
        
        # Fusion layer to combine temporal features with item embeddings
        self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, item_emb, normalized_timestamps):
        """
        Forward pass for the learned temporal encoder.

        Args:
            item_emb (torch.Tensor): Item embeddings of shape [batch_size, embed_dim].
            normalized_timestamps (torch.Tensor): Normalized timestamps of shape [batch_size, 1].
        
        Returns:
            torch.Tensor: Enhanced item embeddings of shape [batch_size, embed_dim].
        """
        # Ensure timestamps have a feature dimension
        if normalized_timestamps.dim() == 1:
            normalized_timestamps = normalized_timestamps.unsqueeze(-1)
            
        # [batch_size, 1] -> [batch_size, temporal_dim]
        temporal_features_base = F.relu(self.feature_projection(normalized_timestamps))

        # [batch_size, temporal_dim] -> [batch_size, embed_dim]
        temporal_features = self.temporal_projection(temporal_features_base)
        
        # Combine item and temporal features
        combined_features = torch.cat([item_emb, temporal_features], dim=1)  # [batch_size, embed_dim * 2]
        
        # Fuse and return
        fused_embedding = self.fusion_layer(combined_features)  # [batch_size, embed_dim]
        
        return fused_embedding


def create_sparse_tensor(sparse_matrix: sp.csr_matrix, device: torch.device) -> torch.sparse.FloatTensor:
    """
    Convert scipy sparse matrix to PyTorch sparse tensor.
    
    Args:
        sparse_matrix: Scipy sparse matrix
        device: Target device
        
    Returns:
        PyTorch sparse tensor
    """
    sparse_matrix = sparse_matrix.tocoo()  # Convert to COO format
    
    # Get indices and values
    indices = torch.from_numpy(
        np.vstack((sparse_matrix.row, sparse_matrix.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_matrix.data.astype(np.float32))
    shape = sparse_matrix.shape
    
    # Create sparse tensor
    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape).to(device)
    
    return sparse_tensor


if __name__ == "__main__":
    # Minimal sanity check
    try:
        # Test graph convolution
        gc = GraphConvolution(64, 32)
        print("✓ GraphConvolution created")
        
        # Test intent convolution
        ic = IntentConvolution(64, 32, num_intents=5)
        print("✓ IntentConvolution created")
        
        # Test attention layer
        attn = AttentionLayer(64, num_heads=8)
        print("✓ AttentionLayer created")
        
        # Test temporal encoder
        te = TemporalEncoder(64, temporal_dim=2)
        print("✓ TemporalEncoder created")
        
        # Test sequential encoder
        se = SequentialEncoder(64, 32, rnn_type="LSTM")
        print("✓ SequentialEncoder created")
        
        print("✓ All components created successfully")
        
    except Exception as e:
        print(f"✗ Component test failed: {e}") 