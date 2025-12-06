"""
MITGNN model implementation for basket prediction.

This module implements the Multi-Intent Translation Graph Neural Network (MITGNN)
for basket recommendation. MITGNN models multiple user intents through graph 
convolution and intent-aware representations.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base_model import BaseModel
from .components import GraphConvolution, IntentConvolution, create_sparse_tensor
from utils.config import Config


logger = logging.getLogger(__name__)


class MITGNNModel(BaseModel):
    """
    Multi-Intent Translation Graph Neural Network for basket prediction.
    
    MITGNN models user-basket-item interactions through multi-intent graph
    convolution, allowing the model to capture diverse user preferences and
    shopping patterns.
    """
    
    def __init__(self, config: Config, data_config: Dict[str, Any]):
        """
        Initialize MITGNN model.
        
        Args:
            config: Global configuration object
            data_config: Data configuration from data module
        """
        # Set MITGNN-specific parameters BEFORE calling super()
        # This is needed because base class calls _init_variant_components()
        # which requires these attributes to be set
        self.num_intent = config.model.num_intent
        self.sigma = config.model.sigma
        self.adj_type = config.model.adj_type
        self.alg_type = config.model.alg_type
        
        # New configuration options for matching original implementation
        self.prediction_method = getattr(config.model, 'prediction_method', 'mlp')  # 'mlp' or 'dot_product'
        self.use_explicit_regularization = getattr(config.model, 'use_explicit_regularization', False)
        self.regularization_weight = getattr(config.model, 'regularization_weight', 1e-3)
        
        # Get adjacency matrices from data config
        self.adj_matrices = data_config.get('adj_matrices', {})
        
        # Initialize base model AFTER setting variant-specific attributes
        super(MITGNNModel, self).__init__(config, data_config)
        
        logger.info(f"MITGNN initialized with {self.num_intent} intents, sigma={self.sigma}")
        logger.info(f"Prediction method: {self.prediction_method}")
        logger.info(f"Explicit regularization: {self.use_explicit_regularization}")
    
    def _init_variant_components(self) -> None:
        """Initialize MITGNN-specific components."""
        # Graph convolution layers for different algorithms
        if self.alg_type == "intent_conv":
            self._init_intent_conv_layers()
        elif self.alg_type == "gcn":
            self._init_gcn_layers()
        elif self.alg_type == "gcmc":
            self._init_gcmc_layers()
        else:
            # Default to intent convolution
            self._init_intent_conv_layers()
        
        # Output prediction layers
        self._init_prediction_layers()
        
        # Convert adjacency matrices to sparse tensors
        self._process_adjacency_matrices()
        
        logger.info(f"MITGNN components initialized with algorithm: {self.alg_type}")
    
    def _init_intent_conv_layers(self) -> None:
        """Initialize intent convolution layers."""
        self.intent_conv_layers = nn.ModuleList()
        
        # Create multiple layers of intent convolution
        layer_dims = [self.embed_size] + self.layer_sizes
        
        for i in range(len(layer_dims) - 1):
            intent_layer = IntentConvolution(
                in_features=layer_dims[i],
                out_features=layer_dims[i + 1],
                num_intents=self.num_intent
            )
            self.intent_conv_layers.append(intent_layer)
        
        logger.info(f"Created {len(self.intent_conv_layers)} intent convolution layers")
    
    def _init_gcn_layers(self) -> None:
        """Initialize standard GCN layers."""
        self.gcn_layers = nn.ModuleList()
        
        layer_dims = [self.embed_size] + self.layer_sizes
        
        for i in range(len(layer_dims) - 1):
            gcn_layer = GraphConvolution(
                in_features=layer_dims[i],
                out_features=layer_dims[i + 1],
                activation='relu',
                dropout=self.dropout
            )
            self.gcn_layers.append(gcn_layer)
        
        logger.info(f"Created {len(self.gcn_layers)} GCN layers")
    
    def _init_gcmc_layers(self) -> None:
        """Initialize Graph Convolutional Matrix Completion layers."""
        # Similar to GCN but with matrix completion specific modifications
        self._init_gcn_layers()  # For now, use same as GCN
        logger.info("GCMC layers initialized (using GCN implementation)")
    
    def _init_prediction_layers(self) -> None:
        """Initialize prediction layers."""
        final_dim = self.layer_sizes[-1] if self.layer_sizes else self.embed_size
        
        # Prediction MLP
        self.prediction_layers = nn.Sequential(
            nn.Linear(final_dim * 3, final_dim * 2),  # user + basket + item
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_dim * 2, final_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_dim, 1)
        )
        
        # Intent attention for basket representation
        self.basket_intent_attention = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Linear(final_dim // 2, self.num_intent),
            nn.Softmax(dim=-1)
        )
        
        logger.info("Prediction layers initialized")
    
    def _process_adjacency_matrices(self) -> None:
        """Convert adjacency matrices to PyTorch sparse tensors."""
        self.sparse_adj = {}
        
        if self.adj_matrices:
            for adj_name, adj_matrix in self.adj_matrices.items():
                if adj_matrix is not None:
                    # Convert scipy sparse matrix to PyTorch sparse tensor
                    sparse_adj = self._create_sparse_tensor(adj_matrix)
                    # Register as a buffer so DataParallel replicates per-device
                    buffer_name = f"_buffer_adj__{adj_name}"
                    self.register_buffer(buffer_name, sparse_adj)
                    # Keep mapping for existing call sites
                    self.sparse_adj[adj_name] = getattr(self, buffer_name)
                    logger.info(f"Converted {adj_name} to sparse tensor: {sparse_adj.shape}")
    
    def _create_sparse_tensor(self, scipy_sparse_matrix):
        """
        Convert scipy sparse matrix to PyTorch sparse tensor.
        
        Args:
            scipy_sparse_matrix: Input scipy sparse matrix
            
        Returns:
            PyTorch sparse tensor
        """
        # Convert to COO format for PyTorch
        coo_matrix = scipy_sparse_matrix.tocoo()
        
        # Create indices tensor
        indices = torch.stack([
            torch.from_numpy(coo_matrix.row.astype('int64')),
            torch.from_numpy(coo_matrix.col.astype('int64'))
        ])
        
        # Create values tensor
        values = torch.from_numpy(coo_matrix.data.astype('float32'))
        
        # Create sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, coo_matrix.shape, dtype=torch.float32
        )
        
        return sparse_tensor.coalesce()
    
    def to(self, device):
        """Ensure parameters and registered buffers move devices; avoid reassigning buffers."""
        super().to(device)
        return self
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of MITGNN model.
        
        Args:
            batch: Input batch containing user_id, basket_id, item_id, etc.
            
        Returns:
            Dictionary containing predictions and auxiliary outputs
        """
        # Extract batch data
        user_ids = batch.get('user_id')
        basket_ids = batch.get('basket_id') 
        item_ids = batch.get('item_id')
        neg_item_ids = batch.get('neg_item_id')  # For BPR loss
        
        if user_ids is None or basket_ids is None or item_ids is None:
            available_keys = list(batch.keys())
            raise ValueError(f"Batch must contain user_id, basket_id, and item_id. "
                           f"Available keys: {available_keys}")
        
        # Get initial embeddings, TODO: these seems to be redundant?
        user_emb = self.get_user_embeddings(user_ids)  # [batch_size, embed_size]
        basket_emb = self.get_basket_embeddings(basket_ids)  # [batch_size, embed_size]
        item_emb = self.get_item_embeddings(item_ids)  # [batch_size, embed_size]
        
        # Apply graph convolution to get enhanced embeddings
        enhanced_embeddings = self._apply_graph_convolution()
        
        # Extract enhanced embeddings for current batch
        enhanced_user_emb = enhanced_embeddings['user'][user_ids]
        enhanced_basket_emb = enhanced_embeddings['basket'][basket_ids]
        enhanced_item_emb = enhanced_embeddings['item'][item_ids]
        
        # Apply intent-aware basket representation
        intent_basket_emb = self._compute_intent_basket_representation(
            enhanced_basket_emb, enhanced_user_emb
        )
        
        # Compute prediction scores
        predictions = self._compute_predictions(
            enhanced_user_emb, intent_basket_emb, enhanced_item_emb
        )
        
        # Prepare output
        output = {
            'predictions': predictions,
            'logits': predictions,  # Alias for compatibility
            'user_embeddings': enhanced_user_emb,
            'basket_embeddings': intent_basket_emb,
            'item_embeddings': enhanced_item_emb
        }
        
        # For BPR loss, also compute negative scores if negative items are provided
        if neg_item_ids is not None:
            enhanced_neg_item_emb = enhanced_embeddings['item'][neg_item_ids]
            neg_predictions = self._compute_predictions(
                enhanced_user_emb, intent_basket_emb, enhanced_neg_item_emb
            )
            output['pos_scores'] = predictions
            output['neg_scores'] = neg_predictions
            output['neg_item_embeddings'] = enhanced_neg_item_emb
        
        # Compute auxiliary losses
        aux_losses = self._compute_auxiliary_losses(enhanced_embeddings, output)
        output['aux_losses'] = aux_losses
        
        return output
    
    def _apply_graph_convolution(self) -> Dict[str, torch.Tensor]:
        """
        Apply graph convolution to all embeddings.
        
        Returns:
            Dictionary with enhanced embeddings for users, baskets, items
        """
        # Get all embeddings
        all_user_emb = self.user_embedding.weight  # [n_users, embed_size]
        all_basket_emb = self.basket_embedding.weight  # [n_baskets, embed_size]
        all_item_emb = self.item_embedding.weight  # [n_items, embed_size]
        
        if self.alg_type == "intent_conv":
            return self._apply_intent_convolution(all_user_emb, all_basket_emb, all_item_emb)
        elif self.alg_type in ["gcn", "gcmc"]:
            return self._apply_standard_gcn(all_user_emb, all_basket_emb, all_item_emb)
        else:
            # Fallback to intent convolution
            return self._apply_intent_convolution(all_user_emb, all_basket_emb, all_item_emb)
    
    def _apply_intent_convolution(self, 
                                 user_emb: torch.Tensor,
                                 basket_emb: torch.Tensor,
                                 item_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply intent convolution layers."""
        # Create combined embedding matrix
        # Order: [users, baskets, items]
        combined_emb = torch.cat([user_emb, basket_emb, item_emb], dim=0)
        
        # Get combined adjacency matrix
        adj_matrix = self.sparse_adj.get('norm_adj_combined')
        if adj_matrix is None:
            logger.warning("Combined adjacency matrix not found, using identity")
            adj_matrix = torch.eye(combined_emb.size(0), device=combined_emb.device).to_sparse()
        
        # Apply intent convolution layers
        current_emb = combined_emb
        layer_outputs = [current_emb]
        
        for layer in self.intent_conv_layers:
            current_emb = layer(current_emb, adj_matrix)
            current_emb = self.apply_dropout(current_emb, self.training)
            layer_outputs.append(current_emb)
        
        # Combine outputs from all layers (similar to LightGCN)
        final_emb = torch.mean(torch.stack(layer_outputs), dim=0)
        
        # Split back into user, basket, item embeddings
        enhanced_user_emb = final_emb[:self.n_users]
        enhanced_basket_emb = final_emb[self.n_users:self.n_users + self.n_baskets]
        enhanced_item_emb = final_emb[self.n_users + self.n_baskets:]
        
        return {
            'user': enhanced_user_emb,
            'basket': enhanced_basket_emb,
            'item': enhanced_item_emb
        }
    
    def _apply_standard_gcn(self,
                          user_emb: torch.Tensor,
                          basket_emb: torch.Tensor, 
                          item_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply standard GCN layers."""
        # Similar to intent convolution but without intent-specific processing
        combined_emb = torch.cat([user_emb, basket_emb, item_emb], dim=0)
        
        adj_matrix = self.sparse_adj.get('norm_adj_combined')
        if adj_matrix is None:
            adj_matrix = torch.eye(combined_emb.size(0), device=self.device).to_sparse()
        
        current_emb = combined_emb
        layer_outputs = [current_emb]
        
        for layer in self.gcn_layers:
            current_emb = layer(current_emb, adj_matrix)
            layer_outputs.append(current_emb)
        
        # Combine outputs
        final_emb = torch.mean(torch.stack(layer_outputs), dim=0)
        
        # Split embeddings
        enhanced_user_emb = final_emb[:self.n_users]
        enhanced_basket_emb = final_emb[self.n_users:self.n_users + self.n_baskets]
        enhanced_item_emb = final_emb[self.n_users + self.n_baskets:]
        
        return {
            'user': enhanced_user_emb,
            'basket': enhanced_basket_emb,
            'item': enhanced_item_emb
        }
    
    def _compute_intent_basket_representation(self,
                                            basket_emb: torch.Tensor,
                                            user_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute intent-aware basket representation.
        
        Args:
            basket_emb: Basket embeddings [batch_size, embed_size]
            user_emb: User embeddings [batch_size, embed_size]
            
        Returns:
            Intent-aware basket embeddings [batch_size, embed_size]
        """
        # Compute user-basket interaction strength
        user_basket_interaction = torch.sum(user_emb * basket_emb, dim=-1, keepdim=True)
        
        # Apply sigma weighting (from original MITGNN)
        weighted_basket = basket_emb * (self.sigma * torch.sigmoid(user_basket_interaction))
        
        # Compute intent attention weights
        intent_weights = self.basket_intent_attention(weighted_basket)  # [batch_size, num_intent]
        
        # Create intent-specific basket representations
        intent_basket_representations = []
        for i in range(self.num_intent):
            intent_weight = intent_weights[:, i:i+1]  # [batch_size, 1]
            intent_repr = weighted_basket * intent_weight
            intent_basket_representations.append(intent_repr)
        
        # Combine intent representations
        combined_intent_basket = torch.stack(intent_basket_representations, dim=1)  # [batch_size, num_intent, embed_size]
        final_basket_emb = torch.sum(combined_intent_basket, dim=1)  # [batch_size, embed_size]
        
        return final_basket_emb
    
    def _compute_predictions(self,
                           user_emb: torch.Tensor,
                           basket_emb: torch.Tensor,
                           item_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute final prediction scores.
        
        Args:
            user_emb: User embeddings [batch_size, embed_size]
            basket_emb: Basket embeddings [batch_size, embed_size]
            item_emb: Item embeddings [batch_size, embed_size]
            
        Returns:
            Prediction scores [batch_size]
        """
        if self.prediction_method == 'dot_product':
            # Simple dot product prediction (matches original MITGNN)
            # Use user embeddings as basket attention (following original implementation)
            basket_att_emb = user_emb
            predictions = torch.sum(basket_att_emb * item_emb, dim=-1)
        else:
            # MLP prediction (default complex method)
            combined_features = torch.cat([user_emb, basket_emb, item_emb], dim=-1)
            predictions = self.prediction_layers(combined_features).squeeze(-1)
        
        return predictions
    
    def _compute_auxiliary_losses(self, enhanced_embeddings: Dict[str, torch.Tensor], 
                                output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary losses for regularization.
        
        Args:
            enhanced_embeddings: Enhanced embeddings from graph convolution
            output: Model output containing embeddings used in current batch
            
        Returns:
            Dictionary of auxiliary losses
        """
        aux_losses = {}
        
        if self.use_explicit_regularization:
            # Explicit regularization matching original MITGNN implementation
            # Regularize only the embeddings used in the current batch
            reg_loss = 0.0
            
            # Regularize user, item, and negative item embeddings used in batch
            if 'user_embeddings' in output:
                reg_loss += torch.norm(output['user_embeddings'], p=2, dim=1).mean()
            if 'item_embeddings' in output:
                reg_loss += torch.norm(output['item_embeddings'], p=2, dim=1).mean()
            if 'neg_item_embeddings' in output:
                reg_loss += torch.norm(output['neg_item_embeddings'], p=2, dim=1).mean()
            
            aux_losses['explicit_reg'] = reg_loss * self.regularization_weight
        else:
            # Default L2 regularization on enhanced embeddings
            reg_loss = 0.0
            for emb_type, embeddings in enhanced_embeddings.items():
                reg_loss += torch.norm(embeddings, p=2)
            
            aux_losses['embedding_reg'] = reg_loss * 0.0001  # Small weight for regularization
        
        return aux_losses
    
    def get_all_item_scores(self, user_ids: torch.Tensor, basket_ids: torch.Tensor) -> torch.Tensor:
        """
        Get prediction scores for all items for given users and baskets.
        Matches original TensorFlow batch_ratings: matmul(basket_attention_embeddings, item_embeddings.T)
        
        Args:
            user_ids: User IDs [batch_size]
            basket_ids: Basket IDs [batch_size]
            
        Returns:
            Scores for all items [batch_size, n_items]
        """
        # Get enhanced embeddings
        enhanced_embeddings = self._apply_graph_convolution()
        
        # Get user and basket embeddings
        user_emb = enhanced_embeddings['user'][user_ids]  # [batch_size, embed_size]
        basket_emb = enhanced_embeddings['basket'][basket_ids]  # [batch_size, embed_size]
        
        # Get all item embeddings  
        all_item_emb = enhanced_embeddings['item']  # [n_items, embed_size]
        
        if self.prediction_method == 'dot_product':
            # Original TensorFlow computation: use user embeddings as basket attention
            # batch_ratings = tf.matmul(b_at_embeddings, pos_i_g_embeddings, transpose_b=True)
            # [batch_size, embed_size] @ [embed_size, n_items] = [batch_size, n_items]
            basket_att_emb = user_emb
            scores = torch.matmul(basket_att_emb, all_item_emb.T)
        else:
            # MLP prediction for all items (more computationally expensive)
            batch_size = user_emb.size(0)
            all_scores = []
            
            # Process in chunks to avoid memory issues
            chunk_size = 512
            for i in range(0, self.n_items, chunk_size):
                end_idx = min(i + chunk_size, self.n_items)
                item_chunk = all_item_emb[i:end_idx]  # [chunk_size, embed_size]
                
                # Expand user and basket embeddings for this chunk
                user_expanded = user_emb.unsqueeze(1).expand(-1, item_chunk.size(0), -1).reshape(-1, user_emb.size(-1))
                basket_expanded = basket_emb.unsqueeze(1).expand(-1, item_chunk.size(0), -1).reshape(-1, basket_emb.size(-1))
                item_expanded = item_chunk.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, item_chunk.size(-1))
                
                # Compute predictions for this chunk
                combined_features = torch.cat([user_expanded, basket_expanded, item_expanded], dim=-1)
                chunk_scores = self.prediction_layers(combined_features).squeeze(-1)
                chunk_scores = chunk_scores.view(batch_size, -1)
                
                all_scores.append(chunk_scores)
            
            scores = torch.cat(all_scores, dim=1)
        
        return scores


if __name__ == "__main__":
    # Minimal sanity check
    import sys
    from pathlib import Path
    
    # Add parent directories to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from utils.config import Config, ModelConfig, DataConfig, TrainingConfig, EvaluationConfig, SystemConfig
        
        # Create test config
        test_config = Config(
            model=ModelConfig(variant="mitgnn", num_intent=5),
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
        model = MITGNNModel(test_config, test_data_config)
        
        # Test forward pass
        batch = {
            'user_id': torch.randint(0, 100, (32,)),
            'basket_id': torch.randint(0, 200, (32,)),
            'item_id': torch.randint(0, 1000, (32,))
        }
        
        output = model(batch)
        
        print("✓ MITGNNModel created and tested successfully")
        print(f"  Output keys: {list(output.keys())}")
        print(f"  Predictions shape: {output['predictions'].shape}")
        
    except Exception as e:
        print(f"✗ MITGNNModel test failed: {e}")
        print("Note: Full testing requires proper imports and data") 