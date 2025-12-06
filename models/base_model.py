"""
Base model interface for the unified basket prediction pipeline.

This module defines the abstract base class that all model variants must implement.
It provides common functionality and ensures consistency across different approaches.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from utils.config import Config


logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all basket prediction models.
    
    This class defines the interface that all model variants (MITGNN, GRNN, TGN)
    must implement while providing common functionality for initialization,
    embedding creation, and utility methods.
    """
    
    def __init__(self, config: Config, data_config: Dict[str, Any]):
        """
        Initialize base model.
        
        Args:
            config: Global configuration object
            data_config: Data configuration from data module
        """
        super(BaseModel, self).__init__()
        
        self.config = config
        self.data_config = data_config
        
        # Extract common parameters
        self.n_users = data_config['n_users']
        self.n_baskets = data_config['n_baskets'] 
        self.n_items = data_config['n_items']
        self.embed_size = config.model.embed_size
        self.layer_sizes = config.model.layer_sizes
        self.dropout = config.training.dropout
        
        # Model variant
        self.variant = config.model.variant
        
        # Initialize embeddings
        self._init_embeddings()
        
        # Initialize variant-specific components
        self._init_variant_components()
        
        logger.info(f"Initialized {self.variant.upper()} model")
        logger.info(f"Users: {self.n_users}, Baskets: {self.n_baskets}, Items: {self.n_items}")
    
    def _init_embeddings(self) -> None:
        """Initialize embedding layers for users, baskets, and items."""
        # User embeddings
        self.user_embedding = nn.Embedding(
            num_embeddings=self.n_users,
            embedding_dim=self.embed_size,
            padding_idx=0
        )
        
        # Basket embeddings
        self.basket_embedding = nn.Embedding(
            num_embeddings=self.n_baskets,
            embedding_dim=self.embed_size,
            padding_idx=0
        )
        
        # Item embeddings
        self.item_embedding = nn.Embedding(
            num_embeddings=self.n_items,
            embedding_dim=self.embed_size,
            padding_idx=0
        )
        
        # If price vector is provided and use_price is enabled, initialize item embeddings from it
        try:
            use_price = getattr(self.config.model, 'use_price', False)
            price_vec = None
            if use_price and isinstance(self.data_config, dict):
                price_vec = self.data_config.get('item_price_vector')
            if use_price and price_vec is not None:
                # Create a price-based embedding following TF style: first dim = z-score price,
                # remaining dims sinusoidal transforms; then project to embed_size
                import math
                import numpy as np
                price_np = price_vec if isinstance(price_vec, np.ndarray) else None
                if price_np is None:
                    # Convert torch to numpy if needed
                    try:
                        price_np = price_vec.detach().cpu().numpy()
                    except Exception:
                        price_np = None
                if price_np is not None:
                    n_items = int(self.n_items)
                    emb = np.zeros((n_items, self.embed_size), dtype=np.float32)
                    for idx in range(n_items):
                        p = float(price_np[idx]) if idx < len(price_np) else 0.0
                        if self.embed_size > 0:
                            emb[idx, 0] = p
                        # Fill rest with sin/cos frequencies similar to TF
                        for j in range(1, self.embed_size, 2):
                            val = (10000.0 ** (-(j / max(self.embed_size, 1))))
                            if j < self.embed_size:
                                emb[idx, j] = math.sin(p * val)
                            if j + 1 < self.embed_size:
                                emb[idx, j + 1] = math.cos(p * val)
                    with torch.no_grad():
                        self.item_embedding.weight.copy_(torch.from_numpy(emb))
                    logger.info("Initialized item embeddings from price vector (multimodal)")
        except Exception as e:
            logger.warning(f"Price-based item embedding initialization skipped: {e}")
        
        # Initialize embeddings with normal distribution
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.basket_embedding.weight, std=0.1)
        # Do not overwrite price-initialized item embeddings
        if not (getattr(self.config.model, 'use_price', False) and isinstance(self.data_config, dict) and self.data_config.get('item_price_vector') is not None):
            nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # Set padding embeddings to zero
        with torch.no_grad():
            self.user_embedding.weight[0].fill_(0)
            self.basket_embedding.weight[0].fill_(0)
            self.item_embedding.weight[0].fill_(0)
        
        logger.info("Initialized embedding layers")
    
    @abstractmethod
    def _init_variant_components(self) -> None:
        """
        Initialize variant-specific components.
        
        This method must be implemented by each model variant to set up
        their specific layers and components.
        """
        pass
    
    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            batch: Input batch containing user_id, basket_id, item_id, etc.
            
        Returns:
            Dictionary containing predictions and auxiliary outputs
        """
        pass
    
    def get_user_embeddings(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Get user embeddings.
        
        Args:
            user_ids: User IDs tensor [batch_size]
            
        Returns:
            User embeddings [batch_size, embed_size]
        """
        return self.user_embedding(user_ids)
    
    def get_basket_embeddings(self, basket_ids: torch.Tensor) -> torch.Tensor:
        """
        Get basket embeddings.
        
        Args:
            basket_ids: Basket IDs tensor [batch_size]
            
        Returns:
            Basket embeddings [batch_size, embed_size]
        """
        return self.basket_embedding(basket_ids)
    
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Get item embeddings.
        
        Args:
            item_ids: Item IDs tensor [batch_size]
            
        Returns:
            Item embeddings [batch_size, embed_size]
        """
        return self.item_embedding(item_ids)
    
    def compute_prediction_score(self, 
                                user_emb: torch.Tensor,
                                basket_emb: torch.Tensor, 
                                item_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction score for user-basket-item interaction.
        
        Args:
            user_emb: User embeddings [batch_size, embed_size]
            basket_emb: Basket embeddings [batch_size, embed_size]
            item_emb: Item embeddings [batch_size, embed_size]
            
        Returns:
            Prediction scores [batch_size]
        """
        # Simple dot product interaction
        # Can be overridden by subclasses for more complex interactions
        combined_emb = user_emb + basket_emb  # Simple combination
        scores = torch.sum(combined_emb * item_emb, dim=-1)
        return scores
    
    def apply_dropout(self, embeddings: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply dropout to embeddings.
        
        Args:
            embeddings: Input embeddings
            training: Whether in training mode
            
        Returns:
            Embeddings with dropout applied
        """
        if training and self.dropout > 0:
            return F.dropout(embeddings, p=self.dropout, training=training)
        return embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information for logging.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'variant': self.variant,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'embed_size': self.embed_size,
            'layer_sizes': self.layer_sizes,
            'n_users': self.n_users,
            'n_baskets': self.n_baskets,
            'n_items': self.n_items
        }
    
    def save_embeddings(self, save_path: str) -> None:
        """
        Save learned embeddings to file.
        
        Args:
            save_path: Path to save embeddings
        """
        embeddings = {
            'user_embeddings': self.user_embedding.weight.data.cpu().numpy(),
            'basket_embeddings': self.basket_embedding.weight.data.cpu().numpy(),
            'item_embeddings': self.item_embedding.weight.data.cpu().numpy()
        }
        
        torch.save(embeddings, save_path)
        logger.info(f"Saved embeddings to {save_path}")
    
    def load_embeddings(self, load_path: str) -> None:
        """
        Load pre-trained embeddings from file.
        
        Args:
            load_path: Path to load embeddings from
        """
        embeddings = torch.load(load_path, map_location='cpu')
        
        if 'user_embeddings' in embeddings:
            self.user_embedding.weight.data = torch.from_numpy(embeddings['user_embeddings'])
        if 'basket_embeddings' in embeddings:
            self.basket_embedding.weight.data = torch.from_numpy(embeddings['basket_embeddings'])
        if 'item_embeddings' in embeddings:
            self.item_embedding.weight.data = torch.from_numpy(embeddings['item_embeddings'])
        
        logger.info(f"Loaded embeddings from {load_path}")
    
    def freeze_embeddings(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze embedding layers.
        
        Args:
            freeze: Whether to freeze embeddings
        """
        for param in self.user_embedding.parameters():
            param.requires_grad = not freeze
        for param in self.basket_embedding.parameters():
            param.requires_grad = not freeze
        for param in self.item_embedding.parameters():
            param.requires_grad = not freeze
        
        status = "frozen" if freeze else "unfrozen"
        logger.info(f"Embeddings {status}")
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss for embeddings.
        
        Returns:
            L2 regularization loss
        """
        reg_loss = 0.0
        
        # L2 regularization on embeddings
        reg_loss += torch.norm(self.user_embedding.weight, p=2)
        reg_loss += torch.norm(self.basket_embedding.weight, p=2)
        reg_loss += torch.norm(self.item_embedding.weight, p=2)
        
        return reg_loss
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device


if __name__ == "__main__":
    # Minimal sanity check
    print("✓ BaseModel defined successfully")
    print("Note: This is an abstract class and cannot be instantiated directly") 