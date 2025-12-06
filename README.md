# Unified Basket Prediction Pipeline 🛒

A production-ready PyTorch implementation that unifies **MITGNN**, **GRNN**, and **TGN** models for next-basket recommendation. This pipeline consolidates three separate TensorFlow implementations into a single, configurable framework driven entirely by YAML configuration.

## 🚀 Key Features

- **Unified Architecture**: Single codebase supporting all three model variants
- **YAML Configuration**: No command-line complexity - everything controlled via config files
- **Production Ready**: Clean code with type hints, comprehensive logging, and error handling
- **Extensible Design**: Plugin system for variant-specific behavior without core changes
- **Comprehensive Evaluation**: Full metrics suite with detailed explanations
- **Reproducible**: Deterministic behavior with proper seed management

## 📁 Project Structure

```
pytorch_pipeline/
├── run.py                    # Main entry point
├── configs/
│   └── experiment.yaml       # Global configuration
├── trainer/
│   ├── __init__.py
│   └── trainer.py           # Generic training loop with plugin support
├── data/
│   ├── __init__.py
│   ├── base_data.py         # BaseDataModule (common functionality)
│   ├── mitgnn_data.py       # MITGNNDataModule
│   ├── grnn_data.py         # GRNNDataModule (sequential processing)
│   └── tgn_data.py          # TGNDataModule (temporal processing)
├── models/                   # Model implementations (to be completed)
│   ├── __init__.py
│   ├── base_model.py        # BaseModel interface
│   ├── mitgnn_model.py      # MITGNN implementation
│   ├── grnn_model.py        # MITGNN + GRNN implementation
│   ├── tgn_model.py         # MITGNN + TGN implementation
│   └── components.py        # Shared components
├── evaluator/
│   ├── __init__.py
│   ├── evaluator.py         # Common evaluator
│   └── metrics.py           # Metrics with detailed documentation
├── plugins/                  # Variant-specific extensions (to be completed)
│   ├── __init__.py
│   └── training_plugins.py
├── utils/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── logger.py           # Structured logging
│   └── reproducibility.py  # Seed management
└── requirements.txt
```

## 🔧 Installation

1. **Create and activate virtual environment:**
```bash
cd pytorch_pipeline
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Basic Training
```bash
# Train MITGNN model
python3 run.py --config configs/experiment.yaml

# Train GRNN model (edit config first to set variant: "grnn")
python3 run.py --config configs/experiment.yaml

# Train TGN model (edit config first to set variant: "tgn")
python3 run.py --config configs/experiment.yaml
```

### Testing
```bash
# Test trained model
python3 run.py --config configs/experiment.yaml --mode test --resume path/to/checkpoint.pth

# Automatic checkpoint detection
python3 run.py --config configs/experiment.yaml --mode test
```

### Advanced Usage
```bash
# Override output directory
python3 run.py --config configs/experiment.yaml --output-dir custom_output

# Force CPU usage
python3 run.py --config configs/experiment.yaml --device cpu

# Verbose logging
python3 run.py --config configs/experiment.yaml --verbose

# Dry run (validation only)
python3 run.py --config configs/experiment.yaml --dry-run
```

## ⚙️ Configuration

The entire pipeline is controlled through a single YAML file. Key sections:

### Model Configuration
```yaml
model:
  variant: "mitgnn"  # Options: "mitgnn", "grnn", "tgn"
  embed_size: 64
  layer_sizes: [64, 64, 64]
  num_intent: 5
  
  # GRNN-specific
  grnn:
    rnn_hidden_dim: 64
    sequence_max_length: 10
    rnn_type: "LSTM"
  
  # TGN-specific  
  tgn:
    temporal_dim: 2
    time_encoding: "cosine"
```

### Data Configuration
```yaml
data:
  data_path: "BasketPrediction/Codes-MITGNN/DataPreparationandTransformation"
  datagroup: "groceries_data"
  dataset: "grocerieswithres"
  within_basket: true
  batch_size: 256
```

### Training Configuration
```yaml
training:
  lr: 0.0001
  epochs: 200
  early_stopping:
    enabled: true
    patience: 20
    metric: "recall@20"
```

## 📊 Model Variants

### MITGNN (Multi-Intent Graph Neural Network)
- **Focus**: Multi-intent modeling through graph convolution
- **Key Features**: Intent separation, collaborative filtering
- **Use Case**: Basic basket prediction with diverse user preferences

### GRNN (Graph Recurrent Neural Network)  
- **Focus**: Sequential modeling with RNN components
- **Key Features**: Temporal sequences, LSTM/GRU processing
- **Use Case**: Sequential basket prediction considering purchase order

### TGN (Temporal Graph Network)
- **Focus**: Time-aware graph neural networks
- **Key Features**: Temporal embeddings, time encoding
- **Use Case**: Time-sensitive recommendations with temporal patterns

## 📈 Evaluation Metrics

The pipeline includes comprehensive evaluation with detailed metric explanations:

### Core Metrics
- **Recall@K**: Fraction of relevant items in top-K recommendations
- **NDCG@K**: Normalized DCG considering ranking quality
- **Hit Ratio@K**: Percentage of users with at least one hit
- **Precision@K**: Fraction of top-K recommendations that are relevant

### Additional Metrics
- **AUC**: Area under ROC curve
- **Average Precision**: Area under precision-recall curve
- **Coverage@K**: Diversity of recommended items
- **Novelty@K**: Unexpectedness based on popularity

Each metric includes:
- Clear definition and intuition
- Mathematical formula
- Implementation with proper edge case handling

## 🗂️ Data Schema

### Expected Input Files
```
data_directory/
├── train_u2b.txt     # User-to-basket mappings
├── train_b2i.txt     # Basket-to-item mappings  
├── test_b2i.txt      # Test basket-to-item mappings
└── basket_timestamps.txt  # Optional: timestamps for TGN
```

### File Formats
- **train_u2b.txt**: `user_id basket_id1 basket_id2 ...`
- **train_b2i.txt**: `basket_id item_id1 item_id2 ...`
- **test_b2i.txt**: `basket_id item_id1 item_id2 ...`
- **basket_timestamps.txt**: `basket_id timestamp` (TGN only)

### Data Types
- User IDs: `int32`, range [0, n_users)
- Basket IDs: `int32`, range [0, n_baskets)
- Item IDs: `int32`, range [0, n_items)
- Timestamps: `float32` (Unix timestamp or relative time)

## 🔌 Extension Points

### Adding New Model Variants
1. Create new data module inheriting from `BaseDataModule`
2. Implement model class inheriting from `BaseModel`
3. Add variant to `run.py` model factory
4. Update configuration schema

### Custom Plugins
The plugin system allows variant-specific behavior without modifying core trainer:
```python
# plugins/training_plugins.py
class CustomTrainingPlugin:
    def on_epoch_start(self, trainer, epoch):
        # Custom behavior
        pass
```

## 🏁 Future Improvements

The code includes `FUTURE_IMPROVEMENT` markers for areas of enhancement:

- **Distributed Training**: Multi-GPU and multi-node support
- **Mixed Precision**: FP16 training for memory efficiency
- **Dynamic Batching**: Adaptive batch sizing
- **Inference Pipeline**: Production deployment support
- **Advanced Temporal Features**: Multiple time granularities

## 🧪 Testing

### Unit Tests
```bash
# Test individual components
python3 -m pytest tests/

# Test configuration loading
python3 utils/config.py

# Test metrics computation  
python3 evaluator/metrics.py

# Test data loading
python3 data/base_data.py
```

### Integration Testing
```bash
# Dry run test
python3 run.py --config configs/experiment.yaml --dry-run

# Quick training test (1 epoch)
python3 run.py --config configs/test.yaml
```

## 🔧 Troubleshooting

### Common Issues

**Configuration Errors**
```bash
# Validate configuration
python3 utils/config.py
```

**Data Loading Issues**
- Verify file paths in configuration
- Check data file formats match expected schema
- Ensure proper file permissions

**Memory Issues**
- Reduce batch size in configuration
- Use gradient accumulation
- Enable mixed precision (future feature)

**Reproducibility Issues**
- Verify deterministic flag in configuration
- Check random seed consistency
- Note GPU non-deterministic operations

## 📝 Logging

The pipeline provides structured logging with multiple levels:

- **Console Output**: Training progress and key metrics
- **File Logging**: Detailed logs with timestamps and context
- **TensorBoard**: Training curves and model visualizations
- **Metrics Logging**: Structured evaluation results

## 🏗️ Development Status

### ✅ Completed Components
- Configuration management with validation
- Data loading pipeline for all variants
- Training loop with early stopping
- Comprehensive evaluation metrics
- Logging and reproducibility utilities
- CLI interface and argument parsing

### 🚧 In Progress
- Model implementations (base classes defined)
- Plugin system implementation
- Advanced evaluation features

### 📋 Planned
- Inference pipeline
- Model deployment utilities
- Advanced temporal features
- Distributed training support

## 📖 References

This pipeline consolidates and improves upon the original TensorFlow implementations:
- **MITGNN**: Multi-Intent Translation Graph Neural Network
- **GRNN**: Graph Recurrent Neural Network  
- **TGN**: Temporal Graph Network

The unified PyTorch implementation provides better maintainability, extensibility, and production readiness while preserving the core algorithmic contributions of each approach.

---

**Built with ❤️ for the Columbia University basket prediction project** 