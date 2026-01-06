# Transformer from Scratch

A complete, production-ready implementation of the Transformer architecture for sequence-to-sequence tasks including conversational AI, machine translation, and text generation.

Built from scratch in PyTorch with comprehensive training infrastructure, visualization tools, and an interactive chat interface.

---

## 🌟 Features

### Core Architecture
- ✅ **Full Transformer encoder-decoder** with multi-head attention
- ✅ **Sinusoidal positional encoding** (pre-computed for efficiency)
- ✅ **Layer normalization** and residual connections
- ✅ **Scaled dot-product attention** with masking support
- ✅ **Feed-forward networks** with ReLU activation

### Training Infrastructure
- ✅ **Flexible data loading** - Support for parallel files, JSON, JSONL, and CSV
- ✅ **Custom learning rate scheduler** - Transformer warmup schedule from the original paper
- ✅ **Checkpoint management** - Save, load, and resume training
- ✅ **Validation tracking** - Monitor performance with sample generation
- ✅ **Progress visualization** - Plot training curves and metrics
- ✅ **Gradient clipping** and label smoothing
- ✅ **Device flexibility** - CPU, CUDA (NVIDIA), or MPS (Apple Silicon)

### Inference & Interaction
- ✅ **Interactive chat interface** with command support
- ✅ **Temperature-based sampling** - Control randomness in generation
- ✅ **Greedy and sampling decoding** strategies
- ✅ **Character and word-level tokenization**

---

## 📁 Project Structure

```
transformer/
├── Core Model
│   ├── transformer.py          # Main Transformer implementation
│   ├── attentions.py           # Multi-head attention mechanism
│   ├── positional_encoding.py # Sinusoidal positional encoding
│   └── utils.py                # Utility functions
│
├── Training Infrastructure
│   ├── config.py               # Configuration management
│   ├── dataset.py              # Data loading and vocabulary
│   ├── train.py                # Training script with full pipeline
│   └── visualize.py            # Training visualization tools
│
├── Inference
│   └── inference.py            # Interactive chat interface
│
├── Setup & Documentation
│   ├── setup.sh                # Automated environment setup
│   ├── setup_example.py        # Generate example datasets
│   ├── requirements.txt        # Python dependencies
│   ├── README.md               # This file
│   └── GETTING_STARTED.md      # Quick start guide
│
└── Generated During Use
    ├── data/                   # Your datasets
    ├── checkpoints/            # Saved models
    ├── logs/                   # Training metrics
    ├── venv/                   # Virtual environment
    └── config.json             # Your configuration
```

---

## 🚀 Quick Start

### One-Command Setup

```bash
./setup.sh
```

This script will:
1. Create a Python virtual environment
2. Install all dependencies (PyTorch, tqdm, matplotlib)
3. Generate example datasets for testing
4. Create necessary directories

### Train on Example Data

```bash
source venv/bin/activate
python train.py --config config_small.json
```

This trains a small model (2 layers, 128 embed_dim) on the example copy task - perfect for testing!

### Chat with Your Model

```bash
python inference.py --checkpoint ./checkpoints/best_model.pt
```

---

## 📚 Detailed Usage

### 1. Installation

#### Automated Setup (Recommended)
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

#### Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch tqdm matplotlib

# Create directories
mkdir -p data checkpoints logs
```

### 2. Prepare Your Dataset

Choose the format that best fits your data:

#### Option A: Parallel Text Files (Recommended)

Best for translation or paired conversational data.

```bash
# data/src.txt - Input sequences (one per line)
Hello, how are you?
What is your name?
Tell me about transformers

# data/tgt.txt - Target sequences (one per line, matching order)
I'm doing great, thanks!
I'm a chatbot
Transformers are neural network architectures
```

**Requirements:**
- Both files must have the same number of lines
- Line N in `src.txt` corresponds to line N in `tgt.txt`
- UTF-8 encoding

#### Option B: JSON Format

Best for structured conversational data.

```json
[
  {
    "prompt": "Hello, how are you?",
    "response": "I'm doing great, thanks!"
  },
  {
    "prompt": "What is your name?",
    "response": "I'm a chatbot"
  }
]
```

Alternative field names supported: `src/tgt`, `input/output`

**JSONL Format** (one JSON object per line):
```jsonl
{"prompt": "Hello", "response": "Hi there!"}
{"prompt": "How are you?", "response": "I'm good!"}
```

#### Option C: CSV Format

```csv
src,tgt
"Hello, how are you?","I'm doing great!"
"What is your name?","I'm a chatbot"
```

Alternative column names: `source/target`, `input/output`

### 3. Configure Your Training

Generate a default configuration:
```bash
python config.py  # Creates config.json
```

Edit `config.json` to customize your training:

```json
{
  "model": {
    "embed_dim": 256,        // Embedding dimension
    "num_heads": 8,          // Number of attention heads (must divide embed_dim)
    "num_layers": 4,         // Encoder/decoder layers
    "dim_ffn": 1024,         // Feed-forward hidden dimension
    "input_dropout": 0.1     // Dropout rate
  },
  "training": {
    "batch_size": 32,        // Batch size
    "num_epochs": 100,       // Training epochs
    "learning_rate": 0.0001, // Initial learning rate
    "warmup_steps": 4000,    // LR warmup steps
    "device": "mps",         // "cpu", "cuda", or "mps"
    "save_every": 5,         // Save checkpoint every N epochs
    "val_every": 1           // Validate every N epochs
  },
  "data": {
    "data_path": "./data",
    "dataset_type": "parallel",  // "parallel", "json", or "csv"
    "src_file": "src.txt",
    "tgt_file": "tgt.txt",
    "tokenizer_type": "char"     // "char" or "word"
  }
}
```

### 4. Train Your Model

**Basic training:**
```bash
python train.py --config config.json
```

**Resume from checkpoint:**
```bash
python train.py --config config.json --resume ./checkpoints/latest_model.pt
```

**What happens during training:**
- Loads data and builds vocabulary (saved to `data/vocab.json`)
- Initializes model with your configuration
- Trains with progress bars showing loss and learning rate
- Validates and generates sample outputs every N epochs
- Saves checkpoints and metrics automatically
- Creates `best_model.pt` (lowest validation loss) and `latest_model.pt`

### 5. Monitor Training Progress

#### View Training Curves
```bash
python visualize.py --metrics ./logs/metrics.json
```

Displays:
- Training vs validation loss over epochs
- Learning rate schedule visualization

#### Save Plots
```bash
python visualize.py --metrics ./logs/metrics.json --save training_plot.png
```

#### Print Summary
```bash
python visualize.py --metrics ./logs/metrics.json --summary
```

Output:
```
==================================================
TRAINING SUMMARY
==================================================
Current Epoch: 25
Best Validation Loss: 0.8234
Latest Train Loss: 0.7456
Latest Validation Loss: 0.8234
==================================================
```

### 6. Use Your Trained Model

#### Interactive Chat
```bash
python inference.py --checkpoint ./checkpoints/best_model.pt
```

**Available commands in chat:**
- `/temp 0.8` - Set sampling temperature (0.1-2.0)
- `/len 150` - Set maximum response length
- `/reset` - Clear conversation history
- `/help` - Show help message
- `/exit` or `/quit` - Exit chat

**Example session:**
```
You: Hello!
Bot: Hi there! How can I help you?

You: /temp 0.7
Temperature set to 0.7

You: What is AI?
Bot: Artificial Intelligence is the simulation of human intelligence...

You: /exit
Goodbye!
```

#### Single Query Mode
```bash
python inference.py --checkpoint ./checkpoints/best_model.pt --single "Hello, how are you?"
```

#### With Custom Parameters
```bash
python inference.py \
  --checkpoint ./checkpoints/best_model.pt \
  --temperature 0.8 \
  --max-length 100 \
  --device mps
```

---

## ⚙️ Configuration Reference

### Model Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `src_vocab_size` | 10000 | Auto | Source vocabulary size (set from data) |
| `tgt_vocab_size` | 10000 | Auto | Target vocabulary size (set from data) |
| `src_max_seq_len` | 512 | 1-2048 | Maximum source sequence length |
| `tgt_max_seq_len` | 512 | 1-2048 | Maximum target sequence length |
| `embed_dim` | 256 | 64-1024 | Embedding dimension (must be divisible by num_heads) |
| `num_heads` | 8 | 1-16 | Number of attention heads |
| `dim_ffn` | 1024 | 256-4096 | Feed-forward network hidden dimension |
| `num_layers` | 4 | 1-12 | Number of encoder/decoder layers |
| `input_dropout` | 0.1 | 0.0-0.5 | Dropout rate for embeddings |

### Training Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `batch_size` | 32 | 1-256 | Training batch size |
| `num_epochs` | 100 | 1-1000 | Number of training epochs |
| `learning_rate` | 0.0001 | 1e-5 to 1e-3 | Initial learning rate |
| `warmup_steps` | 4000 | 100-10000 | Warmup steps for LR scheduler |
| `max_lr` | 0.0005 | 1e-4 to 1e-2 | Maximum learning rate |
| `weight_decay` | 0.01 | 0.0-0.1 | L2 regularization strength |
| `gradient_clip` | 1.0 | 0.0-10.0 | Gradient clipping threshold (0 to disable) |
| `label_smoothing` | 0.1 | 0.0-0.3 | Label smoothing factor |
| `save_every` | 5 | 1-100 | Save checkpoint every N epochs |
| `val_split` | 0.1 | 0.05-0.3 | Validation set proportion |
| `val_every` | 1 | 1-10 | Validate every N epochs |
| `log_every` | 100 | 10-1000 | Log metrics every N batches |
| `device` | "cuda" | - | Training device: "cpu", "cuda", "mps" |
| `checkpoint_dir` | "./checkpoints" | - | Checkpoint save directory |
| `log_dir` | "./logs" | - | Metrics log directory |

### Data Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_path` | "./data" | Path to dataset directory |
| `dataset_type` | "parallel" | Format: "parallel", "json", or "csv" |
| `src_file` | "src.txt" | Source data file name |
| `tgt_file` | "tgt.txt" | Target data file name |
| `vocab_file` | "vocab.json" | Vocabulary file name |
| `tokenizer_type` | "char" | Tokenization: "char" or "word" |
| `min_freq` | 2 | Minimum token frequency for vocabulary |
| `pad_token` | "\<PAD\>" | Padding token |
| `bos_token` | "\<BOS\>" | Beginning-of-sequence token |
| `eos_token` | "\<EOS\>" | End-of-sequence token |
| `unk_token` | "\<UNK\>" | Unknown token |

---

## 💡 Training Tips & Best Practices

### For Small Datasets (<1000 examples)

```json
{
  "model": {
    "embed_dim": 128,
    "num_layers": 2,
    "input_dropout": 0.2
  },
  "training": {
    "batch_size": 16,
    "warmup_steps": 1000
  }
}
```

**Why?**
- Smaller model prevents overfitting
- Higher dropout for regularization
- Smaller batch size works better with limited data

### For Large Datasets (>10,000 examples)

```json
{
  "model": {
    "embed_dim": 512,
    "num_layers": 6,
    "dim_ffn": 2048
  },
  "training": {
    "batch_size": 64,
    "warmup_steps": 8000
  }
}
```

**Why?**
- Larger model can capture more patterns
- More layers for deeper representations
- Larger batches for stable gradients

### Improving Generation Quality

1. **Train longer**: Increase `num_epochs`
2. **Tune temperature**:
   - Lower (0.5-0.7) = more focused/repetitive
   - Medium (0.8-1.0) = balanced
   - Higher (1.1-1.5) = more creative/random
3. **Better data**: Quality > quantity
4. **Use word-level tokenization** for better semantic understanding
5. **Add label smoothing** to prevent overconfidence

### Faster Training

1. **Use GPU**: Set `device="mps"` (Mac) or `device="cuda"` (NVIDIA)
2. **Increase batch size**: If GPU memory allows
3. **Reduce validation**: Set `val_every=5` or higher
4. **Mixed precision**: Consider adding AMP (Automatic Mixed Precision)

### Memory Management

If you encounter **Out of Memory** errors:

```json
{
  "model": {
    "embed_dim": 128,     // Reduce from 256
    "num_layers": 2,      // Reduce from 4
    "src_max_seq_len": 256,  // Reduce from 512
    "tgt_max_seq_len": 256
  },
  "training": {
    "batch_size": 8       // Reduce from 32
  }
}
```

---

## 🏗️ Architecture Details

### Transformer Model

```
Input Sequence
    ↓
[Embedding + Positional Encoding]
    ↓
┌─────────────────────────────────┐
│   ENCODER (×N layers)           │
│  ┌──────────────────────┐       │
│  │ Multi-Head Attention │       │
│  └──────────────────────┘       │
│           ↓                      │
│  ┌──────────────────────┐       │
│  │ Feed-Forward Network │       │
│  └──────────────────────┘       │
│  (+ Residual + LayerNorm)       │
└─────────────────────────────────┘
    ↓ Encoder Output

Target Sequence
    ↓
[Embedding + Positional Encoding]
    ↓
┌─────────────────────────────────┐
│   DECODER (×N layers)           │
│  ┌──────────────────────────────┐│
│  │ Masked Multi-Head Attention ││
│  └──────────────────────────────┘│
│           ↓                       │
│  ┌──────────────────────────────┐│
│  │ Cross-Attention              ││
│  │ (attends to Encoder Output)  ││
│  └──────────────────────────────┘│
│           ↓                       │
│  ┌──────────────────────────────┐│
│  │ Feed-Forward Network         ││
│  └──────────────────────────────┘│
│  (+ Residual + LayerNorm)        │
└──────────────────────────────────┘
    ↓
[Linear Projection to Vocabulary]
    ↓
Output Probabilities
```

### Key Components

**1. Scaled Dot-Product Attention**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**2. Multi-Head Attention**
- Splits embedding into multiple heads
- Each head learns different aspects
- Concatenate and project outputs

**3. Positional Encoding**
- Sinusoidal functions: PE(pos, 2i) = sin(pos/10000^(2i/d))
- Allows model to use position information
- Pre-computed for efficiency

**4. Learning Rate Schedule**
```python
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
```
- Warmup phase: Linear increase
- Decay phase: Inverse square root decay

**5. Teacher Forcing**
- Training: Decoder sees ground-truth previous tokens
- Inference: Decoder sees its own predictions

---

## 📖 Example Workflows

### 1. Machine Translation (English → French)

```bash
# Prepare data
cat > data/src.txt << EOF
Hello
Good morning
Thank you
Goodbye
EOF

cat > data/tgt.txt << EOF
Bonjour
Bon matin
Merci
Au revoir
EOF

# Configure
python config.py
# Edit config.json: set tokenizer_type="word"

# Train
python train.py --config config.json

# Test
python inference.py --single "Hello"
```

### 2. Chatbot Training

```bash
# Prepare conversations
cat > data/conversations.json << 'EOF'
[
  {"prompt": "Hi", "response": "Hello! How can I help?"},
  {"prompt": "What's your name?", "response": "I'm a transformer chatbot."},
  {"prompt": "How are you?", "response": "I'm doing great, thanks!"},
  {"prompt": "Tell me a joke", "response": "Why did the neural network go to therapy? Too many issues!"}
]
EOF

# Configure
python config.py
# Edit config.json:
#   - set dataset_type="json"
#   - set src_file="conversations.json"

# Train
python train.py --config config.json

# Chat
python inference.py
```

### 3. Text Completion

```bash
# Prepare sentence pairs
cat > data/src.txt << EOF
The weather today is
I really enjoy
My favorite food is
EOF

cat > data/tgt.txt << EOF
sunny and warm
reading books and learning new things
pizza with extra cheese
EOF

# Train and test as above
```

---

## 🎓 Citation

This implementation is based on the original Transformer paper:

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

---

## 🔗 Additional Resources

- **Quick Start Guide**: See [GETTING_STARTED.md](GETTING_STARTED.md)
- **Original Paper**: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- **PyTorch Documentation**: [pytorch.org](https://pytorch.org/docs/)

