import json
import os
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Model architecture hyperparameters"""
    src_vocab_size: int = 10000
    tgt_vocab_size: int = 10000
    src_max_seq_len: int = 512
    tgt_max_seq_len: int = 512
    embed_dim: int = 256
    num_heads: int = 8
    dim_ffn: int = 1024
    num_layers: int = 4
    input_dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.0001
    warmup_steps: int = 4000
    max_lr: float = 0.0005
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1

    # Checkpoint settings
    save_every: int = 5
    checkpoint_dir: str = "./checkpoints"

    # Validation
    val_split: float = 0.1
    val_every: int = 1

    # Logging
    log_dir: str = "./logs"
    log_every: int = 100

    # Device
    device: str = "cuda"  # or "cpu" or "mps" for Mac


@dataclass
class DataConfig:
    """Dataset configuration"""
    data_path: str = "./data"
    dataset_type: str = "parallel"  # "parallel", "json", or "csv"
    src_file: str = "src.txt"
    tgt_file: str = "tgt.txt"
    vocab_file: str = "vocab.json"
    tokenizer_type: str = "char"  # "char" or "word"
    min_freq: int = 2  # Minimum frequency for vocabulary

    # Special tokens
    pad_token: str = "<PAD>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    unk_token: str = "<UNK>"


class Config:
    """Main configuration class"""

    def __init__(self, model_config=None, training_config=None, data_config=None):
        self.model = model_config or ModelConfig()
        self.training = training_config or TrainingConfig()
        self.data = data_config or DataConfig()

    @classmethod
    def from_json(cls, json_path):
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))

        return cls(model_config, training_config, data_config)

    def to_json(self, json_path):
        """Save configuration to JSON file"""
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data)
        }

        os.makedirs(os.path.dirname(json_path) if os.path.dirname(json_path) else '.', exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def get_special_token_ids(self, vocab):
        """Get special token IDs from vocabulary"""
        return {
            'pad_id': vocab.get(self.data.pad_token, 0),
            'bos_id': vocab.get(self.data.bos_token, 1),
            'eos_id': vocab.get(self.data.eos_token, 2),
            'unk_id': vocab.get(self.data.unk_token, 3)
        }


def create_default_config(save_path="config.json"):
    """Create and save a default configuration"""
    config = Config()
    config.to_json(save_path)
    print(f"Default configuration saved to {save_path}")
    return config


if __name__ == "__main__":
    # Create a default config file when running this script directly
    create_default_config()
