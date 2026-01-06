import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import os
import json
import argparse
from datetime import datetime

from transformer import Transformer
from dataset import create_dataloaders
from config import Config


class TransformerLRScheduler:
    """Learning rate scheduler from 'Attention is All You Need'"""

    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        step = max(self.current_step, 1)
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))
        return (self.d_model ** (-0.5)) * min(arg1, arg2)


class Trainer:
    """Trainer class for transformer model"""

    def __init__(self, config, model, train_loader, val_loader, vocab):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab

        # Setup device
        self.device = torch.device(config.training.device if torch.cuda.is_available() or config.training.device == "cpu" else "cpu")
        self.model.to(self.device)

        # Get special token IDs
        special_tokens = config.get_special_token_ids(vocab.token2id)
        self.pad_id = special_tokens['pad_id']
        self.bos_id = special_tokens['bos_id']
        self.eos_id = special_tokens['eos_id']

        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id,
            label_smoothing=config.training.label_smoothing
        )

        # Setup optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=config.training.weight_decay
        )

        # Setup learning rate scheduler
        self.scheduler = TransformerLRScheduler(
            self.optimizer,
            d_model=config.model.embed_dim,
            warmup_steps=config.training.warmup_steps
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # Create directories
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(config.training.log_dir, exist_ok=True)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)

            # Prepare decoder input and target
            # Decoder input: [BOS, token1, token2, ..., tokenN-1]
            # Target: [token1, token2, ..., tokenN, EOS]
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(src, tgt_input)

            # Compute loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            # Update progress bar
            if batch_idx % self.config.training.log_every == 0:
                current_lr = self.scheduler.get_lr()
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.6f}'
                })
                self.learning_rates.append(current_lr)

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)

        for batch in tqdm(self.val_loader, desc="Validating"):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = self.model(src, tgt_input)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def generate_samples(self, num_samples=3):
        """Generate sample outputs for monitoring"""
        self.model.eval()
        samples = []

        # Get a few examples from validation set
        for i, batch in enumerate(self.val_loader):
            if i >= num_samples:
                break

            src = batch['src'][:1].to(self.device)  # Take first example
            src_text = batch['src_text'][0]
            tgt_text = batch['tgt_text'][0]

            # Generate
            generated = self.model.generate(
                src,
                max_new_tokens=self.config.model.tgt_max_seq_len,
                temperature=0.8
            )

            # Decode
            generated_text = self.vocab.decode(generated[0].cpu().tolist())

            samples.append({
                'input': src_text,
                'target': tgt_text,
                'generated': generated_text
            })

        return samples

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_step': self.scheduler.current_step,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': {
                'model': vars(self.config.model),
                'training': vars(self.config.training),
                'data': vars(self.config.data)
            }
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.training.checkpoint_dir,
            f"checkpoint_epoch_{self.current_epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = os.path.join(
                self.config.training.checkpoint_dir,
                "best_model.pt"
            )
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")

        # Save latest model (for easy resuming)
        latest_path = os.path.join(
            self.config.training.checkpoint_dir,
            "latest_model.pt"
        )
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.current_step = checkpoint['scheduler_step']
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch + 1}")

    def save_metrics(self):
        """Save training metrics to JSON"""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'current_epoch': self.current_epoch
        }

        metrics_path = os.path.join(self.config.training.log_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def train(self, resume_from=None):
        """Main training loop"""
        if resume_from:
            self.load_checkpoint(resume_from)

        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch

            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")
            print(f"{'='*50}")

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")

            # Validate
            if (epoch + 1) % self.config.training.val_every == 0:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.4f}")

                # Check if best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"New best validation loss: {val_loss:.4f}")

                # Generate samples
                print("\nSample generations:")
                samples = self.generate_samples(num_samples=2)
                for i, sample in enumerate(samples, 1):
                    print(f"\nSample {i}:")
                    print(f"  Input:     {sample['input']}")
                    print(f"  Target:    {sample['target']}")
                    print(f"  Generated: {sample['generated']}")
            else:
                is_best = False

            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every == 0:
                self.save_checkpoint(is_best=is_best)

            # Save metrics
            self.save_metrics()

        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Transformer model')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        config = Config.from_json(args.config)
        print(f"Loaded config from {args.config}")
    else:
        print(f"Config file not found at {args.config}, using default config")
        config = Config()
        config.to_json(args.config)

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, vocab = create_dataloaders(config)
    print(f"Vocabulary size: {len(vocab)}")

    # Update vocab sizes in config if they were default
    config.model.src_vocab_size = len(vocab)
    config.model.tgt_vocab_size = len(vocab)

    # Create model
    print("\nCreating model...")
    model = Transformer(
        src_vocab_size=config.model.src_vocab_size,
        tgt_vocab_size=config.model.tgt_vocab_size,
        src_max_seq_len=config.model.src_max_seq_len,
        tgt_max_seq_len=config.model.tgt_max_seq_len,
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        dim_ffn=config.model.dim_ffn,
        num_layers=config.model.num_layers,
        input_dropout=config.model.input_dropout
    )

    # Create trainer
    trainer = Trainer(config, model, train_loader, val_loader, vocab)

    # Train
    print("\nStarting training...")
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
