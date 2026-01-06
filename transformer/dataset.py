import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json
import os
import csv


class Vocabulary:
    """Vocabulary class for token-to-ID mapping"""

    def __init__(self, pad_token="<PAD>", bos_token="<BOS>", eos_token="<EOS>", unk_token="<UNK>"):
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        # Initialize with special tokens
        self.token2id = {
            pad_token: 0,
            bos_token: 1,
            eos_token: 2,
            unk_token: 3
        }
        self.id2token = {v: k for k, v in self.token2id.items()}

    def build_vocab(self, texts, min_freq=2, tokenizer_type='char'):
        """Build vocabulary from texts"""
        counter = Counter()

        for text in texts:
            tokens = self.tokenize(text, tokenizer_type)
            counter.update(tokens)

        # Add tokens that meet minimum frequency
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.token2id:
                token_id = len(self.token2id)
                self.token2id[token] = token_id
                self.id2token[token_id] = token

        print(f"Vocabulary built with {len(self.token2id)} tokens")

    def tokenize(self, text, tokenizer_type='char'):
        """Tokenize text based on tokenizer type"""
        if tokenizer_type == 'char':
            return list(text)
        elif tokenizer_type == 'word':
            return text.split()
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    def encode(self, text, tokenizer_type='char', add_special_tokens=True):
        """Convert text to token IDs"""
        tokens = self.tokenize(text, tokenizer_type)
        ids = [self.token2id.get(token, self.token2id[self.unk_token]) for token in tokens]

        if add_special_tokens:
            ids = [self.token2id[self.bos_token]] + ids + [self.token2id[self.eos_token]]

        return ids

    def decode(self, ids, skip_special_tokens=True):
        """Convert token IDs back to text"""
        special_ids = {
            self.token2id[self.pad_token],
            self.token2id[self.bos_token],
            self.token2id[self.eos_token]
        }

        tokens = []
        for id in ids:
            if skip_special_tokens and id in special_ids:
                continue
            tokens.append(self.id2token.get(id, self.unk_token))

        return ''.join(tokens) if len(tokens) > 0 and len(tokens[0]) == 1 else ' '.join(tokens)

    def __len__(self):
        return len(self.token2id)

    def save(self, path):
        """Save vocabulary to file"""
        vocab_dict = {
            'token2id': self.token2id,
            'special_tokens': {
                'pad': self.pad_token,
                'bos': self.bos_token,
                'eos': self.eos_token,
                'unk': self.unk_token
            }
        }
        with open(path, 'w') as f:
            json.dump(vocab_dict, f, indent=2)

    @classmethod
    def load(cls, path):
        """Load vocabulary from file"""
        with open(path, 'r') as f:
            vocab_dict = json.load(f)

        special_tokens = vocab_dict['special_tokens']
        vocab = cls(
            pad_token=special_tokens['pad'],
            bos_token=special_tokens['bos'],
            eos_token=special_tokens['eos'],
            unk_token=special_tokens['unk']
        )
        vocab.token2id = vocab_dict['token2id']
        vocab.id2token = {int(v): k for k, v in vocab.token2id.items()}

        return vocab


class TextDataset(Dataset):
    """Dataset for text data with flexible loading"""

    def __init__(self, config, vocab=None, split='train'):
        self.config = config
        self.split = split
        self.tokenizer_type = config.data.tokenizer_type

        # Load or build vocabulary
        if vocab is None:
            self.vocab = self._build_vocabulary()
        else:
            self.vocab = vocab

        # Load data
        self.src_data, self.tgt_data = self._load_data()

        # Apply train/val split
        if split == 'train':
            split_idx = int(len(self.src_data) * (1 - config.training.val_split))
            self.src_data = self.src_data[:split_idx]
            self.tgt_data = self.tgt_data[:split_idx]
        else:  # val
            split_idx = int(len(self.src_data) * (1 - config.training.val_split))
            self.src_data = self.src_data[split_idx:]
            self.tgt_data = self.tgt_data[split_idx:]

        print(f"{split} dataset size: {len(self.src_data)} examples")

    def _load_data(self):
        """Load data based on dataset type"""
        dataset_type = self.config.data.dataset_type

        if dataset_type == 'parallel':
            return self._load_parallel_files()
        elif dataset_type == 'json':
            return self._load_json_file()
        elif dataset_type == 'csv':
            return self._load_csv_file()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def _load_parallel_files(self):
        """Load parallel text files"""
        src_path = os.path.join(self.config.data.data_path, self.config.data.src_file)
        tgt_path = os.path.join(self.config.data.data_path, self.config.data.tgt_file)

        with open(src_path, 'r', encoding='utf-8') as f:
            src_lines = [line.strip() for line in f if line.strip()]

        with open(tgt_path, 'r', encoding='utf-8') as f:
            tgt_lines = [line.strip() for line in f if line.strip()]

        assert len(src_lines) == len(tgt_lines), "Source and target files must have same number of lines"

        return src_lines, tgt_lines

    def _load_json_file(self):
        """Load JSON/JSONL file with prompt-response pairs"""
        json_path = os.path.join(self.config.data.data_path, self.config.data.src_file)
        src_lines = []
        tgt_lines = []

        with open(json_path, 'r', encoding='utf-8') as f:
            if json_path.endswith('.jsonl'):
                for line in f:
                    data = json.loads(line)
                    src_lines.append(data.get('prompt', data.get('src', data.get('input', ''))))
                    tgt_lines.append(data.get('response', data.get('tgt', data.get('output', ''))))
            else:
                data = json.load(f)
                for item in data:
                    src_lines.append(item.get('prompt', item.get('src', item.get('input', ''))))
                    tgt_lines.append(item.get('response', item.get('tgt', item.get('output', ''))))

        return src_lines, tgt_lines

    def _load_csv_file(self):
        """Load CSV file with source and target columns"""
        csv_path = os.path.join(self.config.data.data_path, self.config.data.src_file)
        src_lines = []
        tgt_lines = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                src_lines.append(row.get('src', row.get('source', row.get('input', ''))))
                tgt_lines.append(row.get('tgt', row.get('target', row.get('output', ''))))

        return src_lines, tgt_lines

    def _build_vocabulary(self):
        """Build vocabulary from all data"""
        vocab_path = os.path.join(self.config.data.data_path, self.config.data.vocab_file)

        # Try to load existing vocabulary
        if os.path.exists(vocab_path):
            print(f"Loading vocabulary from {vocab_path}")
            return Vocabulary.load(vocab_path)

        # Build new vocabulary
        print("Building vocabulary...")
        vocab = Vocabulary(
            pad_token=self.config.data.pad_token,
            bos_token=self.config.data.bos_token,
            eos_token=self.config.data.eos_token,
            unk_token=self.config.data.unk_token
        )

        # Load all data temporarily to build vocab
        src_data, tgt_data = self._load_data()
        all_texts = src_data + tgt_data
        vocab.build_vocab(all_texts, min_freq=self.config.data.min_freq, tokenizer_type=self.tokenizer_type)

        # Save vocabulary
        vocab.save(vocab_path)
        print(f"Vocabulary saved to {vocab_path}")

        return vocab

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        """Get a single example"""
        src_text = self.src_data[idx]
        tgt_text = self.tgt_data[idx]

        # Encode texts
        src_ids = self.vocab.encode(src_text, self.tokenizer_type, add_special_tokens=True)
        tgt_ids = self.vocab.encode(tgt_text, self.tokenizer_type, add_special_tokens=True)

        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long),
            'src_text': src_text,
            'tgt_text': tgt_text
        }


def collate_fn(batch, pad_id=0):
    """Collate function for DataLoader with padding"""
    src_seqs = [item['src'] for item in batch]
    tgt_seqs = [item['tgt'] for item in batch]

    # Pad sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=pad_id)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_id)

    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_text': [item['src_text'] for item in batch],
        'tgt_text': [item['tgt_text'] for item in batch]
    }


def create_dataloaders(config, vocab=None):
    """Create train and validation dataloaders"""
    train_dataset = TextDataset(config, vocab=vocab, split='train')
    val_dataset = TextDataset(config, vocab=train_dataset.vocab, split='val')

    pad_id = train_dataset.vocab.token2id[config.data.pad_token]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id),
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id),
        num_workers=0
    )

    return train_loader, val_loader, train_dataset.vocab
