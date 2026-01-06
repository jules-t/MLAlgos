import torch
import argparse
import os
from transformer import Transformer
from dataset import Vocabulary
from config import ModelConfig


class ChatBot:
    """Interactive chatbot using trained transformer"""

    def __init__(self, checkpoint_path, device='cpu'):
        self.device = torch.device(device)

        # Load checkpoint
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load configuration
        model_config_dict = checkpoint['config']['model']
        self.model_config = ModelConfig(**model_config_dict)

        data_config = checkpoint['config']['data']

        # Load vocabulary
        vocab_path = os.path.join(data_config['data_path'], data_config['vocab_file'])
        print(f"Loading vocabulary from {vocab_path}...")
        self.vocab = Vocabulary.load(vocab_path)
        print(f"Vocabulary size: {len(self.vocab)}")

        # Create and load model
        print("Creating model...")
        self.model = Transformer(
            src_vocab_size=self.model_config.src_vocab_size,
            tgt_vocab_size=self.model_config.tgt_vocab_size,
            src_max_seq_len=self.model_config.src_max_seq_len,
            tgt_max_seq_len=self.model_config.tgt_max_seq_len,
            embed_dim=self.model_config.embed_dim,
            num_heads=self.model_config.num_heads,
            dim_ffn=self.model_config.dim_ffn,
            num_layers=self.model_config.num_layers,
            input_dropout=self.model_config.input_dropout
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from epoch {checkpoint['epoch'] + 1}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")

        # Settings
        self.temperature = 1.0
        self.max_length = min(self.model_config.tgt_max_seq_len, 200)
        self.tokenizer_type = data_config.get('tokenizer_type', 'char')

    def generate_response(self, input_text):
        """Generate response for input text"""
        # Encode input
        input_ids = self.vocab.encode(input_text, self.tokenizer_type, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_tensor,
                max_new_tokens=self.max_length,
                temperature=self.temperature
            )

        # Decode
        response = self.vocab.decode(output[0].cpu().tolist(), skip_special_tokens=True)
        return response

    def chat(self):
        """Interactive chat loop"""
        print("\n" + "=" * 50)
        print("TRANSFORMER CHATBOT")
        print("=" * 50)
        print("\nCommands:")
        print("  /exit or /quit - Exit the chat")
        print("  /temp <value>  - Set temperature (e.g., /temp 0.8)")
        print("  /len <value>   - Set max response length (e.g., /len 100)")
        print("  /reset         - Reset conversation")
        print("  /help          - Show this help message")
        print("\n" + "=" * 50 + "\n")

        conversation_history = []

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.split()[0].lower()

                    if command in ['/exit', '/quit']:
                        print("Goodbye!")
                        break

                    elif command == '/temp':
                        try:
                            temp = float(user_input.split()[1])
                            self.temperature = max(0.1, min(temp, 2.0))
                            print(f"Temperature set to {self.temperature}")
                        except (IndexError, ValueError):
                            print("Usage: /temp <value> (e.g., /temp 0.8)")

                    elif command == '/len':
                        try:
                            length = int(user_input.split()[1])
                            self.max_length = max(10, min(length, self.model_config.tgt_max_seq_len))
                            print(f"Max length set to {self.max_length}")
                        except (IndexError, ValueError):
                            print("Usage: /len <value> (e.g., /len 100)")

                    elif command == '/reset':
                        conversation_history = []
                        print("Conversation reset.")

                    elif command == '/help':
                        print("\nCommands:")
                        print("  /exit or /quit - Exit the chat")
                        print("  /temp <value>  - Set temperature")
                        print("  /len <value>   - Set max response length")
                        print("  /reset         - Reset conversation")
                        print("  /help          - Show this help message\n")

                    else:
                        print(f"Unknown command: {command}")

                    continue

                # Generate response
                response = self.generate_response(user_input)
                print(f"Bot: {response}\n")

                # Update conversation history (optional, for future context-aware models)
                conversation_history.append({
                    'user': user_input,
                    'bot': response
                })

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.")


def main():
    parser = argparse.ArgumentParser(description='Interactive chat with trained transformer')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to run inference on')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0)')
    parser.add_argument('--max-length', type=int, default=None,
                        help='Maximum response length (default: model max)')
    parser.add_argument('--single', type=str, default=None,
                        help='Generate single response and exit (non-interactive)')

    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Please train the model first or provide a valid checkpoint path.")
        return

    # Create chatbot
    chatbot = ChatBot(args.checkpoint, device=args.device)

    # Set parameters
    if args.temperature:
        chatbot.temperature = args.temperature
    if args.max_length:
        chatbot.max_length = args.max_length

    # Single query mode or interactive mode
    if args.single:
        response = chatbot.generate_response(args.single)
        print(f"Input: {args.single}")
        print(f"Response: {response}")
    else:
        chatbot.chat()


if __name__ == "__main__":
    main()
