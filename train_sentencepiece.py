import argparse
import sentencepiece as spm

def train_sentencepiece(args):
    """
    Train a SentencePiece model using the provided arguments.
    """
    # Build training command
    spm_cmd = (
        f"--input={args.input} "
        f"--model_prefix={args.model_prefix} "
        f"--vocab_size={args.vocab_size} "
        f"--character_coverage={args.character_coverage} "
        f"--model_type={args.model_type} "
        f"--unk_piece={args.unk_piece} "
        f"--pad_piece={args.pad_piece} "
        f"--bos_piece={args.bos_piece} "
        f"--eos_piece={args.eos_piece} "
    )
    # Train the model
    spm.SentencePieceTrainer.Train(spm_cmd)
    print(f"Trained SentencePiece model and vocab saved with prefix '{args.model_prefix}'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece model for ASR language modeling"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the input text file for training (one sentence per line)"
    )
    parser.add_argument(
        "--model_prefix", type=str, default="spm_model",
        help="Prefix for the output model and vocab files"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=1024,
        help="Size of the vocabulary (number of tokens)"
    )
    parser.add_argument(
        "--character_coverage", type=float, default=1.0,
        help="Amount of characters covered by the model (0.98 covers 98%% of characters)"
    )
    parser.add_argument(
        "--model_type", type=str, choices=["unigram", "bpe", "char", "word"],
        default="unigram",
        help="Type of model: unigram, bpe, char, or word"
    )
    parser.add_argument(
        "--unk_piece", type=str, default="<unk>",
        help="String for the UNK token"
    )
    parser.add_argument(
        "--pad_piece", type=str, default="<pad>",
        help="String for the PAD token"
    )
    parser.add_argument(
        "--bos_piece", type=str, default="<s>",
        help="String for the BOS token"
    )
    parser.add_argument(
        "--eos_piece", type=str, default="</s>",
        help="String for the EOS token"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_sentencepiece(args)

