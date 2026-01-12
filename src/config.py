import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a Transformer model from scratch using PyTorch"
    )

    # =========================
    # Dataset & Languages
    # =========================
    parser.add_argument(
        "--lang_src",
        type=str,
        default="en",
        help='ISO code for the source language (e.g. "en")'
    )
    parser.add_argument(
        "--lang_tgt",
        type=str,
        default="it",
        help='ISO code for the target language (e.g. "it")'
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=350,
        help="Maximum sequence length for source and target"
    )

    # =========================
    # Training Batch Sizes
    # =========================
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size used during training"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Batch size used during evaluation"
    )

    # =========================
    # Transformer Hyperparameters
    # =========================
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="Embedding dimension"
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=6,
        help="Number of encoder/decoder layers"
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=2048,
        help="Feed-forward network hidden size"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )

    # =========================
    # Optimization
    # =========================
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer"
    )

    # =========================
    # Checkpointing
    # =========================
    parser.add_argument(
        "--model_folder",
        type=str,
        default="weights",
        help="Directory to store model checkpoints"
    )
    parser.add_argument(
        "--model_basename",
        type=str,
        default="tmodel_",
        help="Prefix for saved model files"
    )
    parser.add_argument(
        "--preload",
        type=str,
        default=None,
        help="Checkpoint path or identifier to resume training from"
    )

    # =========================
    # Device
    # =========================
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device: cuda | mps | cpu"
    )

    # =========================
    # Tokenizer & Logging
    # =========================
    parser.add_argument(
        "--tokenizer_file",
        type=str,
        default="tokenizer_{0}.json",
        help="Tokenizer filename pattern (use .format(i))"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="tmodel_en_it",
        help="Experiment name (used for logging & tracking)"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs",
        help="Base directory for logs (TensorBoard / experiments)"
    )

    return parser.parse_args()
