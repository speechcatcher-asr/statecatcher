#!/usr/bin/env python3
import argparse
import os
import shutil
import time
import math
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torchaudio
import sentencepiece as spm
import dataset
from jiwer import wer
from model import make_frontend, ASRModel, build_encoder


# Try to import RNN-T loss; if unavailable, only CTC will work
try:
    from warp_rnnt import RNNTLoss
    HAVE_RNNT = True
except ImportError:
    HAVE_RNNT = False

try:
    import aim
    HAVE_AIM = True
except ImportError:
    HAVE_AIM = False

class RNNTPredictorJoiner(nn.Module):
    """
    Minimal RNN-T predictor+joiner:
      - Embeds the label sequence (including blank prefix)
      - Adds encoder and predictor embeddings
      - Projects to vocab-size logits
    """
    def __init__(self, enc_dim: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, enc_dim)
        self.joiner = nn.Linear(enc_dim, vocab_size)

    def forward(self, enc_out: torch.Tensor, prefix: torch.Tensor):
        # enc_out: (B, T, enc_dim)
        # prefix:  (B, U, enc_dim) via embedding
        pred_emb = self.embedding(prefix)               # (B, U, enc_dim)
        # broadcast sum
        joint = enc_out.unsqueeze(2) + pred_emb.unsqueeze(1)  # (B, T, U, enc_dim)
        joint = torch.tanh(joint)
        logits = self.joiner(joint)                     # (B, T, U, V)
        return logits

def assert_all_detached(x):
    if isinstance(x, torch.Tensor):
        assert not x.requires_grad, "Tensor still requires grad"
    elif isinstance(x, (list, tuple)):
        for v in x:
            assert_all_detached(v)
    elif isinstance(x, dict):
        for v in x.values():
            assert_all_detached(v)

def detach_states(states):
    """Recursively detach all tensors in nested state structures (dicts, tuples, lists)."""
    if states is None:
        return None
    elif isinstance(states, torch.Tensor):
        return states.detach()
    elif isinstance(states, dict):
        return {k: detach_states(v) for k, v in states.items()}
    elif isinstance(states, tuple):
        return tuple(detach_states(s) for s in states)
    elif isinstance(states, list):
        return [detach_states(s) for s in states]
    else:
        # Catch any unexpected data type (e.g., numbers, strings)
        return states

def debug_print(debug, *args):
    """Helper function to print debug messages."""
    if debug:
        print("[DEBUG]", *args)

def setup_model_directory(args):
    """Create a directory for saving model checkpoints and logs."""
    timestamp = str(int(time.time()))
    model_dir = os.path.join("models", timestamp)
    os.makedirs(model_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(model_dir, "config.yaml"))
    return model_dir

def setup_logging(model_dir):
    """Configure logging to file and console."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(model_dir, "train.log"))
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

def setup_device():
    """Determine and set the device for training (GPU or CPU)."""
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    return device, device_str

def load_sentencepiece_model(args):
    """Load the SentencePiece model for tokenization."""
    sp = spm.SentencePieceProcessor()
    sp.load(args.sp_model)
    vocab_size = sp.get_piece_size()
    blank_id = 0
    return sp, vocab_size, blank_id

def build_model(args, device, feat_dim, vocab_size):
    """Build and initialize the ASR model."""
    encoder = build_encoder(args, vocab_size)
    model = ASRModel(
        frontend=make_frontend(args.frontend, args.batch_samplerate).to(device),
        encoder=encoder,
        vocab_size=vocab_size,
        feat_dim=feat_dim,
        proj_dim=args.embedding_dim,
        debug=args.debug
    ).to(device)
    return model

def setup_optimizer(args, model, joiner=None):
    """Set up the optimizer for training."""
    params = list(model.parameters())
    if joiner:
        params.append(joiner.parameters())

    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
    elif args.optimizer == "lion":
        from lion_pytorch import Lion
        optimizer = Lion(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            params,
            lr=args.lr,
        )
    return optimizer

def setup_criterion(args, blank_id):
    """Set up the loss function based on the training mode."""
    if args.mode == "ctc":
        criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    else:
        criterion = RNNTLoss(blank=blank_id)
    return criterion

def lr_lambda(step, warmup_steps, total_steps):
    """Learning rate scheduler lambda function."""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def setup_learning_rate_scheduler(optimizer, args):
    """Set up the learning rate scheduler."""
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_lambda(step, args.warmup_steps, args.total_steps))
    return scheduler

def setup_dataset(args):
    """Initialize and start the dataset session."""
    ds = dataset.SpeechDataset(
        config_path=args.config,
        verbose=args.verbose,
        debug_spectrograms=False,
        batch_segment_strategy=args.batch_segment_strategy,
        batch_samplerate=args.batch_samplerate
    )
    ds.start_session(
        batch_size=args.batch_size,
        order=args.order,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
    return ds

def process_batch_item(ds, item, target_samples):
    """Process a single batch item and handle any preprocessing errors."""
    try:
        audios, texts, masks = ds._load_and_preprocess_batch_item(item, target_samples)
        return audios, texts, masks
    except Exception as e:
        logging.getLogger("train").error(f"Data preprocess error: {e}; trying to leave out batch item!")
        return None

def prepare_batch_data(batch_audio_items, batch_texts_items, batch_masks_items, seg_idx, target_samples, device):
    """Prepare batch data for a specific segment index."""
    slice_audio, slice_masks, slice_texts = [], [], []
    for idx, (audios, texts, masks) in enumerate(zip(batch_audio_items, batch_texts_items, batch_masks_items)):
        if seg_idx < len(audios):
            slice_audio.append(audios[seg_idx])
            slice_masks.append(masks[seg_idx])
            slice_texts.append(texts[seg_idx])
        else:
            slice_audio.append(torch.zeros(target_samples, dtype=torch.float32))
            slice_masks.append(torch.zeros(target_samples, dtype=torch.bool))
            slice_texts.append("")

    batch_tensor = torch.stack(slice_audio).to(device)
    mask_tensor = torch.stack(slice_masks).to(device)
    return batch_tensor, mask_tensor, slice_texts

def prepare_tokens_and_lengths(slice_texts, sp, blank_id, device):
    """Prepare token IDs and lengths for the batch texts."""
    token_ids = [sp.encode(txt, out_type=int) for txt in slice_texts]
    tgt_lens = [len(t) for t in token_ids]
    max_tgt_len = max(tgt_lens)
    tokens = torch.full((len(token_ids), max_tgt_len), blank_id, dtype=torch.long, device=device)
    for i, t in enumerate(token_ids):
        if t:
            tokens[i, :len(t)] = torch.tensor(t, device=device)
    return tokens, tgt_lens

from jiwer import wer

def log_train_metrics(
    args, run, logger, losses, enc_out, tokens, tgt_lens, in_lens, sp, blank_id, global_step
):
    """Compute and log average loss and TER every 10 steps."""
    losses.append(float(losses[-1]))  # Make sure the current loss is included

    if len(losses) < 10:
        return losses  # Wait until we have 10 losses

    avg_loss = sum(losses) / len(losses)
    all_preds, all_refs = [], []
    debug_ref = debug_pred = None

    with torch.no_grad():
        top_tokens = enc_out.argmax(dim=-1)  # [B, T]
        for i in range(len(in_lens)):
            pred_ids = top_tokens[i, :in_lens[i]].tolist()
            pred_ids = [tid for tid in pred_ids if tid != blank_id]
            ref_ids = tokens[i, :tgt_lens[i]].tolist()

            pred_str = sp.decode_ids(pred_ids)
            ref_str = sp.decode_ids(ref_ids)

            pred_str_tok = " ".join(pred_str)
            ref_str_tok = " ".join(ref_str)

            all_preds.append(pred_str_tok)
            all_refs.append(ref_str_tok)

            if args.debug and i == 0:
                debug_ref = ref_str
                debug_pred = pred_str

    try:
        ter = wer(all_refs, all_preds)
    except Exception as e:
        logger.warning(f"Error computing TER: {e}")
        ter = -1.0

    if run is not None:
        run.track(avg_loss, name="avg_loss_10", step=global_step)
        run.track(ter, name="train_ter_10", step=global_step)

    logger.info(f"[Step {global_step}] Loss: {avg_loss:.4f}, Train TER: {ter:.4f}")

    if args.debug and debug_ref is not None:
        debug_print(args.debug, f"Reference [0]: {''.join(debug_ref)}")
        debug_print(args.debug, f"Predicted [0]: {''.join(debug_pred)}")

    return []


def train(args):
    """Main training function."""
    if HAVE_AIM:
        run = aim.Run(experiment="asr_statecatcher")
    else:
        run = None

    # Setup model directory and logging
    model_dir = setup_model_directory(args)
    logger = setup_logging(model_dir)
    logger.info(f"Model directory: {model_dir}")
    logger.info("Aim logging enabled." if HAVE_AIM else "Aim not available. Skipping experiment logging.")

    # Setup device for training
    device, device_str = setup_device()
    logger.info(f"Using device: {device}")

    # Load SentencePiece model
    sp, vocab_size, blank_id = load_sentencepiece_model(args)

    logger.info(f"Vocab size (output tokens): {vocab_size}")

    # Build frontend and infer feature dimension
    frontend = make_frontend(args.frontend, args.batch_samplerate).to(device)
    with torch.no_grad():
        dummy = torch.zeros(int(args.target_duration * args.batch_samplerate), device=device)
        dummy = dummy.unsqueeze(0).unsqueeze(1)
        feats = frontend(dummy)
        feats = feats.transpose(1, 2).contiguous()
        feat_dim = feats.shape[-1]

    # Build and initialize the model
    model = build_model(args, device, feat_dim, vocab_size)
    logger.info(f"Model built: {args.encoder}, feat_dim={feat_dim}, vocab_size={vocab_size}")

    if HAVE_AIM:
        run["hparams"] = {
            "encoder": args.encoder,
            "frontend": args.frontend,
            "mode": args.mode,
            "lr": args.lr,
            "batch_samplerate": args.batch_samplerate,
            "target_duration": args.target_duration,
            "optimizer": args.optimizer,
            "max_grad_norm": args.max_grad_norm,
            "epochs": args.epochs,
            "accumulation_steps": args.accumulation_steps
        }
        run["model/num_params"] = sum(p.numel() for p in model.parameters())
        run["tags"] = ['rnnt'] if args.mode == 'rnnt' else ['ctc']

    # Optionally build RNNT predictor+joiner
    joiner = None
    if args.mode == "rnnt":
        if not HAVE_RNNT:
            logger.error("warp_rnnt not available, cannot train RNN-T")
            return
        joiner = RNNTPredictorJoiner(model.enc_out_dim, vocab_size).to(device)
        logger.info("Initialized RNNT predictor+joiner")

    # Setup loss function
    criterion = setup_criterion(args, blank_id)
    logger.info(f"Using loss: {args.mode}")

    # Setup optimizer
    optimizer = setup_optimizer(args, model, joiner)
    logger.info(f"Optimizer: {args.optimizer}, lr={args.lr}")

    # Setup learning rate scheduler and gradient scaler
    scheduler = setup_learning_rate_scheduler(optimizer, args)
    scaler = GradScaler()

    # Initialize dataset session
    ds = setup_dataset(args)
    prev_epoch = None
    global_step = 0
    logger.info(f"Starting training for {args.epochs} epochs")
    sr = ds.batch_samplerate
    target_samples = int(sr * args.target_duration)
    losses = []

    # Main training loop
    while True:
        try:
            epoch, batch_id, batch = ds.fetch_next_batch()
        except Exception as e:
            logger.error(f"Data fetch error in fetch_next_batch(): {e}; sleeping for 10 seconds before retrying!")
            time.sleep(10)
            continue

        if prev_epoch is None:
            prev_epoch = epoch
        elif epoch != prev_epoch:
            ckpt = os.path.join(model_dir, f"model_epoch{prev_epoch+1}.pt")
            torch.save({
                "model": model.state_dict(),
                **({"joiner": joiner.state_dict()} if args.mode == "rnnt" else {})
            }, ckpt)
            logger.info(f"Saved checkpoint: {ckpt}")
            if prev_epoch + 1 >= args.epochs:
                break
            prev_epoch = epoch

        if epoch >= args.epochs:
            break

        batch_audio_items, batch_texts_items, batch_masks_items = [], [], []
        for item in batch:
            result = process_batch_item(ds, item, target_samples)
            if result:
                audios, texts, masks = result
                batch_audio_items.append(audios)
                batch_texts_items.append(texts)
                batch_masks_items.append(masks)

        if not batch_audio_items:
            logger.error("Batch is empty, probably due to previous errors. Retrying with a new batch after one second.")
            time.sleep(1)
            continue

        seg_counts = [len(segs) for segs in batch_audio_items]
        K = min(seg_counts) if ds.batch_segment_strategy == "clipping" else max(seg_counts)

        encoder_state = None
        for seg_idx in range(K):
            batch_tensor, mask_tensor, slice_texts = prepare_batch_data(
                batch_audio_items, batch_texts_items, batch_masks_items, seg_idx, target_samples, device
            )

            debug_print(args.debug, f"Batch shape: {tuple(batch_tensor.shape)}")
            debug_print(args.debug, f"Mask shape: {tuple(mask_tensor.shape)}")

            with torch.no_grad():
                feats = frontend(batch_tensor)
            feats = feats.transpose(1, 2).contiguous()

            tokens, tgt_lens = prepare_tokens_and_lengths(slice_texts, sp, blank_id, device)
            subsample = mask_tensor.size(1) // feats.size(1)
            in_lens = (mask_tensor.sum(dim=1) // subsample).clamp(max=feats.size(1)).long().tolist()

            debug_print(args.debug, f"Output {subsample=} and {in_lens=}")
            debug_print(args.debug, f"Tokens shape: {tuple(tokens.shape)}")
            debug_print(args.debug, f"Input lengths: {in_lens}")
            debug_print(args.debug, f"Target lengths: {tgt_lens}")

            if args.encoder == "lstm" and encoder_state is not None:
                input_state = encoder_state
            else:
                input_state = encoder_state

            with autocast(device_type=device_str, dtype=torch.float16):
                if input_state:
                    input_state = detach_states(input_state)
                    if args.debug:
                        assert_all_detached(input_state)
                enc_out, output_state = model(feats, input_state)
                logp = enc_out.log_softmax(-1).transpose(0, 1)
                loss = criterion(logp, tokens, in_lens, tgt_lens)
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if HAVE_AIM:
                run.track(loss.item(), name="loss", step=global_step)

            if (global_step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if HAVE_AIM:
                    grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    run.track(grad_norm ** 0.5, name="grad_norm", step=global_step)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            encoder_state = output_state
            global_step += 1

            # log train losses every 10 steps, compute TER 
            losses = log_train_metrics(args, run, logger, losses + [loss.item()], enc_out, tokens, tgt_lens, in_lens, sp, blank_id, global_step)

            if args.steps and global_step >= args.steps:
                break

        try:
            ds.mark_batch_done(epoch, batch_id)
        except Exception as e:
            logger.error(f"Problem with marking batch done: {e}; probably due to previous errors. Trying to fetch next batch.")
            continue

        ds.log("INFO", f"Completed batch {batch_id} @ epoch {epoch}")

        if args.steps and global_step >= args.steps:
            break

    ds.end_session()
    logger.info("Training complete.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Real ASR training loop")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sp-model", required=True, help="Path to SentencePiece model")
    parser.add_argument("--frontend", choices=["mfcc", "mel"], default="mfcc", help="Feature frontend")
    parser.add_argument("--encoder", choices=["lstm", "xlstm"], default="lstm")
    parser.add_argument("--batch-samplerate", type=int, default=16000)
    parser.add_argument("--batch-segment-strategy", choices=["clipping", "padding"], default="clipping")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--order", choices=["asc", "desc", "random"], default="asc")
    parser.add_argument("--min-duration", type=float, default=0.0)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--target-duration", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--mode", choices=["ctc", "rnnt"], default="ctc")
    parser.add_argument("--optimizer", choices=["adam", "adamw", "lion"], default="adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--chunkwise-kernel", type=str, default="chunkwise--native_autograd")
    parser.add_argument("--sequence-kernel", type=str, default="native_sequence__native")
    parser.add_argument("--step-kernel", type=str, default="native")
    # with triton:
    #parser.add_argument("--chunkwise-kernel", type=str, default="chunkwise--triton_xl_chunk")
    #parser.add_argument("--sequence-kernel", type=str, default="native_sequence__triton")
    #parser.add_argument("--step-kernel", type=str, default="triton")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    train(args)

