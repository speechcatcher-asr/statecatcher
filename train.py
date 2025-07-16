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
from torch.cuda.amp import autocast, GradScaler
import torchaudio
import sentencepiece as spm

import dataset
from model import make_frontend, ASRModel

# Try to import RNN-T loss; if unavailable, only CTC will work
try:
    from warp_rnnt import RNNTLoss
    HAVE_RNNT = True
except ImportError:
    HAVE_RNNT = False


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


def detach_states(states):
    if states is None:
        return None
    if isinstance(states, tuple):
        return tuple(s.detach() for s in states)
    else:
        return states.detach()


def train(args):
    # create model directory with timestamp
    timestamp = str(int(time.time()))
    model_dir = os.path.join("models", timestamp)
    os.makedirs(model_dir, exist_ok=True)

    # copy config
    shutil.copy(args.config, os.path.join(model_dir, "config.yaml"))

    # set up logging to file and stdout
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
    logger.info(f"Model directory: {model_dir}")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # SentencePiece tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(args.sp_model)
    vocab_size = sp.get_piece_size()
    blank_id = 0

    # frontend & infer feature dim
    frontend = make_frontend(args.frontend, args.batch_samplerate).to(device)
    with torch.no_grad():
        dummy = torch.zeros(1, int(args.target_duration * args.batch_samplerate),
                              device=device)
        # if your frontend expects a channel dimension, keep the unsqueeze:
        feats  = frontend(dummy.unsqueeze(0))
        feat_dim = feats.shape[1]  # (B, F, T)

    # build encoder model
    enc_cfg = {
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        **({
            "embedding_dim": args.embedding_dim,
            "num_heads": args.num_heads,
            "num_blocks": args.num_blocks,
            "vocab_size": vocab_size,
            "return_last_states": True,
            "mode": "train",
            "chunkwise_kernel": args.chunkwise_kernel,
            "sequence_kernel": args.sequence_kernel,
            "step_kernel": args.step_kernel,
        } if args.encoder == "xlstm" else {})
    }
    model = ASRModel(
        frontend=frontend,
        encoder_type=args.encoder,
        encoder_cfg=enc_cfg,
        vocab_size=vocab_size,
        feat_dim=feat_dim
    ).to(device)
    logger.info(f"Model built: {args.encoder}, feat_dim={feat_dim}, vocab_size={vocab_size}")

    # optionally build RNNT predictor+joiner
    if args.mode == "rnnt":
        if not HAVE_RNNT:
            logger.error("warp_rnnt not available, cannot train RNN-T")
            return
        joiner = RNNTPredictorJoiner(model.enc_out_dim, vocab_size).to(device)
        logger.info("Initialized RNNT predictor+joiner")

    # loss
    if args.mode == "ctc":
        criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    else:
        criterion = RNNTLoss(blank=blank_id)
    logger.info(f"Using loss: {args.mode}")

    # optimizer
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            list(model.parameters()) + ([joiner.parameters()] if args.mode=="rnnt" else []),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
    elif args.optimizer == "lion":
        from lion_pytorch import Lion
        optimizer = Lion(
            list(model.parameters()) + ([joiner.parameters()] if args.mode=="rnnt" else []),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            list(model.parameters()) + ([joiner.parameters()] if args.mode=="rnnt" else []),
            lr=args.lr,
        )
    logger.info(f"Optimizer: {args.optimizer}, lr={args.lr}")

    # LR scheduling
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, args.total_steps - args.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    # dataset session
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

    prev_epoch = None
    state = None
    global_step = 0

    logger.info(f"Starting training for {args.epochs} epochs")
    while True:
        try:
            epoch, batch_id, batch = ds.fetch_next_batch()
        except RuntimeError as e:
            logger.error(f"Data fetch error: {e}; stopping.")
            break

        # epoch boundary: save checkpoint
        if prev_epoch is None:
            prev_epoch = epoch
        elif epoch != prev_epoch:
            ckpt = os.path.join(model_dir, f"model_epoch{prev_epoch+1}.pt")
            torch.save({
                "model": model.state_dict(),
                **({"joiner": joiner.state_dict()} if args.mode=="rnnt" else {})
            }, ckpt)
            logger.info(f"Saved checkpoint: {ckpt}")
            if prev_epoch + 1 >= args.epochs:
                break
            prev_epoch = epoch
        if epoch >= args.epochs:
            break

        # load & chunk
        items = []
        for itm in batch:
            auds, txts, masks = ds._load_and_preprocess_batch_item(
                itm, int(args.batch_samplerate * args.target_duration)
            )
            items.append((auds, txts, masks))

        seg_counts = [len(aud) for aud, _, _ in items]
        K = min(seg_counts) if args.batch_segment_strategy == "clipping" else max(seg_counts)
        state = None  # reset state at epoch start

        for seg_idx in range(K):
            # build slice
            slice_audio = []; slice_masks = []; slice_texts = []
            for auds, txts, masks in items:
                if seg_idx < len(auds):
                    slice_audio.append(auds[seg_idx])
                    slice_masks.append(masks[seg_idx])
                    slice_texts.append(txts[seg_idx])
                else:
                    slice_audio.append(torch.zeros_like(auds[0]))
                    slice_masks.append(torch.zeros_like(masks[0]))
                    slice_texts.append("")

            # to device
            batch_audio = torch.stack(slice_audio).to(device)
            batch_masks = torch.stack(slice_masks).to(device)

            # features
            with torch.no_grad():
                feats = frontend(batch_audio.unsqueeze(1))
            feats = feats.transpose(1, 2).contiguous()  # (B, T, F)

            # tokenize & prepare labels
            token_ids = [sp.encode(s, out_type=int) for s in slice_texts]
            tgt_lens  = [len(t) for t in token_ids]
            max_tgt   = max(tgt_lens, default=0)
            tokens = torch.full(
                (len(token_ids), max_tgt),
                blank_id, dtype=torch.long, device=device
            )
            for i, t in enumerate(token_ids):
                if t:
                    tokens[i, :len(t)] = torch.tensor(t, device=device)

            # input lengths
            subsample = batch_masks.size(1) // feats.size(1)
            in_lens = (batch_masks.sum(dim=1) // subsample).long().tolist()

            # forward+loss
            with autocast():
                enc_out, new_state = model(feats, state)  # enc_out: (B, T, E)
                if args.mode == "ctc":
                    logp = enc_out.log_softmax(-1).transpose(0, 1)  # (T, B, V)
                    loss = criterion(logp, tokens, in_lens, tgt_lens)
                else:
                    # build RNNT prefixes: blank + labels
                    prefix_len = max_tgt + 1
                    prefix = torch.full(
                        (len(token_ids), prefix_len),
                        blank_id, dtype=torch.long, device=device
                    )
                    for i, t in enumerate(token_ids):
                        prefix[i, 1:1+len(t)] = torch.tensor(t, device=device)

                    joint_logits = joiner(enc_out, prefix)           # (B, T, U, V)
                    logp = joint_logits.log_softmax(-1)
                    loss = criterion(
                        logp,
                        tokens,           # labels shape should be (B, U-1)
                        torch.tensor(in_lens, dtype=torch.int32, device=device),
                        torch.tensor(tgt_lens, dtype=torch.int32, device=device)
                    )
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (global_step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            state = detach_states(new_state)
            global_step += 1

            if args.verbose:
                logger.info(f"[Step {global_step}] Loss: {loss.item():.4f}")

            if args.steps and global_step >= args.steps:
                break

        ds.mark_batch_done(epoch, batch_id)
        ds.log("INFO", f"Completed batch {batch_id} @ epoch {epoch}")
        if args.steps and global_step >= args.steps:
            break

    ds.end_session()
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real ASR training loop")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sp-model", required=True,
                        help="Path to SentencePiece model")
    parser.add_argument("--frontend", choices=["mfcc", "mel"],
                        default="mfcc", help="Feature frontend")
    parser.add_argument("--encoder", choices=["lstm", "xlstm"],
                        default="lstm")
    parser.add_argument("--batch-samplerate", type=int,
                        default=16000)
    parser.add_argument("--batch-segment-strategy",
                        choices=["clipping", "padding"],
                        default="clipping")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--order", choices=["asc", "desc", "random"],
                        default="asc")
    parser.add_argument("--min-duration", type=float, default=0.0)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--target-duration", type=float, default=30.0)
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--mode", choices=["ctc", "rnnt"],
                        default="ctc")
    parser.add_argument("--optimizer", choices=["adam", "adamw", "lion"],
                        default="adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--chunkwise-kernel", type=str,
                        default="chunkwise--triton_xl_chunk")
    parser.add_argument("--sequence-kernel", type=str,
                        default="native_sequence__triton")
    parser.add_argument("--step-kernel", type=str, default="triton")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    train(args)

