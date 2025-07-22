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
from model import make_frontend, ASRModel, build_encoder

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
        # Dummy input should match training: (1,1,T)
        dummy = torch.zeros(int(args.target_duration * args.batch_samplerate),
                                 device=device)
        dummy = dummy.unsqueeze(0).unsqueeze(1)
        feats = frontend(dummy)
        # Now feats is (1, C, T) or (1, F, T); transpose to (B, T, F)
        feats = feats.transpose(1, 2).contiguous()
        feat_dim = feats.shape[-1]

    # build encoder
    encoder = build_encoder(args, vocab_size)
    # assemble the ASR model
    model = ASRModel(
        frontend=frontend,
        encoder=encoder,
        vocab_size=vocab_size,
        feat_dim=feat_dim,
        proj_dim=args.embedding_dim,
        debug=args.debug
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
    global_step = 0
    logger.info(f"Starting training for {args.epochs} epochs")

    sr = ds.batch_samplerate
    target_samples = int(sr * args.target_duration)

    prev_epoch = None
    global_step = 0
    logger.info(f"Starting training for {args.epochs} epochs")

    sr = ds.batch_samplerate
    target_samples = int(sr * args.target_duration)

    while True:
        try:
            epoch, batch_id, batch = ds.fetch_next_batch()
        except RuntimeError as e:
            logger.error(f"Data fetch error: {e}; stopping.")
            break

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
            audios, texts, masks = ds._load_and_preprocess_batch_item(item, target_samples)
            batch_audio_items.append(audios)
            batch_texts_items.append(texts)
            batch_masks_items.append(masks)

        seg_counts = [len(segs) for segs in batch_audio_items]
        K = min(seg_counts) if ds.batch_segment_strategy == "clipping" else max(seg_counts)

        # Clean state handling: single encoder_state, type depends on encoder
        encoder_state = None

        for seg_idx in range(K):
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

            if args.debug:
                print(f"[DEBUG] Batch shape: {tuple(batch_tensor.shape)}")
                print(f"[DEBUG] Mask shape: {tuple(mask_tensor.shape)}")

            with torch.no_grad():
                feats = frontend(batch_tensor)
            feats = feats.transpose(1, 2).contiguous()

            token_ids = [sp.encode(txt, out_type=int) for txt in slice_texts]
            tgt_lens = [len(t) for t in token_ids]
            max_tgt_len = max(tgt_lens)

            tokens = torch.full((len(token_ids), max_tgt_len), blank_id, dtype=torch.long, device=device)
            for i, t in enumerate(token_ids):
                if t:
                    tokens[i, :len(t)] = torch.tensor(t, device=device)

            subsample = mask_tensor.size(1) // feats.size(1)
            in_lens = (mask_tensor.sum(dim=1) // subsample).long().tolist()

            if args.debug:
                print(f"[DEBUG] Tokens shape: {tuple(tokens.shape)}")
                print(f"[DEBUG] Input lengths: {in_lens}")
                print(f"[DEBUG] Target lengths: {tgt_lens}")

            # Prepare input state for model
            if args.encoder == "lstm" and encoder_state is not None:
                # already in (h, c) form — use as is
                input_state = encoder_state
            else:
                input_state = encoder_state  # None or list (xLSTM)

            with autocast():
                enc_out, output_state = model(feats, input_state)

                logp = enc_out.log_softmax(-1).transpose(0, 1)
                loss = criterion(logp, tokens, in_lens, tgt_lens)
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (global_step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            # Save detached states for next segment
            encoder_state = detach_states(output_state)

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
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--chunkwise-kernel", type=str,
                        default="chunkwise--triton_xl_chunk")
    parser.add_argument("--sequence-kernel", type=str,
                        default="native_sequence__triton")
    parser.add_argument("--step-kernel", type=str, default="triton")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    train(args)

