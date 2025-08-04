#!/usr/bin/env python3
import argparse
import os
import shutil
import time
import math
import logging
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import torchaudio
import sentencepiece as spm
import multiprocessing
import dataset
from typing import Optional, Tuple, Any
from jiwer import wer
from model import make_frontend, ASRModel, build_encoder, RNNTPredictorJoiner, RNNTCompactPredictorJoiner, compute_loss
from decoder import ctc_greedy_decoder
from concurrent.futures import ProcessPoolExecutor, as_completed

_worker_ds = None

# for process pooling
def init_worker(args):
    global _worker_ds
    _worker_ds = dataset.SpeechDataset(
        config_path=args.config,
        verbose=args.verbose,
        debug_spectrograms=False,
        batch_segment_strategy=args.batch_segment_strategy,
        batch_samplerate=args.batch_samplerate
    )

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


def debug_print(debug, *args):
    """Helper function to print debug messages."""
    if debug:
        print("[DEBUG]", *args)

def setup_model_directory(args):
    """Create a directory for saving model checkpoints and logs."""
    timestamp = str(int(time.time()))
    model_dir = os.path.join("models", timestamp)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save command-line arguments to a JSON file
    args_dict = vars(args)
    args_json_path = os.path.join(model_dir, "training_args.json")
    with open(args_json_path, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

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
        proj_dim=args.input_proj_dim,
        debug=args.debug
    ).to(device)
    return model

def setup_optimizer(args, model, joiner=None):
    """Set up the optimizer for training."""
    params = list(model.parameters())
    if joiner:
        params += list(joiner.parameters())

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
        criterion = RNNTLoss
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
    global _worker_ds
    try:
        audios, texts, masks = _worker_ds.load_and_preprocess_batch_item(item, target_samples)
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

def log_train_metrics(
    args, run, logger, losses, enc_out, tokens, tgt_lens, in_lens, sp, blank_id, global_step
):
    losses.append(float(losses[-1]))
    if len(losses) < 10:
        return losses

    avg_loss = sum(losses) / len(losses)
    all_preds, all_refs = [], []
    debug_ref = debug_pred = None

    with torch.no_grad():
        # Step 1: Trim padded encoder output using in_lens
        trimmed = [enc_out[i, :in_lens[i]] for i in range(enc_out.size(0))]

        # Step 2: Pad back to a batch tensor and log_softmax
        log_probs = torch.nn.utils.rnn.pad_sequence(trimmed, batch_first=True)
        log_probs = log_probs.log_softmax(dim=-1)  # [B, T, V]

        # Step 3: Run CTC greedy decoding
        decoded = ctc_greedy_decoder(log_probs, torch.tensor(in_lens, device=enc_out.device), blank=blank_id)

        for i, pred_ids in enumerate(decoded):
            ref_ids = tokens[i, :tgt_lens[i]].tolist()

            pred_str = sp.decode_ids(pred_ids)
            ref_str = sp.decode_ids(ref_ids)

            all_preds.append(" ".join(pred_str))
            all_refs.append(" ".join(ref_str))

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

def save_checkpoint(model_dir, model, joiner, epoch, global_step=None, logger=None):
    """Save a model checkpoint."""
    if global_step is not None:
        ckpt_filename = f"model_epoch{epoch + 1}_step{global_step}.pt"
    else:
        ckpt_filename = f"model_epoch{epoch + 1}.pt"

    ckpt_path = os.path.join(model_dir, ckpt_filename)
    torch.save({
        "model": model.state_dict(),
        **({"joiner": joiner.state_dict()} if joiner is not None else {})
    }, ckpt_path)

    if logger is not None:
        logger.info(f"Saved checkpoint: {ckpt_path}")

    return ckpt_path

def process_batch_item_wrapper(args):
    global _worker_ds
    item, target_samples = args
    try:
        audios, texts, masks = _worker_ds.load_and_preprocess_batch_item(item, target_samples)
        return audios, texts, masks
    except Exception as e:
        logging.getLogger("train").error(f"Data preprocess error: {e}; trying to leave out batch item!")
        return None

def train(args, executer=None):
    """Main training function."""
    
    # Setup model directory and logging
    model_dir = setup_model_directory(args)
    model_dir_last_folder = model_dir.split('/')[-1]
    experiment_name = f"asr_statecatcher_{model_dir_last_folder}"

    if HAVE_AIM:
        run = aim.Run(experiment=experiment_name)
    else:
        run = None

    logger = setup_logging(model_dir)
    logger.info(f"Model directory: {model_dir}")
    logger.info("Aim logging enabled." if HAVE_AIM else "Aim not available. Skipping aim experiment logging."
                "You can activate aim by running 'pip3 install aim' followed by 'aim up'.")

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
    #model = torch.compile(model) #, fullgraph=True, mode="max-autotune")
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
        run["tags"] = ['rnnt'] if args.mode == 'rnnt' else ['ctc'] + [args.encoder]

    # Optionally build RNNT predictor+joiner
    joiner = None
    if args.mode == "rnnt":
        rnnt_pred_joiner_class = RNNTCompactPredictorJoiner if args.compact_rnnt else RNNTPredictorJoiner
        if not HAVE_RNNT:
            logger.error("warp_rnnt not available, cannot train RNN-T")
            return
        joiner = rnnt_pred_joiner_class(enc_out_dim=model.enc_out_dim, pred_emb_dim=args.rnnt_pred_emb_dim,
                                     join_dim=args.rnnt_joiner_dim, vocab_size=vocab_size).to(device)
        logger.info(f"Initialized RNNT predictor+joiner with {model.enc_out_dim=} and {vocab_size=}.")

    # Setup loss function
    criterion = setup_criterion(args, blank_id)
    logger.info(f"Using loss: {args.mode}")

    # Setup optimizer
    optimizer = setup_optimizer(args, model, joiner)
    logger.info(f"Optimizer: {args.optimizer}, lr={args.lr}")

    # Optional scheduler and scaler
    scheduler = setup_learning_rate_scheduler(optimizer, args) if args.use_scheduler else None
    scaler = GradScaler() if args.use_scaler else None

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
        t_begin = time.time()
        try:
            epoch, batch_id, batch = ds.fetch_next_batch()
        except Exception as e:
            logger.error(f"Data fetch error in fetch_next_batch(): {e}; sleeping for 10 seconds before retrying!")
            time.sleep(10)
            continue
        t_end = time.time()
        debug_print(args.debug, f"ds.fetch_next_batch() took {t_end - t_begin:.2f}s")

        # save model after every epoch
        if prev_epoch is None:
            prev_epoch = epoch
        elif epoch != prev_epoch:
            save_checkpoint(model_dir, model, joiner if args.mode == "rnnt" else None, prev_epoch, logger=logger)
            if prev_epoch + 1 >= args.epochs:
                break
            prev_epoch = epoch

        if epoch >= args.epochs:
            break

        t_begin = time.time()
        batch_audio_items, batch_texts_items, batch_masks_items = [], [], []

        if executor:
            futures = [executor.submit(process_batch_item_wrapper, (item, target_samples))
                        for item in batch]
            results = [f.result() for f in futures]
        else:
            results = [ds.load_and_preprocess_batch_item(item, target_samples) for item in batch]

        for result in results:
            if result:
                audio_np_list, text_list, mask_np_list = result

                audio_tensor_list = [torch.from_numpy(arr) for arr in audio_np_list]
                mask_tensor_list  = [torch.from_numpy(arr) for arr in mask_np_list]

                batch_audio_items.append(audio_tensor_list)
                batch_texts_items.append(text_list)
                batch_masks_items.append(mask_tensor_list)

        t_end = time.time()
        debug_print(args.debug, f"Process batch items took {t_end - t_begin:.2f}s")
       
        if not batch_audio_items:
            logger.error("Batch is empty, probably due to previous errors. Retrying with a new batch after one second.")
            time.sleep(1)
            continue

        seg_counts = [len(segs) for segs in batch_audio_items]
        K = min(seg_counts) if ds.batch_segment_strategy == "clipping" else max(seg_counts)

        debug_print(args.debug, f"Prepare batch with {K} segments. ds.batch_segment_strategy is {ds.batch_segment_strategy}.")

        encoder_state = None
        for seg_idx in range(K):
            t_begin = time.time() 
            batch_tensor, mask_tensor, slice_texts = prepare_batch_data(
                batch_audio_items, batch_texts_items, batch_masks_items, seg_idx, target_samples, device
            )
            t_end = time.time()
            debug_print(args.debug, f"Prepare_batch_data took {t_end - t_begin:.2f}s")

            debug_print(args.debug, f"Batch shape: {tuple(batch_tensor.shape)}")
            debug_print(args.debug, f"Mask shape: {tuple(mask_tensor.shape)}")

            t_begin = time.time()
            with torch.no_grad():
                feats = frontend(batch_tensor)
            feats = feats.transpose(1, 2).contiguous()
            t_end = time.time()
            debug_print(args.debug, f"Applying frontend took {t_end - t_begin:.2f}s")

            t_begin = time.time()
            tokens, tgt_lens = prepare_tokens_and_lengths(slice_texts, sp, blank_id, device)
            subsample = mask_tensor.size(1) // feats.size(1)
            t_end = time.time()
            debug_print(args.debug, f"Prepare_tokens_and_lengths took {t_end - t_begin:.2f}s")

            effective_stack = model.cfg.stack_order if model.cfg and hasattr(model.cfg, "stack_order") else 1
            subsample = (mask_tensor.size(1) // feats.size(1)) * effective_stack
            in_lens = (mask_tensor.sum(dim=1) // subsample).clamp(max=feats.size(1)).long().tolist()

            debug_print(args.debug, f"Subsample: {subsample}")

            in_lens = (mask_tensor.sum(dim=1) // subsample).clamp(max=feats.size(1)).long().tolist()

            for ilen, tlen in zip(in_lens, tgt_lens):
                if ilen < tlen:
                    logger.warning(f"Input length ({ilen}) < target length ({tlen}) — may cause CTC error")

            debug_print(args.debug, f"Output {subsample=} and {in_lens=}")
            debug_print(args.debug, f"Tokens shape: {tuple(tokens.shape)}")
            debug_print(args.debug, f"Input lengths: {in_lens}")
            debug_print(args.debug, f"Target lengths: {tgt_lens}")

            if args.encoder == "lstm" and encoder_state is not None:
                input_state = encoder_state
            else:
                input_state = encoder_state

            t_begin = time.time()
            # calculate loss (E.g. CTC or RNN-T)
            if args.use_scaler:
                with autocast(device_type=device_str, dtype=torch.float16):
                    loss, output_state, enc_out, output_state = compute_loss(
                        mode=args.mode, criterion=criterion, model=model,
                        feats=feats, tokens=tokens, in_lens=in_lens,
                        tgt_lens=tgt_lens, blank_id=blank_id,
                        use_rnnt_joiner=joiner if args.mode == "rnnt" else None,
                        input_state=input_state, args=args, compact=args.compact_rnnt)

                    loss = loss / args.accumulation_steps

                scaler.scale(loss).backward()

            else:
                loss, output_statie, enc_out, output_state = compute_loss(
                    mode=args.mode, criterion=criterion, model=model,
                    feats=feats, tokens=tokens, in_lens=in_lens,
                    tgt_lens=tgt_lens, blank_id=blank_id,
                    use_rnnt_joiner=joiner if args.mode == "rnnt" else None,
                    input_state=input_state, args=args)
                loss = loss / args.accumulation_steps
                loss.backward()
            t_end = time.time()

            debug_print(args.debug, f"Model and loss computation for one batch took {t_end - t_begin:.2f}s")

            t_begin = time.time()
            if HAVE_AIM:
                run.track(loss.item(), name="loss", step=global_step)
            t_end = time.time()

            debug_print(args.debug, f"Run AIM track took {t_end - t_begin:.2f}s")

            t_begin = time.time()
            if (global_step + 1) % args.accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if HAVE_AIM:
                    grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    run.track(grad_norm ** 0.5, name="grad_norm", step=global_step)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad()
            t_end = time.time()

            debug_print(args.debug, f"Optimizer step took {t_end - t_begin:.2f}s")

            # Save model every n updates
            if args.save_every_n_updates is not None and (global_step + 1) % args.save_every_n_updates == 0:
                save_checkpoint(model_dir, model, joiner if args.mode == "rnnt" else None, epoch, global_step + 1, logger)

            encoder_state = output_state
            global_step += 1

            t_begin = time.time()
            # log train losses every 10 steps, compute TER 
            losses = log_train_metrics(args, run, logger, losses + [loss.item()], enc_out, tokens, tgt_lens, in_lens, sp, blank_id, global_step)
            t_end = time.time()
            debug_print(args.debug, f"Log train metrics took {t_end - t_begin:.2f}s")

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
    parser.add_argument("--encoder", choices=["lstm", "xlstm", "lucyrnn"], default="lstm")
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
    parser.add_argument("--compact-rnnt", action="store_true")
    parser.add_argument("--optimizer", choices=["adam", "adamw", "lion"], default="adam")
    parser.add_argument("--use-scaler", action="store_true", help="Enable AMP gradient scaler (default: off)")
    parser.add_argument("--use-scheduler", action="store_true", help="Enable learning rate scheduler (default: off)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=50.0)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--rnnt-pred-emb-dim", type=int, default=64)
    parser.add_argument("--rnnt-joiner-dim", type=int, default=64)
    parser.add_argument("--input-proj-dim", type=int, default=-1)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--chunkwise-kernel", type=str, default="chunkwise--native_autograd")
    parser.add_argument("--sequence-kernel", type=str, default="native_sequence__native")
    parser.add_argument("--step-kernel", type=str, default="native")
    parser.add_argument("--save-every-n-updates", type=int, default=None, help="Save model every n updates")
    parser.add_argument("--num-workers", type=int, default=32,
        help="Number of parallel workers for batch item processing. Set to -1 to disable multiprocessing.")

    # with triton:
    #parser.add_argument("--chunkwise-kernel", type=str, default="chunkwise--triton_xl_chunk")
    #parser.add_argument("--sequence-kernel", type=str, default="native_sequence__triton")
    #parser.add_argument("--step-kernel", type=str, default="triton")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    ctx = multiprocessing.get_context("spawn") 
    executor = None
    if args.num_workers > 0:
        executor = ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=init_worker,
            initargs=(args,),
            mp_context=ctx
        )

    try:
        train(args, executor)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
