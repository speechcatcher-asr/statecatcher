#!/usr/bin/env python3
import argparse
import time

import torch
import dataset

def simulate_training_loop(ds, steps=None, sleep=1.0, target_duration=30.0):
    """
    Run a curriculum-style training loop over ds.fetch_next_batch(),
    segmenting each item into fixed-length chunks, aligning them
    vertically across the batch according to ds.batch_segment_strategy,
    and optionally plotting each segment-slice.
    """
    step = 0
    sr = ds.batch_samplerate
    target_samples = int(sr * target_duration)

    while True:
        # 1) fetch next batch
        try:
            epoch, batch_id, batch = ds.fetch_next_batch()
        except RuntimeError as e:
            print(f"[ERROR] Stopped training: {e}")
            break

        # 2) preprocess each item into (audios, texts, masks) lists
        batch_audio_items = []
        batch_texts_items = []
        batch_masks_items = []

        for item in batch:
            try:
                audios, texts, masks = ds.load_and_preprocess_batch_item(item, target_samples)
                ds._vprint(f"Item segments: {[a.shape for a in audios]}")
                batch_audio_items.append(audios)
                batch_texts_items.append(texts)
                batch_masks_items.append(masks)
            except Exception as e:
                ds._vprint(f"[ERROR] Skipping item: {e}")

        if not batch_audio_items:
            ds._vprint("[WARN] No valid items in batch. Skipping…")
            continue

        # 3) determine number of segment-slices K
        seg_counts = [len(segs) for segs in batch_audio_items]
        if ds.batch_segment_strategy == "clipping":
            K = min(seg_counts)
        else:  # padding
            K = max(seg_counts)

        # 4) iterate vertically over segment index
        for seg_idx in range(K):
            slice_audio = []
            slice_masks = []
            slice_texts = []

            for audios, texts, masks in zip(batch_audio_items,
                                            batch_texts_items,
                                            batch_masks_items):
                if seg_idx < len(audios):
                    slice_audio.append(audios[seg_idx])
                    slice_masks.append(masks[seg_idx])
                    slice_texts.append(texts[seg_idx])
                else:
                    # padding for missing segments
                    slice_audio.append(torch.zeros(target_samples, dtype=torch.float32))
                    slice_masks.append(torch.zeros(target_samples, dtype=torch.bool))
                    slice_texts.append("")

            # stack into a single mini-batch
            batch_tensor = torch.stack(slice_audio)    # (B, target_samples)
            mask_tensor  = torch.stack(slice_masks)    # (B, target_samples)

            # debug-plot if enabled
            if ds.debug_spectrograms:
                ds._plot_batch_waveforms(
                    slice_audio,
                    slice_texts,
                    epoch,
                    batch_id,
                    seg_idx
                )

            # (here your training step would consume batch_tensor & mask_tensor)
            time.sleep(sleep)

        # 5) mark batch done once all segment-slices are processed
        ds.mark_batch_done(epoch, batch_id)
        ds.log("INFO", f"Completed batch {batch_id} @ epoch {epoch}")

        step += 1
        if steps and step >= steps:
            break

    ds.end_session()
    ds._vprint("Training session ended.")


def main():
    parser = argparse.ArgumentParser(description="Simulated training client")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of items per server batch")
    parser.add_argument("--order", choices=["asc", "desc", "random"],
                        default="asc", help="Sampling order")
    parser.add_argument("--min_duration", type=float, default=0.0,
                        help="Minimum clip duration filter (seconds)")
    parser.add_argument("--max_duration", type=float, default=None,
                        help="Maximum clip duration filter (seconds)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Max number of batches to process")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Pause between segment-slices (seconds)")
    parser.add_argument("--target_duration", type=float, default=30.0,
                        help="Target segment length (seconds)")
    parser.add_argument("--batch-samplerate", type=int, default=16000,
                        help="Working sample rate for all audio resampling (Hz)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--debug-spectrograms", action="store_true",
                        help="Save waveform plots for debugging")
    parser.add_argument("--batch-segment-strategy",
                        choices=["clipping", "padding"],
                        default="clipping",
                        help="Align segments across batch: "
                             "'clipping' uses min count, "
                             "'padding' uses max and pads others")

    args = parser.parse_args()

    ds = dataset.SpeechDataset(
        config_path=args.config,
        verbose=args.verbose,
        debug_spectrograms=args.debug_spectrograms,
        batch_segment_strategy=args.batch_segment_strategy,
        batch_samplerate=args.batch_samplerate
    )

    ds.start_session(
        batch_size=args.batch_size,
        order=args.order,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )

    simulate_training_loop(
        ds,
        steps=args.steps,
        sleep=args.sleep,
        target_duration=args.target_duration
    )


if __name__ == "__main__":
    main()

