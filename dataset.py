# dataset.py
import argparse
import yaml
import requests
import time
import io
import os
import numpy as np
import torch
import ffmpeg
import soundfile as sf
from matplotlib import pyplot as plt
from matplotlib import gridspec
from pathlib import Path
from parse_vtts import vtt_to_segments_with_text, parse_timestamp

class SpeechDataset:
    def __init__(self, config_path="config.yaml", verbose=False, debug_spectrograms=False, batch_segment_strategy="clipping"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.api_key = self.config["secret_api_key"]
        self.api_url = self.config["server_api_url"].rstrip("/")
        self.language = self.config.get("podcast_language", "en")
        self.session_id = None
        self.verbose = verbose
        self.debug_spectrograms = debug_spectrograms
        self.batch_segment_strategy = batch_segment_strategy  # 'clipping' or 'padding'

        if self.debug_spectrograms:
            Path("plots").mkdir(parents=True, exist_ok=True)

    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print("[INFO]", *args, **kwargs)

    def start_session(self, batch_size=8, order="asc", min_duration=0.0, max_duration=None):
        url = f"{self.api_url}/start_training_session/{self.api_key}"
        payload = {
            "language": self.language,
            "batch_size": batch_size,
            "order": order,
            "min_duration": min_duration,
            "max_duration": max_duration,
        }
        self._vprint(f"Starting training session with batch_size={batch_size}, order={order}, min_dur={min_duration}, max_dur={max_duration}")
        response = requests.post(url, json=payload)
        if response.ok:
            result = response.json()
            if result["success"]:
                self.session_id = result["session_id"]
                self._vprint(f"Started session {self.session_id}")
            else:
                raise RuntimeError(f"Failed to start session: {result['error']}")
        else:
            raise RuntimeError(f"Failed to connect to server: {response.status_code}")

    def fetch_next_batch(self):
        url = f"{self.api_url}/get_next_batch/{self.session_id}/{self.api_key}"
        self._vprint("Fetching next batch...")
        try:
            response = requests.get(url, timeout=10)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request error while fetching batch: {e}")
        if not response.ok:
            try:
                error_detail = response.json()
            except Exception:
                error_detail = response.text
            raise RuntimeError(f"API request failed with status {response.status_code}: {error_detail}")
        result = response.json()
        if not result.get("success"):
            raise RuntimeError(f"Error fetching batch: {result.get('error', 'Unknown error')}")
        return result["epoch"], result["batch_id"], result["batch"]

    def mark_batch_done(self, epoch, batch_id):
        url = f"{self.api_url}/mark_batch_done/{self.session_id}/{batch_id}/{self.api_key}?epoch={epoch}"
        self._vprint(f"Marking batch {batch_id} from epoch {epoch} as done...")
        try:
            response = requests.post(url)
        except Exception as e:
            print(f"[WARN] Network error during mark_batch_done: {e}")
            return
        if not response.ok:
            print(f"[WARN] Failed to mark batch done. HTTP {response.status_code}")
            try:
                print(f"[WARN] Response: {response.json()}")
            except Exception:
                print(f"[WARN] Non-JSON response: {response.text}")
            return
        result = response.json()
        if not result.get("success"):
            print(f"[WARN] API error marking batch done: {result.get('error', 'Unknown error')}")

    def log(self, level, message):
        url = f"{self.api_url}/log/{self.session_id}/{self.api_key}"
        self._vprint(f"Logging: [{level}] {message}")
        try:
            requests.post(url, json={"level": level, "message": message})
        except Exception:
            pass  # Non-fatal

    def end_session(self):
        url = f"{self.api_url}/end_training_session/{self.session_id}/{self.api_key}"
        self._vprint("Ending training session...")
        requests.post(url)


    def _load_and_preprocess_batch_item(self, item, target_samples):
        """Download one audio+VTT pair, split into fixed-length chunks, pad/trim,
        and return (audio_tensors, texts, masks)."""
        # 1) Download & decode audio
        audio_url = item["cache_audio_url"]
        transcript_url = item["transcript_file"].replace('/var/www/', 'https://')

        try:
            self._vprint(f"Downloading audio from {audio_url}")
            audio_resp = requests.get(audio_url, timeout=10)
            audio_resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to download audio: {e}")

        try:
            out, _ = (
                ffmpeg.input("pipe:0")
                .output("pipe:", format="wav", acodec="pcm_s16le", ac=1, ar="16000")
                .run(input=audio_resp.content, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError("FFmpeg error occurred:\n" + e.stderr.decode("utf-8"))

        wav_data, _ = sf.read(io.BytesIO(out), dtype="int16")
        audio_float = wav_data.astype(np.float32) / 32767.0
        sr = 16000
        total_duration = len(audio_float) / sr
        window_sec = target_samples / sr

        # 2) Download & parse VTT
        try:
            self._vprint(f"Downloading transcript from {transcript_url}")
            tr_resp = requests.get(transcript_url, timeout=10)
            tr_resp.raise_for_status()
            segments = vtt_to_segments_with_text(tr_resp.text)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch/parse transcript: {e}")

        # 3) Group VTT cues into “chunks” near target_duration
        chunks = []
        cur = []            # list of (start,end,text)
        for (start, end, text) in segments:
            if not cur:
                # start a fresh chunk
                cur = [(start, end, text)]
                dur = end - start
            else:
                prev_start = cur[0][0]
                prev_end = cur[-1][1]
                undershoot = prev_end - prev_start
                overshoot = end - prev_start

                if overshoot < window_sec:
                    # still under target -> just append
                    cur.append((start, end, text))
                else:
                    # we would cross the target if we add this cue:
                    # decide which is closer
                    if abs(overshoot - window_sec) < abs(window_sec - undershoot):
                        # include this cue
                        cur.append((start, end, text))
                        prev_end = end
                    # finalize this chunk
                    chunks.append((prev_start, prev_end, [t for _, _, t in cur]))
                    # start next chunk
                    cur = [(start, end, text)]
        # whatever remains
        if cur:
            s0, e0, _ = cur[0]
            chunks.append((s0, cur[-1][1], [t for _, _, t in cur]))

        # 4) Convert each chunk into fixed-size audio + mask + joined text
        segment_tensors = []
        segment_texts = []
        segment_masks = []

        for (c_start, c_end, texts) in chunks:
            s_samp = int(c_start * sr)
            e_samp = int(c_end * sr)
            seg = audio_float[s_samp:e_samp]
            real_len = len(seg)

            if real_len >= target_samples:
                # too long -> trim
                seg = seg[:target_samples]
                mask = torch.ones(target_samples, dtype=torch.bool)
            else:
                # too short -> pad zeros
                pad = target_samples - real_len
                pad_arr = np.zeros(pad, dtype=np.float32)
                seg = np.concatenate([seg, pad_arr], axis=0)
                mask = torch.cat([
                    torch.ones(real_len, dtype=torch.bool),
                    torch.zeros(pad,    dtype=torch.bool)
                ], dim=0)

            segment_tensors.append(torch.from_numpy(seg))
            segment_masks.append(mask)
            segment_texts.append(" ".join(texts))

        # 5) If *no* chunks found (e.g. totally empty VTT), fall back to zero-pad + empty text
        if not segment_tensors:
            seg = audio_float
            real_len = min(len(seg), target_samples)
            pad_len = target_samples - real_len
            seg = np.concatenate([seg[:real_len], np.zeros(pad_len, dtype=np.float32)])
            mask = torch.cat([torch.ones(real_len, dtype=torch.bool),
                              torch.zeros(pad_len,  dtype=torch.bool)])
            segment_tensors = [torch.from_numpy(seg)]
            segment_masks   = [mask]
            segment_texts   = [""]

        return segment_tensors, segment_texts, segment_masks


    def _plot_batch_waveforms(self, batch_audio_items, batch_texts_items, epoch, batch_id):
        num_items = len(batch_audio_items)
        counts = [len(seq) for seq in batch_audio_items]
        if self.batch_segment_strategy == "clipping":
            K = min(counts)
        else:
            K = max(counts)
        for seg_idx in range(K):
            fig = plt.figure(figsize=(12, 2.5 * num_items))
            for i in range(num_items):
                seq, texts = batch_audio_items[i], batch_texts_items[i]
                if seg_idx < len(seq):
                    wave, title = seq[seg_idx], texts[seg_idx]
                else:
                    wave = torch.zeros_like(batch_audio_items[i][0])
                    title = ""
                ax = fig.add_subplot(num_items, 1, i+1)
                ax.plot(wave.numpy())
                ax.set_xlim(0, len(wave))
                ax.set_ylabel(f"Item {i+1}")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(title, fontsize=8, pad=2)
            plt.tight_layout()
            fname = f"plots/batch{epoch:04d}_batch{batch_id:05d}_segment{seg_idx:05d}.pdf"
            plt.savefig(fname); plt.close(fig)
            self._vprint(f"Saved plot to {fname}")

    def simulate_training_loop(self, steps=None, sleep=1.0, target_duration=30.0):
        step = 0; sr = 16000; target = int(sr * target_duration)
        while True:
            try:
                epoch, batch_id, batch = self.fetch_next_batch()
            except RuntimeError as e:
                print(f"[ERROR] Stopped training: {e}"); break

            batch_audio_items, batch_texts_items, batch_masks_items = [], [], []
            for itm in batch:
                try:
                    audios, texts, masks = self._load_and_preprocess_batch_item(itm, target)
                    batch_audio_items.append(audios)
                    batch_texts_items.append(texts)
                    batch_masks_items.append(masks)
                    self._vprint(f"Loaded item → segments: {len(audios)}")
                except Exception as e:
                    self._vprint(f"[ERROR] Skipping item: {e}")
            if not batch_audio_items:
                self._vprint("[WARN] No valid items. Skipping batch...")
                continue

            counts = [len(seq) for seq in batch_audio_items]
            K = min(counts) if self.batch_segment_strategy=="clipping" else max(counts)

            for seg_idx in range(K):
                mini_audio, mini_mask = [], []
                for i in range(len(batch_audio_items)):
                    seq = batch_audio_items[i]
                    mseq = batch_masks_items[i]
                    if seg_idx < len(seq):
                        mini_audio.append(seq[seg_idx])
                        mini_mask.append(mseq[seg_idx])
                    else:
                        zeros = torch.zeros(target, dtype=torch.float32)
                        mini_audio.append(zeros)
                        mini_mask.append(torch.zeros(target, dtype=torch.bool))
                batch_tensor = torch.stack(mini_audio)
                mask_tensor  = torch.stack(mini_mask)

                if self.debug_spectrograms:
                    self._plot_batch_waveforms(batch_audio_items, batch_texts_items, epoch, batch_id)

                self._vprint(f"Simulated train: segment {seg_idx+1}/{K}, batch size {batch_tensor.shape}")
                time.sleep(sleep)

            self.mark_batch_done(epoch, batch_id)
            self.log("INFO", f"Completed batch {batch_id} @ epoch {epoch}")
            step += 1
            if steps and step >= steps:
                break

        self.end_session()

  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated training client for speech data server")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--order", choices=["asc", "desc", "random"], default="asc", help="Sample order")
    parser.add_argument("--min_duration", type=float, default=0.0, help="Minimum duration filter")
    parser.add_argument("--max_duration", type=float, default=None, help="Maximum duration filter")
    parser.add_argument("--steps", type=int, default=None, help="Max number of batches to process")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep time per batch (simulate training)")
    parser.add_argument("--target_duration", type=float, default=30.0, help="Target duration in seconds per sample")
    parser.add_argument("--verbose", action="store_true", help="Print detailed debug output")
    parser.add_argument("--debug-spectrograms", action="store_true", help="Save waveform plots for debugging")

    args = parser.parse_args()
    dataset = SpeechDataset(
        config_path=args.config,
        verbose=args.verbose,
        debug_spectrograms=args.debug_spectrograms
    )
    dataset.start_session(
        batch_size=args.batch_size,
        order=args.order,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
    dataset.simulate_training_loop(
        steps=args.steps,
        sleep=args.sleep,
        target_duration=args.target_duration
    )
