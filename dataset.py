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
    def __init__(self, config_path="config.yaml", verbose=False, debug_spectrograms=False):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.api_key = self.config["secret_api_key"]
        self.api_url = self.config["server_api_url"].rstrip("/")
        self.language = self.config.get("podcast_language", "en")
        self.session_id = None
        self.verbose = verbose
        self.debug_spectrograms = debug_spectrograms
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
                .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
                .run(input=audio_resp.content, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError("FFmpeg error occurred:\n" + e.stderr.decode("utf-8"))

        wav_data, _ = sf.read(io.BytesIO(out), dtype='int16')
        audio_float = wav_data.astype(np.float32) / 32767.0

        try:
            self._vprint(f"Downloading transcript from {transcript_url}")
            transcript_resp = requests.get(transcript_url, timeout=10)
            transcript_resp.raise_for_status()
            transcript_text = transcript_resp.text
        except Exception as e:
            raise RuntimeError(f"Failed to fetch transcript: {e}")

        segments = vtt_to_segments_with_text(transcript_text)
        print("segment:", segments)
        sr = 16000
        segment_tensors = []
        segment_texts = []

        for start, end, text in segments:
            duration = end - start
            if duration >= target_samples / sr:
                start_sample = int(start * sr)
                end_sample = start_sample + target_samples
                segment = audio_float[start_sample:end_sample]
                segment_tensors.append(torch.from_numpy(segment))
                segment_texts.append(text)

        if not segment_tensors:
            segment = np.zeros(target_samples, dtype=np.float32)
            seg_len = min(len(audio_float), len(segment))
            segment[:seg_len] = audio_float[:seg_len]
            segment_tensors.append(torch.from_numpy(segment))
            segment_texts.append("")  # Empty text for padding segment

        return segment_tensors, segment_texts

    def _plot_batch_waveforms(self, batch_audio, batch_texts, epoch, batch_id):
        num_items = len(batch_audio)
        self._vprint(f"Num items to plot in batch:", num_items)
        fig = plt.figure(figsize=(12, 2.5 * num_items))
        spec = gridspec.GridSpec(num_items, 1)
        for i, (waveform, text) in enumerate(zip(batch_audio, batch_texts)):
            ax = fig.add_subplot(spec[i])
            ax.plot(waveform.numpy())
            ax.set_xlim(0, len(waveform))
            ax.set_ylabel(f"Item {i + 1}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(text, fontsize=8, pad=2)
        plt.tight_layout()
        filename = f"plots/batch{epoch:04d}_segment{batch_id:05d}.pdf"
        plt.savefig(filename)
        plt.close()
        self._vprint(f"Saved plot to {filename}")

    def simulate_training_loop(self, steps=None, sleep=1.0, target_duration=30.0):
        step_count = 0
        sample_rate = 16000
        target_samples = int(sample_rate * target_duration)
        while True:
            try:
                epoch, batch_id, batch = self.fetch_next_batch()
            except RuntimeError as e:
                print(f"[ERROR] Stopped training: {e}")
                break

            self._vprint(f"Training on batch with offset {batch_id} from epoch {epoch} ({len(batch)} samples)...")
            batch_audio = []
            batch_texts = []
            for i, item in enumerate(batch):
                self._vprint(f"Processing item {i + 1}/{len(batch)}")
                try:
                    audio_tensors, texts = self._load_and_preprocess_batch_item(item, target_samples)
                    self._vprint(f"Audio tensor shapes: {[tensor.shape for tensor in audio_tensors]}, Texts: {texts}")
                    batch_audio.extend(audio_tensors)
                    batch_texts.extend(texts)
                except Exception as e:
                    self._vprint(f"[ERROR] Failed to process item: {e}")
                    continue

            if not batch_audio:
                self._vprint("[WARN] No valid items in batch. Skipping...")
                continue

            batch_tensor = torch.stack(batch_audio)
            if self.debug_spectrograms:
                self._plot_batch_waveforms(batch_audio, batch_texts, epoch, batch_id)

            self._vprint(f"Processed batch size: {batch_tensor.shape}")
            time.sleep(sleep)
            self.mark_batch_done(epoch, batch_id)
            self.log("INFO", f"Completed batch with offset {batch_id} in epoch {epoch}")
            step_count += 1
            if steps and step_count >= steps:
                break

        self.end_session()
        self._vprint("Training session ended.")

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

