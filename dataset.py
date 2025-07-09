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

    def _plot_batch_waveforms(self, batch_audio, batch_texts, epoch, batch_id, seg_idx):
        """
        Plot a single “vertical slice” of waveforms (one segment index across all batch items),
        and save to: plots/batch{epoch}_batch{batch_id}_segment{seg_idx}.pdf
        """
        num_items = len(batch_audio)
        fig = plt.figure(figsize=(12, 2.5 * num_items))

        for i, (waveform, text) in enumerate(zip(batch_audio, batch_texts)):
            ax = fig.add_subplot(num_items, 1, i + 1)
            ax.plot(waveform.numpy())
            ax.set_xlim(0, len(waveform))
            ax.set_ylabel(f"Item {i + 1}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(text, fontsize=8, pad=2)

        plt.tight_layout()
        fname = f"plots/batch{epoch:04d}_batch{batch_id:05d}_segment{seg_idx:05d}.pdf"
        plt.savefig(fname)
        plt.close(fig)
        self._vprint(f"Saved plot to {fname}")

    
