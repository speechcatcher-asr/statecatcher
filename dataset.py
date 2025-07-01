import argparse
import yaml
import requests
import time
import io
import numpy as np
import torch
import ffmpeg
import soundfile as sf
import re


class SpeechDataset:
    def __init__(self, config_path="config.yaml", verbose=False):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.api_key = self.config["secret_api_key"]
        self.api_url = self.config["server_api_url"].rstrip("/")
        self.language = self.config.get("podcast_language", "en")
        self.session_id = None
        self.verbose = verbose

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
        response = requests.post(url)

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
        requests.post(url, json={"level": level, "message": message})

    def end_session(self):
        url = f"{self.api_url}/end_training_session/{self.session_id}/{self.api_key}"
        self._vprint("Ending training session...")
        requests.post(url)

    def _parse_vtt_segments(self, vtt_text):
        pattern = re.compile(r"(\d+):(\d+\.\d+)\s*-->\s*(\d+):(\d+\.\d+)")
        segments = []
        for line in vtt_text.splitlines():
            match = pattern.match(line)
            if match:
                start = int(match.group(1)) * 60 + float(match.group(2))
                end = int(match.group(3)) * 60 + float(match.group(4))
                if end > start:
                    segments.append((start, end))
        return segments

    def _extract_window(self, waveform, sr, segments, target_seconds):
        target_len = int(target_seconds * sr)
        best_chunk = None
        best_len = 0

        for start, end in segments:
            start_i = int(start * sr)
            end_i = int(end * sr)
            if end_i <= len(waveform):
                length = end_i - start_i
                if best_len < length <= target_len:
                    best_chunk = waveform[start_i:end_i]
                    best_len = length
                    if best_len == target_len:
                        break

        if best_chunk is None:
            return None, None

        pad_len = target_len - len(best_chunk)
        padded = np.pad(best_chunk, (0, pad_len), constant_values=0.0)
        mask = np.zeros(target_len, dtype=np.float32)
        mask[:len(best_chunk)] = 1.0
        return padded, mask

    def _load_and_preprocess_batch_item(self, item, target_duration=30.0):
        audio_url = item["cache_audio_url"]
        transcript_url = item["transcript_file"].replace('/var/www/', 'https://')

        self._vprint(f"Downloading audio from {audio_url}")
        audio_resp = requests.get(audio_url, timeout=10)
        audio_resp.raise_for_status()

        self._vprint(f"Downloading transcript from {transcript_url}")
        transcript_resp = requests.get(transcript_url, timeout=10)
        transcript_resp.raise_for_status()
        transcript_text = transcript_resp.text
        segments = self._parse_vtt_segments(transcript_text)

        try:
            out, _ = (
                ffmpeg.input("pipe:0")
                .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
                .run(input=audio_resp.content, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print("FFmpeg error occurred:")
            print(e.stderr.decode('utf-8'))
            raise

        waveform, sr = sf.read(io.BytesIO(out), dtype='float32')
        assert sr == 16000

        sample, mask = self._extract_window(waveform, sr, segments, target_duration)
        if sample is None:
            raise ValueError("No suitable segment found for target duration")

        return torch.from_numpy(sample), torch.from_numpy(mask)

    def simulate_training_loop(self, steps=None, sleep=1.0, target_duration=30.0):
        step_count = 0
        while True:
            try:
                epoch, batch_id, batch = self.fetch_next_batch()
            except RuntimeError as e:
                print(f"[ERROR] Stopped training: {e}")
                break

            self._vprint(f"Training on batch with offset {batch_id} from epoch {epoch} ({len(batch)} samples)...")

            for i, item in enumerate(batch):
                self._vprint(f"Processing item {i + 1}/{len(batch)}")
                try:
                    audio_tensor, mask = self._load_and_preprocess_batch_item(item, target_duration=target_duration)
                    self._vprint(f"Audio shape: {audio_tensor.shape}, Mask shape: {mask.shape}")
                    # Use audio_tensor and mask in training loop
                except Exception as e:
                    self._vprint(f"[ERROR] Failed to process item: {e}")
                    continue

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
    parser.add_argument("--target_duration", type=float, default=30.0, help="Target audio segment duration in seconds")
    parser.add_argument("--verbose", action="store_true", help="Print detailed debug output")
    args = parser.parse_args()

    dataset = SpeechDataset(config_path=args.config, verbose=args.verbose)
    dataset.start_session(
        batch_size=args.batch_size,
        order=args.order,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
    dataset.simulate_training_loop(steps=args.steps, sleep=args.sleep, target_duration=args.target_duration)

