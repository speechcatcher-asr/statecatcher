import argparse
import yaml
import requests
import os
import time

class SpeechDataset:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.api_key = self.config["secret_api_key"]
        self.api_url = self.config["server_api_url"].rstrip("/")
        self.language = self.config.get("podcast_language", "en")

        self.session_id = None

    def start_session(self, batch_size=8, order="asc", min_duration=0.0, max_duration=None):
        url = f"{self.api_url}/start_training_session/{self.api_key}"
        payload = {
            "language": self.language,
            "batch_size": batch_size,
            "order": order,
            "min_duration": min_duration,
            "max_duration": max_duration,
        }

        response = requests.post(url, json=payload)
        if response.ok:
            result = response.json()
            if result["success"]:
                self.session_id = result["session_id"]
                print(f"Started session {self.session_id} with {result['num_samples']} samples.")
            else:
                raise RuntimeError(f"Failed to start session: {result['error']}")
        else:
            raise RuntimeError(f"Failed to connect to server: {response.status_code}")

    def fetch_next_batch(self):
        url = f"{self.api_url}/get_next_batch/{self.session_id}/{self.api_key}"
        response = requests.get(url)
        if response.ok:
            result = response.json()
            if result["success"]:
                return result["epoch"], result["batch_id"], result["batch"]
            else:
                raise RuntimeError(f"Error fetching batch: {result['error']}")
        else:
            raise RuntimeError("API request failed")

    def mark_batch_done(self, epoch, batch_id):
        url = f"{self.api_url}/mark_batch_done/{self.session_id}/{batch_id}/{self.api_key}?epoch={epoch}"
        response = requests.post(url)
        if not response.ok or not response.json().get("success"):
            print("Warning: Failed to mark batch done.")

    def log(self, level, message):
        url = f"{self.api_url}/log/{self.session_id}/{self.api_key}"
        requests.post(url, json={"level": level, "message": message})

    def end_session(self):
        url = f"{self.api_url}/end_training_session/{self.session_id}/{self.api_key}"
        requests.post(url)

    def simulate_training_loop(self, steps=None, sleep=1.0):
        step_count = 0
        while True:
            try:
                epoch, batch_id, batch = self.fetch_next_batch()
            except RuntimeError as e:
                print(f"Stopped training: {e}")
                break

            print(f"Training on batch {batch_id} from epoch {epoch} ({len(batch)} samples)...")
            time.sleep(sleep)  # Simulate training

            self.mark_batch_done(epoch, batch_id)
            self.log("INFO", f"Completed batch {batch_id} in epoch {epoch}")

            step_count += 1
            if steps and step_count >= steps:
                break

        self.end_session()
        print("Training session ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated training client for speech data server")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--order", choices=["asc", "desc", "random"], default="asc", help="Sample order")
    parser.add_argument("--min_duration", type=float, default=0.0, help="Minimum duration filter")
    parser.add_argument("--max_duration", type=float, default=None, help="Maximum duration filter")
    parser.add_argument("--steps", type=int, default=None, help="Max number of batches to process")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep time per batch (simulate training)")
    args = parser.parse_args()

    dataset = SpeechDataset(config_path=args.config)
    dataset.start_session(
        batch_size=args.batch_size,
        order=args.order,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
    dataset.simulate_training_loop(steps=args.steps, sleep=args.sleep)

