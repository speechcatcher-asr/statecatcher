import torch.nn as nn
import torchaudio
from xlstm.xlstm_large.model import xLSTMLargeConfig

def build_encoder(args, vocab_size):
    if args.encoder == "lstm":
        # standard PyTorch LSTM
        # input_size=embedding_dim, hidden_size, num_layers, batch_first=True for (B,T,Feats)
        return nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            batch_first=True,
            bidirectional=False,     # or True if you want bi‐LSTM
            dropout=0.0              # set dropout between layers if desired
        )

    elif args.encoder == "xlstm":
        # xLSTM “large” variant
        xlstm_cfg = {
            "embedding_dim":           args.embedding_dim,
            "num_heads":               args.num_heads,
            "num_blocks":              args.num_blocks,
            "vocab_size":              vocab_size,
            "return_last_states":      True,
            "mode":                    "train",
            "chunkwise_kernel":        args.chunkwise_kernel,
            "sequence_kernel":         args.sequence_kernel,
            "step_kernel":             args.step_kernel,
            # all other parameters will take their defaults
        }
        return xLSTMLargeConfig(**xlstm_cfg)

    else:
        raise ValueError(f"Unknown encoder type: {args.encoder!r}")


def make_frontend(ftype: str, sample_rate: int):
    if ftype == "mfcc":
        return torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=80,
            log_mels=True
        )
    elif ftype == "mel":
        return nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=80
            ),
            torchaudio.transforms.AmplitudeToDB()
        )
    else:
        raise ValueError(f"Unsupported frontend: {ftype}")

class ASRModel(nn.Module):
    def __init__(self, frontend: nn.Module, encoder_type: str,
                 encoder_cfg: dict, vocab_size: int, feat_dim: int):
        super().__init__()
        self.frontend = frontend
        self.encoder_type = encoder_type

        if encoder_type == "lstm":
            hidden = encoder_cfg["hidden_size"]
            layers = encoder_cfg["num_layers"]
            self.encoder = nn.LSTM(
                input_size=feat_dim,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                bidirectional=True,
            )
            self.enc_out_dim = hidden * 2

        elif encoder_type == "xlstm":
            from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge
            cfg = xLSTMLargeConfig(**encoder_cfg)
            self.encoder = xLSTMLarge(cfg)
            self.proj = nn.Linear(feat_dim, cfg.embedding_dim)
            self.enc_out_dim = cfg.embedding_dim

        else:
            raise ValueError(f"Unknown encoder: {encoder_type}")

        self.classifier = nn.Linear(self.enc_out_dim, vocab_size)

    def forward(self, feats, states=None):
        if self.encoder_type == "xlstm":
            x = self.proj(feats)
            logits, new_states = self.encoder(x, states)
        else:
            logits, new_states = self.encoder(feats, states)

        logits = self.classifier(logits)
        return logits, new_states
