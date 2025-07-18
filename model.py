import torch.nn as nn
import torchaudio
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge

def build_encoder(args, vocab_size):
     if args.encoder == "lstm":
         # standard PyTorch LSTM
         return nn.LSTM(
             input_size=args.embedding_dim,
             hidden_size=args.hidden_size,
             num_layers=args.num_layers,
             batch_first=True,
             bidirectional=False,
             dropout=0.0
         )

     elif args.encoder == "xlstm":
         # return the xLSTM config; instantiation happens in ASRModel
         cfg = xLSTMLargeConfig(
             embedding_dim=args.embedding_dim,
             num_heads=args.num_heads,
             num_blocks=args.num_blocks,
             vocab_size=vocab_size,
             return_last_states=True,
             mode="train",
             chunkwise_kernel=args.chunkwise_kernel,
             sequence_kernel=args.sequence_kernel,
             step_kernel=args.step_kernel,
         )
         return cfg

     else:
         raise ValueError(f"Unknown encoder type: {args.encoder!r}")


def make_frontend(ftype: str, sample_rate: int):
     if ftype == "mfcc":
         return torchaudio.transforms.MFCC(
             sample_rate=sample_rate,
             n_mfcc=80,
             log_mels=True,
             melkwargs={"n_mels": 80}
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
     def __init__(self, frontend: nn.Module, encoder, vocab_size: int, feat_dim: int, debug: bool = True):
         super().__init__()
         self.frontend = frontend
         self.debug = debug

         # PyTorch LSTM encoder instance
         if isinstance(encoder, nn.LSTM):
             self.encoder = encoder
             hidden = encoder.hidden_size
             bidi = encoder.bidirectional
             self.enc_out_dim = hidden * (2 if bidi else 1)

         # xLSTM config: instantiate projector + xLSTMLarge
         elif isinstance(encoder, xLSTMLargeConfig):
             cfg = encoder
             self.proj = nn.Linear(feat_dim, cfg.embedding_dim)
             self.encoder = xLSTMLarge(cfg)
             self.enc_out_dim = cfg.embedding_dim

         else:
             raise ValueError(f"Unknown encoder provided: {type(encoder)}")

         self.classifier = nn.Linear(self.enc_out_dim, vocab_size)

     def forward(self, feats, states=None):
         # Debug shapes before any change
         if self.debug:
             print(f"[DEBUG] Input feats shape: {tuple(feats.shape)}")

         # feats may come as (B, F, T) or (B, T, F); ensure (B, T, F)
         if feats.dim() == 3 and feats.size(1) > feats.size(2):
             if self.debug:
                 print(f"[DEBUG] Transposing feats from (B,F,T) to (B,T,F)")
             feats = feats.transpose(1, 2).contiguous()

         if self.debug:
             print(f"[DEBUG] Feats after transpose shape: {tuple(feats.shape)}")

         # apply encoder
         if hasattr(self, 'proj'):
             if self.debug:
                 print(f"[DEBUG] Projecting feats from dim {feats.size(-1)} to {self.proj.out_features}")
             x = self.proj(feats)
             if self.debug:
                 print(f"[DEBUG] After proj shape: {tuple(x.shape)}")
             logits, new_states = self.encoder(x, states)
         else:
             if self.debug:
                 print(f"[DEBUG] Passing feats into LSTM with input dim {feats.size(-1)}")
             logits, new_states = self.encoder(feats, states)

         if self.debug:
             print(f"[DEBUG] Encoder output logits shape: {tuple(logits.shape)}")
             print(f"[DEBUG] Encoder new_states: {new_states}")

         # map to vocabulary
         logits = self.classifier(logits)
         if self.debug:
             print(f"[DEBUG] After classifier shape: {tuple(logits.shape)}")

         return logits, new_states

