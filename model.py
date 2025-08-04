import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Optional, Tuple, Any
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge
from lucyrnn import LucyRNN
from lucyrnn_conf import LucyRNNConfig
from lucyrnn_triton import LucyRNNtriton

def detach_states(states):
    """Recursively detach all tensors in nested state structures (dicts, tuples, lists)."""
    if states is None:
        return None
    elif isinstance(states, torch.Tensor):
        return states.detach()
    elif isinstance(states, dict):
        return {k: detach_states(v) for k, v in states.items()}
    elif isinstance(states, tuple):
        return tuple(detach_states(s) for s in states)
    elif isinstance(states, list):
        return [detach_states(s) for s in states]
    else:
        # Catch any unexpected data type (e.g., numbers, strings)
        return states

def assert_all_detached(x):
    if isinstance(x, torch.Tensor):
        assert not x.requires_grad, "Tensor still requires grad"
    elif isinstance(x, (list, tuple)):
        for v in x:
            assert_all_detached(v)
    elif isinstance(x, dict):
        for v in x.values():
            assert_all_detached(v)

def compute_loss(
    mode: str,
    criterion: nn.Module,
    model: nn.Module,
    feats: torch.Tensor,              # (B, T, F)
    tokens: torch.Tensor,             # (B, U)
    in_lens: torch.Tensor,            # (B,)
    tgt_lens: torch.Tensor,           # (B,)
    blank_id: int,
    use_rnnt_joiner: Optional[nn.Module] = None,
    input_state: Optional[Any] = None,
    args=None,
    compact=False
) -> Tuple[torch.Tensor, Optional[Any]]:
    """
    Computes loss (CTC or RNN-T) given model, features, and labels.

    Returns:
        loss: scalar tensor
        output_state: model's output state (for stateful models)
    """
    # Detach input state if present
    if input_state:
        input_state = detach_states(input_state)
        if args and args.debug:
            assert_all_detached(input_state)

    # Forward pass through model
    enc_out, output_state = model(feats, input_state)  # (B, T, D)

    if mode == "ctc":
        # CTC expects (T, B, V) after log_softmax
        logp = enc_out.log_softmax(-1).transpose(0, 1)  # (T, B, V)
        loss = criterion(logp, tokens, in_lens, tgt_lens)

    elif mode == "rnnt":
        assert use_rnnt_joiner is not None, "Joiner module required for RNN-T mode"

        # Add blank token as prefix to labels for predictor input
        blank_prefix = torch.full(
            (tokens.size(0), 1),
            blank_id,
            dtype=tokens.dtype,
            device=tokens.device
        )
        predictor_input = torch.cat([blank_prefix, tokens], dim=1)  # (B, U + 1)

        # RNN-T joiner forward pass: logits shape (B, T, U+1, V)
        if args.compact_rnnt:
            logits = use_rnnt_joiner(enc_out, predictor_input, in_lens, tgt_lens)
        else:
            logits = use_rnnt_joiner(enc_out, predictor_input)

        # warp_rnnt expects float32
        log_probs = logits.log_softmax(dim=-1)
        if log_probs.dtype != torch.float32:
            log_probs = log_probs.to(torch.float32)

        # Labels for loss must be (B, U)
        loss = criterion(
            log_probs=log_probs,
            labels=tokens,
            frames_lengths=in_lens,
            labels_lengths=tgt_lens,
            blank_id=blank_id,
            compact=compact,
            gather=True
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return loss, output_state, enc_out, output_state

class RNNTPredictorJoiner(nn.Module):
    def __init__(self, enc_out_dim: int, pred_emb_dim: int, join_dim: int, vocab_size: int, debug: bool=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, pred_emb_dim)

        # Project encoder and predictor into shared join space
        self.enc_proj = nn.Linear(enc_out_dim, join_dim)
        self.pred_proj = nn.Linear(pred_emb_dim, join_dim)

        self.debug=debug

        if self.debug:
            print(f"[DEBUG] RNNT pred init with {enc_out_dim=}, {pred_emb_dim=}, {join_dim=}, {vocab_size=}")

        # Final projection to vocab logits
        self.joiner = nn.Linear(join_dim, vocab_size)

    def forward(self, enc_out: torch.Tensor, prefix: torch.Tensor):
        
        if self.debug:
            print(f"[DEBUG] RNNT: {enc_out.shape=}")

        # enc_out: (B, T, enc_out_dim)
        # prefix:  (B, U)
        pred_emb = self.embedding(prefix)           # (B, U, pred_emb_dim)

        enc_proj = self.enc_proj(enc_out)           # (B, T, join_dim)
        pred_proj = self.pred_proj(pred_emb)        # (B, U, join_dim)

        # Broadcasted addition
        joint = enc_proj.unsqueeze(2) + pred_proj.unsqueeze(1)  # (B, T, U, join_dim)
        joint = torch.tanh(joint)
        logits = self.joiner(joint)                 # (B, T, U, vocab_size)
        return logits

class RNNTCompactPredictorJoiner(nn.Module):
    def __init__(self, enc_out_dim: int, pred_emb_dim: int, join_dim: int, vocab_size: int, debug: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, pred_emb_dim)
        self.enc_proj = nn.Linear(enc_out_dim, join_dim)
        self.pred_proj = nn.Linear(pred_emb_dim, join_dim)
        self.joiner = nn.Linear(join_dim, vocab_size)
        self.debug = debug

    def forward_compact(
        self,
        enc_out: torch.Tensor,      # (B, T, enc_out_dim)
        prefix: torch.Tensor,       # (B, U)
        in_lens: torch.Tensor,      # (B,)
        tgt_lens: torch.Tensor      # (B,)
    ) -> torch.Tensor:
        """
        Computes (STU, V) logits for compact RNN-T loss.
        """
        B = enc_out.size(0)
        device = enc_out.device

        enc_proj = self.enc_proj(enc_out)     # (B, T, J)
        pred_emb = self.embedding(prefix)     # (B, U, E)
        pred_proj = self.pred_proj(pred_emb)  # (B, U, J)

        enc_list = []
        pred_list = []

        for b in range(B):
            T = in_lens[b]
            U = tgt_lens[b] + 1  # include blank prefix

            # Get enc_proj[b, :T] → (T, J)
            # Get pred_proj[b, :U] → (U, J)
            enc_b = enc_proj[b, :T]             # (T, J)
            pred_b = pred_proj[b, :U]           # (U, J)

            # Cartesian product: (T, J) + (U, J) → (T * U, J)
            enc_b_rep = enc_b.unsqueeze(1).expand(T, U, -1)
            pred_b_rep = pred_b.unsqueeze(0).expand(T, U, -1)

            joint = torch.tanh(enc_b_rep + pred_b_rep)     # (T, U, J)
            joint = joint.reshape(-1, joint.size(-1))      # (T*U, J)

            enc_list.append(joint)

        all_joint = torch.cat(enc_list, dim=0)              # (STU, J)
        logits = self.joiner(all_joint)                     # (STU, V)
        return logits


    def forward(self, enc_out, prefix, in_lens, tgt_lens):
        return self.forward_compact(enc_out, prefix, in_lens, tgt_lens)

def build_encoder(args, vocab_size, feat_dim=80, is_training=True):
     if args.encoder == "lstm":
         # standard PyTorch LSTM
         return nn.LSTM(
             input_size=args.input_proj_dim if args.input_proj_dim != -1 else feat_dim,
             hidden_size=args.hidden_size,
             num_layers=args.num_layers,
             batch_first=True,
             bidirectional=False,
             dropout=0.0
         )

     elif args.encoder == "xlstm":
         # return the xLSTM config; instantiation happens in ASRModel
         cfg = xLSTMLargeConfig(
             embedding_dim=feat_dim,
             input_dim=feat_dim,
             num_heads=args.num_heads,
             num_blocks=args.num_blocks,
             vocab_size=vocab_size,
             return_last_states=True,
             mode="train",
             chunkwise_kernel=args.chunkwise_kernel,
             sequence_kernel=args.sequence_kernel,
             step_kernel=args.step_kernel,
             autocast_kernel_dtype="float16",
         )
         return cfg

     elif args.encoder == "lucyrnn":
         lucyrnn_config = LucyRNNConfig(
             input_dim=args.input_proj_dim if args.input_proj_dim != -1 else feat_dim,
             hidden_dim=args.hidden_size,
             num_layers=args.num_layers,
             vocab_size=vocab_size,
             return_last_states=True,
             kernel_impl="triton",  # use 'triton' when implemented
             fused_ops=True,
             stack_order=1,
             layer_norm=False,
             is_training=is_training
         )
         print("[DEBUG:] lucyrnn_config is:", lucyrnn_config)
         return lucyrnn_config
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
    def __init__(self, frontend: nn.Module, encoder, vocab_size: int, feat_dim: int, proj_dim: int, debug: bool = True):
         super().__init__()
         self.frontend = frontend
         self.debug = debug

         # PyTorch LSTM encoder instance
         if isinstance(encoder, nn.LSTM):
             self.cfg = None
             self.encoder = encoder
             hidden = encoder.hidden_size
             bidi = encoder.bidirectional
             self.rnn_out_dim = hidden * (2 if bidi else 1)
             self.enc_out_dim = vocab_size
             self.classifier = nn.Linear(self.rnn_out_dim, vocab_size)
             if proj_dim > 0:
                self.proj = nn.Linear(feat_dim, proj_dim)

         # xLSTM config: instantiate projector + xLSTMLarge
         elif isinstance(encoder, xLSTMLargeConfig):
             self.cfg = encoder
             self.encoder = xLSTMLarge(self.cfg)
             self.enc_out_dim = vocab_size
             self.input_seq_pad_factor = 64
             if proj_dim > 0:
                self.proj = nn.Linear(feat_dim, proj_dim)
         elif isinstance(encoder, LucyRNNConfig):
             self.cfg = encoder
             self.encoder = LucyRNNtriton(self.cfg)
             self.enc_out_dim = vocab_size
             self.input_seq_pad_factor = 8
             if proj_dim > 0:
                self.proj = nn.Linear(feat_dim, proj_dim)
         else:
             raise ValueError(f"Unknown encoder provided: {type(encoder)}")


    def forward(self, feats, states=None):
         # Debug shapes before any change
         if self.debug:
             print(f"[DEBUG] Input feats shape: {tuple(feats.shape)}")
             if states is None:
                print(f"[DEBUG] states is None - initializing encoder with a new state.")

         if self.debug:
             print(f"[DEBUG] Feats after transpose shape: {tuple(feats.shape)}")

         # apply input projection
         if hasattr(self, 'proj'):
             if self.debug:
                 print(f"[DEBUG] Projecting feats from dim {feats.size(-1)} to {self.proj.out_features}")
             feats = self.proj(feats)
             if self.debug:
                 print(f"[DEBUG] After proj shape: {tuple(feats.shape)}")
         else:
             if self.debug:
                 print(f"[DEBUG] Passing feats into RNN with input dim {feats.size(-1)}")

         # pad input sequence length if using xLSTM and sequence not divisible by 16 ---
         if isinstance(self.encoder, xLSTMLarge):
             seq_len = feats.size(1)
             remainder = seq_len % self.input_seq_pad_factor
             if remainder != 0:
                 pad_len = self.input_seq_pad_factor - remainder
                 if self.debug:
                     print(f"[DEBUG] Padding sequence length from {seq_len} to {seq_len + pad_len}")
                 feats = F.pad(feats, (0, 0, 0, pad_len))  # Pad time dimension (dim=1)

         # run encoder, with or without states
         if states is not None:
             if self.debug:
                 print(f"[DEBUG] Run encoder with a previous state (stateful training).")
             logits, new_states = self.encoder(feats, states)
         else:
             if self.debug:
                 print(f"[DEBUG] Run encoder with a new state.")
             logits, new_states = self.encoder(feats)

         if self.debug:
             print(f"[DEBUG] Encoder output logits shape: {tuple(logits.shape)}")

         if hasattr(self, 'classifier'):
             # map to vocabulary for LSTM
             logits = self.classifier(logits)
             if self.debug:
                 print(f"[DEBUG] After classifier shape: {tuple(logits.shape)}")

         return logits, new_states


