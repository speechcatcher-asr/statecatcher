# Statecatcher

**Statecatcher** is part of the [speechcatcher-asr](https://github.com/speechcatcher-asr/speechcatcher) project. It's an engine for training ASR models directly using the [speechcatcher-data API](https://github.com/speechcatcher-asr/speechcatcher-data) for on-the-fly training data generation.  

The primary goal is **stateful training**: long audio recordings (e.g. hour-long podcasts) are segmented for training. Crucially, between gradient updates, **model state is propagated** - the state of the model after processing one segment is used to initialize the next. This enables modeling **very long contexts** without overwhelming VRAM. It also allows to simplify decoding for streaming ASR.

Emerging recurrent architectures like [xLSTM](https://github.com/speechcatcher-asr/xlstm) ([original fork](https://github.com/NX-AI/xlstm)) and [RWKV](https://www.rwkv.com/) — modern RNNs adapted for the transformer era — are likely ideal for this paradigm. See the paper [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) for more background.

## Key Features

- Streaming data directly via HTTP(S) from [speechcatcher-data](https://github.com/speechcatcher-asr/speechcatcher-data) means compute and storage don’t have to reside on the same server.
- Efficient use of TCP keep-alive avoids NFS-related I/O bottlenecks with many small files.
- Enables distributed and scalable training setups with minimal storage constraints.

## Current Status (Work in Progress)

| Component | Status |
|----------|--------|
| PyTorch dataset for speechcatcher-data | ✅ |
| sentencepiece tokens | ✅ |
| Training loop and batching | ✅ |
| `nn.LSTM` | ✅ |
| `xLSTM` ([original](https://github.com/NX-AI/xlstm),[fork](https://github.com/speechcatcher-asr/xlstm)) | ✅ |
| CTC (Connectionist Temporal Classification) loss | ✅ |
| warp-transducer (RNN-T) | ⚠️ Needs more testing |
| [Aim experiment tracking](https://github.com/aimhubio/aim) | ✅ |
| RWKV | 🔄 Planned/WiP |
| Proper input masking | 🔄 Planned/WiP |
| On-the-fly hallucination detection & filtering | 🔄 Planned/WiP |

## Roadmap

The plan is to eventually make **statecatcher** the default ASR backend in [speechcatcher](https://github.com/speechcatcher-asr/speechcatcher) once the engine is stable and mature.

While the current focus is ASR, **stateful training** could be extended to other domains such as:
- Self-supervised speech representation learning
- Text-to-Speech (TTS)
- General speech classification

## Contributions

Pull requests are very welcome! 
