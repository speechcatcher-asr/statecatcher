import torch

def ctc_greedy_decoder(log_probs, input_lengths, blank=0):
    """
    Decodes CTC log probabilities using greedy decoding.

    Args:
        log_probs: Tensor of shape (batch, time, vocab_size), typically log-softmax output.
        input_lengths: 1D Tensor of lengths for each sequence in the batch.
        blank: Index of the CTC blank token.

    Returns:
        List of decoded sequences (List[List[int]])
    """
    # Take argmax over vocab dimension
    preds = torch.argmax(log_probs, dim=-1)  # shape: (batch, time)

    decoded = []
    for b in range(preds.size(0)):
        pred = preds[b, :input_lengths[b]]  # Trim to actual length
        prev_token = None
        out = []
        for token in pred:
            token = token.item()
            if token != blank and token != prev_token:
                out.append(token)
            prev_token = token
        decoded.append(out)

    return decoded
