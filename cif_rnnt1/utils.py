import k2
import torch

def forced_alignment(
    logits : torch.Tensor,
    tokens : list[int],
    blank_id = 0,
) -> list:
    logits = logits[0].cpu()
    T, U, V = logits.shape    
    
    # 1. Build trellis
    trellis = torch.zeros((T, U))
    trellis[1:, 0] = logits[:-1, 0, 0].cumsum(dim=0)
    trellis[0, 1:] = logits[0, range(U-1), tokens].cumsum(dim=0)
    for t in range(1, T):
        for u in range(1, U):
            trellis[t, u] = torch.maximum(
                trellis[t-1, u] + logits[t-1, u, blank_id],
                trellis[t, u-1] + logits[t, u-1, tokens[u-1]],
            )
    
    # 2. Backtrack
    t = T-1
    u = U-1
    path = [(t, u, blank_id)]
    for step in range(T+U, 2, -1):
        if not t:
            u -= 1
            path.append((t, u, tokens[u]))
        elif not u:
            t -= 1
            path.append((t, u, blank_id))
        elif trellis[t-1, u] > trellis[t, u-1]:
            t -= 1
            path.append((t, u, blank_id))
        else:
            u -= 1
            path.append((t, u, tokens[u]))
    assert (t, u) == (0, 0), (t, u)
    
    return path[::-1]   
    
    

def best_path_search(
    model,
    encoder_out: torch.Tensor,
    tokens: list[int],
) -> list[int]:
    """Greedy search for a single utterance.
    Args:
      model:
        An instance of `Transducer`.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder. Support only N==1 for now.
      max_sym_per_frame:
        Maximum number of symbols per frame. If it is set to 0, the WER
        would be 100%.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert encoder_out.ndim == 3

    # support only batch_size == 1 for now
    assert encoder_out.size(0) == 1, encoder_out.size(0)

    blank_id = model.decoder.blank_id

    device = next(model.parameters()).device

    sos_y_padded = torch.tensor([[blank_id, *tokens]], device=device)
    # y = k2.RaggedTensor([tokens]).to(device)
    # sos_y = add_sos(y, sos_id=0)
    # sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

    decoder_out = model.decoder(sos_y_padded)

    full_logits = model.joiner(encoder_out.unsqueeze(2), decoder_out.unsqueeze(1))

    forced_alignment_path = forced_alignment(full_logits, tokens)
    tokens_w_blank = [t for *_, t in forced_alignment_path]

    return tokens_w_blank

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)

def concat(ragged: k2.RaggedTensor, value: int, direction: str) -> k2.RaggedTensor:
    """Prepend a value to the beginning of each sublist or append a value.
    to the end of each sublist.

    Args:
        ragged: 
        A ragged tensor with two axes.
        value:
        The value to prepend or append.
        direction:
        It can be either "left" or "right". If it is "left", we
        prepend the value to the beginning of each sublist;
        if it is "right", we append the value to the end of each
        sublist.

    Returns:
        Return a new ragged tensor, whose sublists either start with
        or end with the given value.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> concat(a, value=0, direction="left")
    [ [ 0 1 3 ] [ 0 5 ] ]
    >>> concat(a, value=0, direction="right")
    [ [ 1 3 0 ] [ 5 0 ] ]

    """
    dtype = ragged.dtype
    device = ragged.device

    assert ragged.num_axes == 2, f"num_axes: {ragged.num_axes}"
    pad_values = torch.full(
        size=(ragged.tot_size(0), 1),
        fill_value=value,
        device=device,
        dtype=dtype,
    )
    pad = k2.RaggedTensor(pad_values)

    if direction == "left":
        ans = k2.ragged.cat([pad, ragged], axis=1)
    elif direction == "right":
        ans = k2.ragged.cat([ragged, pad], axis=1)
    else:
        raise ValueError(
            f'Unsupported direction: {direction}. " \
            "Expect either "left" or "right"'
        )
    return ans


def add_sos(ragged: k2.RaggedTensor, sos_id: int) -> k2.RaggedTensor:
    """Add SOS to each sublist.

    Args:
        ragged:
        A ragged tensor with two axes.
        sos_id:
        The ID of the SOS symbol.

      Returns:
      Return a new ragged tensor, where each sublist starts with SOS.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> add_sos(a, sos_id=0)
    [ [ 0 1 3 ] [ 0 5 ] ]

    """
    return concat(ragged, sos_id, direction="left")


def add_eos(ragged: k2.RaggedTensor, eos_id: int) -> k2.RaggedTensor:
    """Add EOS to each sublist.

    Args:
        ragged:
        A ragged tensor with two axes.
        eos_id:
        The ID of the EOS symbol.

    Returns:
        Return a new ragged tensor, where each sublist ends with EOS.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> add_eos(a, eos_id=0)
    [ [ 1 3 0 ] [ 5 0 ] ]

    """
    return concat(ragged, eos_id, direction="right")

