def build_gist_cache(model, prompt_ids, k, gist_token_id):
    """
    Returns: gist_cache = list[layer] of (K_gist, V_gist),
             where K_gist/V_gist have sequence length k.
    """
    gist_ids = [gist_token_id] * k
    ids = prompt_ids + gist_ids

    # Construct gist mask (training-style): later tokens can’t attend to prompt except via gist.
    attn_mask = make_gist_mask(len(prompt_ids), k, total_len=len(ids))

    # Run a full forward pass (no generation), requesting per-layer KV.
    outputs = model.forward(
        input_ids=ids,
        attention_mask=attn_mask,
        use_cache=True
    )
    past = outputs.past_key_values  # list[layer] of (K_all, V_all), length = len(ids)

    # Slice K/V at gist positions only
    gist_start = len(prompt_ids)
    gist_end = gist_start + k
    gist_cache = []
    for (K_all, V_all) in past:
        K_g = K_all[:, :, gist_start:gist_end, :]  # [B, H, k, Dh]
        V_g = V_all[:, :, gist_start:gist_end, :]
        gist_cache.append((K_g, V_g))

    return gist_cache


def generate_with_gist_cache(model, gist_cache, x_ids, max_new_tokens):
    """
    Autoregressive decoding with prefix KV initialized from gist_cache.
    """
    past = gist_cache
    pos = prefix_len(gist_cache)  # typically k; used for position_ids / RoPE offset

    # Prime with x (can be done in a single forward with past, or token-by-token)
    for tok in x_ids:
        outputs = model.forward(
            input_ids=[tok],
            past_key_values=past,
            use_cache=True,
            position_ids=[pos]   # or rope_offset=pos depending on implementation
        )
        past = outputs.past_key_values
        pos += 1

    # Decode new tokens
    y = []
    for _ in range(max_new_tokens):
        next_tok = sample(outputs.logits[-1])
        y.append(next_tok)

        outputs = model.forward(
            input_ids=[next_tok],
            past_key_values=past,
            use_cache=True,
            position_ids=[pos]
        )
        past = outputs.past_key_values
        pos += 1

    return y

def attention_step(q, past_k, past_v, k_new, v_new):
    K = concat(past_k, k_new)   # prefix (gist) + current
    V = concat(past_v, v_new)
    scores = (q @ K.T) / sqrt(d)
    a = softmax(scores)
    return a @ V