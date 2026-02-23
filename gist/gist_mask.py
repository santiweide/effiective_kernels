import torch

def segmented_attention(q, Kg, Vg, Kr, Vr, scale=None):
    """
    q:  (B, H, Tq, D)
    Kg: (B, H, Tg, D)  shared gist KV
    Vg: (B, H, Tg, D)
    Kr: (B, H, Tr, D)  per-request KV (paged/contiguous logical)
    Vr: (B, H, Tr, D)

    Returns: (B, H, Tq, D)
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    # logits per segment
    lg = torch.matmul(q, Kg.transpose(-1, -2)) * scale   # (B,H,Tq,Tg)
    lr = torch.matmul(q, Kr.transpose(-1, -2)) * scale   # (B,H,Tq,Tr)

    # stable softmax across concatenation WITHOUT concatenating:
    m = torch.maximum(lg.max(dim=-1, keepdim=True).values,
                      lr.max(dim=-1, keepdim=True).values)  # (B,H,Tq,1)

    eg = torch.exp(lg - m)  # (B,H,Tq,Tg)
    er = torch.exp(lr - m)  # (B,H,Tq,Tr)

    Z = eg.sum(dim=-1, keepdim=True) + er.sum(dim=-1, keepdim=True)  # (B,H,Tq,1)

    # weighted sum per segment, normalized by shared denominator
    og = torch.matmul(eg, Vg)  # (B,H,Tq,D)
    orr = torch.matmul(er, Vr) # (B,H,Tq,D)

    out = (og + orr) / Z
    return out

