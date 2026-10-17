import random

import pytest
import torch

from src import data as D
from src import quant as Q


class FakeTok:
    """Character-level tokenizer stand-in with offsets (no model download needed)."""
    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


TEXT = ("<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nCALL<|im_end|>\n"
        "<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n<|im_start|>assistant\nok<|im_end|>\n")


def test_char_roles_labels_every_role():
    ids, roles = D.tokenize_with_roles(FakeTok(), TEXT)
    got = "".join(TEXT[i] for i, r in enumerate(roles) if r == D.ROLE_ID["assistant"])
    assert got == "CALLok"
    assert "".join(TEXT[i] for i, r in enumerate(roles) if r == D.ROLE_ID["system"]) == "SYS"
    assert "".join(TEXT[i] for i, r in enumerate(roles) if r == D.ROLE_ID["user"]) == "hi"
    tr = "".join(TEXT[i] for i, r in enumerate(roles) if r == D.ROLE_ID["tool_result"])
    assert tr.startswith("<tool_response>") and tr.endswith("</tool_response>")
    assert "<|im_start|>" not in got and "<|im_end|>" not in tr


def test_episodes_deterministic_and_well_formed():
    a, b = D.sample_episode(random.Random(3)), D.sample_episode(random.Random(3))
    assert a == b
    for req, name, args, res, ans in a.turns:
        assert name in a.tools
        assert set(args) == set(D.TOOLS[name][2])     # all required args present
        assert ans


def test_quantize_w_levels_and_error():
    torch.manual_seed(0)
    w = torch.randn(8, 256)
    wq = Q.quantize_w(w)
    for g in wq.reshape(8, 2, 128).reshape(16, 128):
        assert len(torch.unique(g)) <= 16                # 4-bit: at most 16 levels per group
    assert (wq - w).abs().max() <= (Q.group_scales(w).max() / 2 + 1e-6)


def test_fp8_naive_scale_cast_hurts_more_than_channel_scale():
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.02                      # LLM-like small weights -> small scales
    e_naive = (Q.quantize_w(w, scale_dtype=torch.float8_e4m3fn) - w).pow(2).mean()
    e_chan = (Q.quantize_w(w, scale_dtype=torch.float8_e4m3fn, channel_scale=True) - w).pow(2).mean()
    e_fp32 = (Q.quantize_w(w) - w).pow(2).mean()
    assert e_chan < e_naive and e_fp32 <= e_chan * 1.5


def test_scale_search_alpha0_is_rtn_and_best_not_worse():
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 32, bias=False)
    x = torch.randn(64, 256)
    x[:, 3] *= 30                                        # salient channel
    s, losses = Q.search_scale(x.abs().mean(0), x, [lin])
    ref = x @ lin.weight.T
    rtn = (x @ Q.quantize_w(lin.weight.data).T - ref).pow(2).mean().item()
    assert losses[0] == pytest.approx(rtn, rel=1e-5)
    assert min(losses) <= rtn


def test_collector_respects_mask():
    class Lin(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.q_proj = torch.nn.Identity(); s.gate_proj = torch.nn.Identity(); s.down_proj = torch.nn.Identity()
    class Layer(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.self_attn = Lin(); s.mlp = Lin()
        def forward(s, x):
            return s.mlp.down_proj(s.mlp.gate_proj(s.self_attn.q_proj(x)))
    class M(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.model = torch.nn.Module(); s.model.layers = torch.nn.ModuleList([Layer()])
        def forward(s, input_ids):
            return s.model.layers[0](input_ids.float()[..., None].repeat(1, 1, 4))
    m = M()
    ids = torch.tensor([[1, 100, 2]])
    keep = torch.tensor([[True, False, True]])
    st = Q.Collector(m, n_loss_tokens=10).run(m, [(ids, keep)])
    xmean, rows = st[(0, "qkv")]
    assert torch.allclose(xmean, torch.full((4,), 1.5))  # the masked 100 is excluded
    assert rows.shape[0] == 2 and 100 not in rows


def test_awq_fold_is_function_preserving_before_rounding():
    torch.manual_seed(0)
    norm_w = torch.rand(256) + 0.5
    lin = torch.nn.Linear(256, 16, bias=False)
    x = torch.randn(5, 256)
    s = torch.rand(256) + 0.5
    y0 = (x * norm_w) @ lin.weight.T
    y1 = (x * (norm_w / s)) @ (lin.weight * s).T
    assert torch.allclose(y0, y1, atol=1e-5)
