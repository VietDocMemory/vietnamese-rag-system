import pytest
import torch

from ctx_to_lora.modeling.idefics2 import (
    Idefics2PerceiverConfig,
    Idefics2PerceiverResampler,
    Idefics2PerceiverSdpaAttention,
    repeat_kv,
)


def config():
    return Idefics2PerceiverConfig(
        input_size=8,
        hidden_size=8,
        n_heads=2,
        head_dim=4,
        num_key_value_heads=1,
        n_latents=2,
        num_blocks=1,
        num_self_attn_per_block=1,
        shared_weights=False,
        intermediate_size_factor=2,
        attn_implementation="sdpa",
    )


@pytest.mark.parametrize("cross", [True, False])
def test_sdpa_matches_explicit_noncausal_attention(cross):
    torch.manual_seed(1)
    module = Idefics2PerceiverSdpaAttention(config()).eval()
    latents = torch.randn(2, 2, 8)
    context = torch.randn(2, 3, 8)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    output = module(latents, cross, context, attention_mask=mask)[0]
    kv = context if cross else latents
    q = module.q_proj(latents).view(2, 2, 2, 4).transpose(1, 2)
    k = repeat_kv(module.k_proj(kv).view(2, -1, 1, 4).transpose(1, 2), 2)
    v = repeat_kv(module.v_proj(kv).view(2, -1, 1, 4).transpose(1, 2), 2)
    weights = q @ k.transpose(-1, -2) / 2
    if cross:
        weights = weights.masked_fill(~mask[:, None, None, :].bool(), float("-inf"))
    expected = (weights.softmax(-1) @ v).transpose(1, 2).reshape(2, 2, 8)
    expected = module.o_proj(expected)
    torch.testing.assert_close(output, expected)


def test_resampler_ignores_masked_context_and_rejects_packed_input():
    torch.manual_seed(2)
    model = Idefics2PerceiverResampler(config()).eval()
    context = torch.randn(1, 3, 8)
    mask = torch.tensor([[1, 1, 0]])
    original = model(context, attention_mask=mask)
    context[:, 2] = 1000
    torch.testing.assert_close(original, model(context, attention_mask=mask))
    assert original.shape == (1, 2, 8)
    with pytest.raises(ValueError, match="packed"):
        model(context, position_ids=torch.tensor([[0, 1, 0]]))
