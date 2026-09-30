"""Import the real bundled runtime without a GPU, downloads or checkpoint."""

from pathlib import Path
import pytest

from ctx_to_lora import model_loading
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel


def test_runtime_imports_and_packaged_template_is_independent_of_cwd(monkeypatch, tmp_path):
    class Tokenizer:
        pad_token_id = 0
        chat_template = "original"

    tokenizer = Tokenizer()
    monkeypatch.setattr(model_loading.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.chdir(tmp_path)
    loaded = model_loading.get_tokenizer("google/gemma-2-2b-it")
    assert "<start_of_turn>" in loaded.chat_template
    assert loaded.chat_template != "original"
    assert callable(ModulatedPretrainedModel.from_state_dict)
    assert Path.cwd() == tmp_path


@pytest.mark.parametrize("encoder_type", ["early_exit", "per_layer_activations"])
def test_real_hypernetwork_runs_generation_and_arbitration_on_tiny_cpu_model(
    monkeypatch, encoder_type
):
    """Exercise actual D2L/PEFT/Transformers code, using random local weights."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    from ctx_to_lora.configs import AggregatorArguments, CtxEncoderArguments, HypernetArguments
    from ctx_to_lora.modeling import hypernet
    from src.generation.d2l_backend import D2LBackend
    from tests.test_d2l_backend import Tokenizer

    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=None,
    )
    base = get_peft_model(
        LlamaForCausalLM(config),
        LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    encoder = LlamaForCausalLM(config)
    monkeypatch.setattr(hypernet, "get_model", lambda *a, **kw: encoder)
    encoder_args = CtxEncoderArguments(ctx_encoder_type=encoder_type, layer_idx=1)
    hyper_config = hypernet.get_hypernet_config(
        base,
        config,
        HypernetArguments(latent_size=8, per_rank_gen=True),
        AggregatorArguments(num_blocks=1, n_latent_queries=2),
        encoder_args,
    )
    model = ModulatedPretrainedModel(
        base, hyper_config, encoder_args, use_sequence_packing=False
    ).eval()
    # Make the random document adapter nonzero (upstream init sets B scale to zero).
    with torch.no_grad():
        for scale in model.hypernet.scaler_B.values():
            scale.fill_(0.1)
    engine = D2LBackend(
        model,
        Tokenizer(),
        Tokenizer(),
        lambda data, tok: {"ctx_ids": [[1, 5, 7, 3]]},
        "tiny-local-llama",
        max_input_tokens=1024,
    )
    result = engine.infer("question", [{"content": "evidence"}], max_new_tokens=2)
    assert result["answer"]
    assert result["routing"]["reason"] == "trustmargin"
    assert set(result["routing"]["likelihoods"]) == {"rag", "d2l"}
    assert model.generated_loras is None
    # A subsequent unadapted request can still generate after all hooks are reset.
    assert engine.infer("next", [{"content": "different"}], mode="rag", max_new_tokens=2)["answer"]
