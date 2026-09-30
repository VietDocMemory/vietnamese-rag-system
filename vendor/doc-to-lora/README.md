# Vendored Doc-to-LoRA inference runtime

Source: https://github.com/SakanaAI/doc-to-lora

Copied from the local `D:/doc-to-lora` checkout at commit
`90df54e3b51b552987b700a004ef633b8139a2de` (MIT; see LICENSE).
Includes the transitive local imports of `ctx_to_lora.modeling.hypernet` and the
upstream chat templates. Training entrypoints, demos and optional training/serving
stacks (DeepSpeed, vLLM, Gradio, etc.) are not needed for this inference package.
The `ctx_to_lora` module namespace is preserved for checkpoint deserialization.

Local changes:
- Minimal package metadata for inference dependencies, linked via `[tool.uv.sources]`.
- `model_loading.get_tokenizer` reads packaged templates relative to the module,
  so the API does not change working directory or depend on a sibling checkout.
- The Perceiver uses PyTorch SDPA for non-packed inference. Projection names and
  shapes, context-only cross-attention and latent-only self-attention match the
  upstream Flash Attention implementation. Packed training is not supported by
  this SDPA path. The legacy eager attention is not used because its concatenated
  context/latent keys differ from the pretrained Flash Attention path.

This is source integration, not a remote API or a worker process. The RAG API
imports and runs this package directly in its own Python environment.
