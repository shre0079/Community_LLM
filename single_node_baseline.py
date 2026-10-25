# single_node_baseline.py
#
# Correctness anchor for KV-caching (Phase 1 → Phase 2 gate).
#
# Runs the FULL model (all 16 layers, is_first=is_last=True — no sharding,
# no network) on ONE machine, greedy decoding a fixed prompt twice:
#   1. with the KV-cache path (reset_cache only on step 0 — the normal,
#      fast path every worker now uses)
#   2. forcing reset_cache=True on every step, which re-embeds/reprocesses
#      the full growing sequence each time (the old, pre-cache behavior)
#
# Greedy decoding is deterministic, so both paths MUST produce the exact
# same token sequence. If they diverge, the cache implementation has a
# correctness bug (wrong position_ids, wrong mask offset, wrong layer_idx
# in the cache, etc.) — do not trust pipeline output until this passes.
#
# This also becomes the reference token sequence for the 2-machine
# pipeline integration test: run the same PROMPT/SEED/MAX_NEW_TOKENS
# through POST /infer on the real coordinator and diff against
# REFERENCE_TOKENS below. Pipeline output must match exactly.
#
# Run with: python single_node_baseline.py
# Requires a machine that can load the full model (GPU strongly
# recommended — this is 16 layers, not a 5-6 layer shard).

import torch

from worker.model import ShardRunner

PROMPT          = "The capital of France is"
MAX_NEW_TOKENS  = 20
TEMPERATURE     = 0.0   # greedy — required for determinism/comparability
SEED            = 1234


def greedy_decode(runner: ShardRunner, prompt: str, max_new_tokens: int, use_cache: bool) -> list[int]:
    """
    Mirrors worker/main.py::_generate's loop, but single-process /
    single-worker (no ZMQ hop): the SAME ShardRunner instance handles
    embedding, all 16 layers, and the output head every step.
    """
    input_ids   = runner.encode(prompt)
    next_input  = input_ids
    tokens: list[int] = []

    for step in range(max_new_tokens):
        reset_cache = (step == 0) if use_cache else True
        token = runner.forward(next_input, temperature=TEMPERATURE, reset_cache=reset_cache)
        token_id = token.item()
        tokens.append(token_id)

        if token_id == runner.eos_token_id:
            break

        if use_cache:
            next_input = torch.tensor([[token_id]], dtype=torch.long)
        else:
            # Uncached path: re-feed the WHOLE sequence so reset_cache=True
            # each step genuinely recomputes from scratch, matching the
            # pre-KV-cache behavior exactly.
            next_input = torch.cat([input_ids, torch.tensor([tokens], dtype=torch.long)], dim=1)

    return tokens


def main():
    torch.manual_seed(SEED)

    runner = ShardRunner(layer_start=0, layer_end=15, is_first=True, is_last=True)
    print(f"Loading full model on {runner._detect_device()}…")
    runner.load()

    print(f"\nPrompt: {PROMPT!r}  max_new_tokens={MAX_NEW_TOKENS}  temperature={TEMPERATURE}")

    print("\n[1/2] Cached decode (prefill once, then single-token steps)…")
    torch.manual_seed(SEED)
    cached_tokens = greedy_decode(runner, PROMPT, MAX_NEW_TOKENS, use_cache=True)
    print("Cached tokens:  ", cached_tokens)
    print("Cached text:    ", repr(runner.decode(cached_tokens)))

    print("\n[2/2] Uncached decode (full-sequence reprocess every step, old behavior)…")
    torch.manual_seed(SEED)
    uncached_tokens = greedy_decode(runner, PROMPT, MAX_NEW_TOKENS, use_cache=False)
    print("Uncached tokens:", uncached_tokens)
    print("Uncached text:  ", repr(runner.decode(uncached_tokens)))

    print("\n=== Result ===")
    if cached_tokens == uncached_tokens:
        print("PASS — cached and uncached decoding produced identical token sequences.")
        print(f"\nREFERENCE_TOKENS for the 2-machine integration test:\n{cached_tokens}")
    else:
        print("FAIL — cached and uncached decoding DIVERGED. Do not trust the KV-cache path.")
        min_len = min(len(cached_tokens), len(uncached_tokens))
        for i in range(min_len):
            if cached_tokens[i] != uncached_tokens[i]:
                print(f"First divergence at step {i}: cached={cached_tokens[i]} uncached={uncached_tokens[i]}")
                break
        raise SystemExit(1)


if __name__ == "__main__":
    main()
