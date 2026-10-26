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


def check_cache_correctness(
    prompt: str = PROMPT,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    seed: int = SEED,
) -> dict:
    """
    Loads the full model once and runs both the cached and uncached decode
    paths, returning a result dict rather than printing/exiting — so this
    same check can be driven either by main() below (human-readable CLI
    output) or by test_regression.py's pytest test (plain assertion).

    Returns:
      {
        "cached_tokens": [...], "uncached_tokens": [...],
        "cached_text": str, "uncached_text": str,
        "match": bool,
        "first_divergence_step": int | None,
      }
    """
    torch.manual_seed(seed)
    runner = ShardRunner(layer_start=0, layer_end=15, is_first=True, is_last=True)
    runner.load()

    torch.manual_seed(seed)
    cached_tokens = greedy_decode(runner, prompt, max_new_tokens, use_cache=True)

    torch.manual_seed(seed)
    uncached_tokens = greedy_decode(runner, prompt, max_new_tokens, use_cache=False)

    match = cached_tokens == uncached_tokens
    first_divergence_step = None
    if not match:
        min_len = min(len(cached_tokens), len(uncached_tokens))
        for i in range(min_len):
            if cached_tokens[i] != uncached_tokens[i]:
                first_divergence_step = i
                break

    return {
        "cached_tokens":          cached_tokens,
        "uncached_tokens":        uncached_tokens,
        "cached_text":            runner.decode(cached_tokens),
        "uncached_text":          runner.decode(uncached_tokens),
        "match":                  match,
        "first_divergence_step":  first_divergence_step,
    }


def main():
    print(f"Loading full model…")
    print(f"\nPrompt: {PROMPT!r}  max_new_tokens={MAX_NEW_TOKENS}  temperature={TEMPERATURE}")

    result = check_cache_correctness()

    print("\n[1/2] Cached decode (prefill once, then single-token steps)…")
    print("Cached tokens:  ", result["cached_tokens"])
    print("Cached text:    ", repr(result["cached_text"]))

    print("\n[2/2] Uncached decode (full-sequence reprocess every step, old behavior)…")
    print("Uncached tokens:", result["uncached_tokens"])
    print("Uncached text:  ", repr(result["uncached_text"]))

    print("\n=== Result ===")
    if result["match"]:
        print("PASS — cached and uncached decoding produced identical token sequences.")
        print(f"\nREFERENCE_TOKENS for the 2-machine integration test:\n{result['cached_tokens']}")
    else:
        print("FAIL — cached and uncached decoding DIVERGED. Do not trust the KV-cache path.")
        print(
            f"First divergence at step {result['first_divergence_step']}: "
            f"cached={result['cached_tokens'][result['first_divergence_step']]} "
            f"uncached={result['uncached_tokens'][result['first_divergence_step']]}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
