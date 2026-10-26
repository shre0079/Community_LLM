# test_regression.py — formalized pytest regression suite (Phase 1
# requirement: "Deterministic regression tests where supported").
#
# Run on a machine that can load the full model:
#   pytest test_regression.py -v
#   pytest -m slow -v              (same thing, by marker)
#
# To exclude this in an environment without a GPU/the model downloaded
# (e.g. a lightweight CI runner):
#   pytest -m "not slow"
#
# single_node_baseline.py's CLI entry point (`python single_node_baseline.py`)
# still exists for human-readable diagnostic output (prints both token
# sequences and the exact divergence point on failure) — this file wraps
# the same check_cache_correctness() logic in a plain pytest assertion so
# it can run unattended / in CI, instead of relying on a human reading
# stdout and a SystemExit code.

import pytest

from single_node_baseline import (
    MAX_NEW_TOKENS,
    PROMPT,
    SEED,
    TEMPERATURE,
    check_cache_correctness,
)


@pytest.mark.slow
def test_kv_cache_matches_uncached_decode():
    """
    Greedy decoding is deterministic: KV-cached and uncached (full-sequence
    reprocess every step) decoding of the same fixed prompt/seed MUST
    produce the exact same token sequence. This is the correctness gate
    that caught the real past_key_value(s) kwarg-name bug during
    development (see worker/model.py) — it should stay green any time the
    KV-cache implementation changes, and is also the source of the
    reference tokens the 2-machine pipeline integration test is checked
    against (see single_node_baseline.py's REFERENCE_TOKENS output).
    """
    result = check_cache_correctness(
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        seed=SEED,
    )
    assert result["match"], (
        f"Cached and uncached decoding diverged at step "
        f"{result['first_divergence_step']}.\n"
        f"cached:   {result['cached_tokens']}\n"
        f"uncached: {result['uncached_tokens']}"
    )
