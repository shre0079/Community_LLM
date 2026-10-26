import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

from shared.config import MODEL_ID, MODEL_CACHE_DIR, SIMULATE_VRAM_GB

log = logging.getLogger(__name__)


def _apply_vram_cap():
    if SIMULATE_VRAM_GB > 0 and torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        fraction = min(SIMULATE_VRAM_GB / total_gb, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        log.info(
            f"VRAM capped to {SIMULATE_VRAM_GB:.1f} GB "
            f"({fraction * 100:.1f}% of {total_gb:.1f} GB total)"
        )


class ShardRunner:
    """
    Loads the full OLMoE model but only executes the assigned layer range.
    Phase 2 will switch to downloading only the required weight shards.
    """

    def __init__(self, layer_start: int, layer_end: int, is_first: bool, is_last: bool):
        self.layer_start = layer_start
        self.layer_end   = layer_end
        self.is_first    = is_first
        self.is_last     = is_last
        self.model       = None
        self.tokenizer   = None
        self.device      = self._detect_device()

        # KV-cache for this worker's layer slice, scoped to the single
        # in-flight request (no concurrent sequences until Phase 4's
        # continuous batching). Reset at the start of every new request.
        self.kv_cache: DynamicCache | None = None

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(self):
        _apply_vram_cap()

        log.info(f"Device: {self.device}  layers: {self.layer_start}–{self.layer_end}")
        log.info("Loading tokenizer…")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE_DIR,
            trust_remote_code=True,
        )

        log.info("Loading model with 4-bit NF4 quantization…")
        bnb_cfg = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            if self.device == "cuda"
            else None
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            torch_dtype=torch.float16,
            device_map="auto" if self.device == "cuda" else None,
            cache_dir=MODEL_CACHE_DIR,
            trust_remote_code=True,
        )
        self.model.eval()
        log.info(f"Model loaded. VRAM used: {self.vram_used_gb():.2f} GB")

    # ── Forward ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        input_data: torch.Tensor,
        temperature: float = 1.0,
        reset_cache: bool = False,
    ) -> torch.Tensor:
        """
        input_data — only the NEW tokens/positions for this step, not the
        full sequence:
          is_first, reset_cache=True  (prefill) → full prompt token ids [1, prompt_len]
          is_first, reset_cache=False (decode)  → single new token id   [1, 1]
          otherwise                             → hidden states for the same
                                                   new positions [1, new_len, hidden_dim]
                                                   (float16, CPU)

        Past context for every position before this step lives in
        self.kv_cache (one DynamicCache per worker, scoped to the single
        in-flight request) and is extended in place by each layer call.

        Returns:
          is_last → next token id as scalar tensor (CPU)
          otherwise → hidden states for the new positions [1, new_len, hidden_dim] (float16, CPU)
        """
        m = self.model

        if reset_cache or self.kv_cache is None:
            self.kv_cache = DynamicCache()

        if self.is_first:
            hidden = m.model.embed_tokens(input_data.to(self.device))
        else:
            hidden = input_data.to(self.device).to(torch.float16)

        new_len      = hidden.shape[1]
        past_len     = self.kv_cache.get_seq_length(self.layer_start)
        position_ids = torch.arange(
            past_len, past_len + new_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        causal_mask = self._causal_mask(new_len, past_len, hidden.device, hidden.dtype)

        # Precompute rotary embeddings once if model exposes them at the top level.
        # Layer-level fallback: each OlmoeAttention computes its own from position_ids.
        position_embeddings = None
        if hasattr(m.model, "rotary_emb"):
            position_embeddings = m.model.rotary_emb(hidden, position_ids)

        for idx in range(self.layer_start, self.layer_end + 1):
            layer  = m.model.layers[idx]
            # NOTE: transformers>=5 (confirmed on 5.15.1) names this kwarg
            # `past_key_values` (plural). OlmoeDecoderLayer.forward() ends in
            # `**kwargs: Unpack[TransformersKwargs]`, a catch-all — passing the
            # wrong name here does NOT raise TypeError, it silently vanishes
            # into **kwargs and the layer runs with no cache at all. That bug
            # shipped once already (every decode step attended to nothing
            # before itself); do not reintroduce a bare name guess here.
            kwargs = dict(
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=self.kv_cache,
                cache_position=position_ids[0],
                use_cache=True,
                output_attentions=False,
                output_router_logits=False,   # suppress MoE router overhead
            )
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            out = layer(hidden, **kwargs)

            hidden = out[0] if isinstance(out, tuple) else out

        if self.is_last:
            hidden      = m.model.norm(hidden)
            logits      = m.lm_head(hidden)       # [1, new_len, vocab]
            last_logits = logits[0, -1, :]        # [vocab]  — last of the new positions
            token_id    = self._sample(last_logits, temperature)
            return torch.tensor(token_id, dtype=torch.long)
        else:
            return hidden.cpu().to(torch.float16)

    # ── Sampling ─────────────────────────────────────────────────────────────

    def _sample(self, logits: torch.Tensor, temperature: float) -> int:
        if temperature <= 0.0:
            return torch.argmax(logits).item()
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    # ── Causal mask ──────────────────────────────────────────────────────────

    @staticmethod
    def _causal_mask(new_len: int, past_len: int, device, dtype) -> torch.Tensor:
        """
        Mask for `new_len` query positions attending over `past_len` cached
        keys plus themselves: shape [1, 1, new_len, past_len + new_len].
        Past positions are always visible; the new-token block is causal
        (upper-triangular -inf) same as before, just offset by past_len.
        With past_len=0 (prefill / no cache) this is identical to the
        original full causal mask.
        """
        neg_inf   = torch.finfo(dtype).min
        total_len = past_len + new_len
        mask      = torch.zeros((new_len, total_len), device=device, dtype=dtype)
        new_block = torch.triu(
            torch.full((new_len, new_len), neg_inf, device=device, dtype=dtype),
            diagonal=1,
        )
        mask[:, past_len:] = new_block
        return mask.unsqueeze(0).unsqueeze(0)

    # ── Tokenizer helpers ────────────────────────────────────────────────────

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, return_tensors="pt")

    def decode(self, token_ids) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def vram_used_gb(self) -> float:
        return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0