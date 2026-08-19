import gc
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from shared.config import MODEL_ID, MODEL_CACHE_DIR

log = logging.getLogger(__name__)


class ShardRunner:
    def __init__(self, layer_start: int, layer_end: int, is_first: bool, is_last: bool):
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.is_first = is_first
        self.is_last = is_last
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        log.info(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE_DIR,
            trust_remote_code=True,
        )

        log.info(f"Loading model with 4-bit quantization on {self.device}...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load full model — each worker executes only its assigned layers
        # Phase 2 will implement partial shard downloading
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb if self.device == "cuda" else None,
            torch_dtype=torch.float16,
            device_map="auto" if self.device == "cuda" else "cpu",
            cache_dir=MODEL_CACHE_DIR,
            trust_remote_code=True,
        )
        self.model.eval()

        vram = torch.cuda.memory_allocated() / 1e9 if self.device == "cuda" else 0
        log.info(f"Model loaded. VRAM used: {vram:.2f} GB")
        log.info(f"This worker owns layers {self.layer_start}–{self.layer_end}")

    @torch.no_grad()
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        input_data:
          - If is_first: token ids [1, seq_len]
          - Otherwise: hidden states [1, seq_len, hidden_dim]
        Returns:
          - If is_last: next token id (scalar tensor)
          - Otherwise: hidden states [1, seq_len, hidden_dim]
        """
        m = self.model

        if self.is_first:
            input_ids = input_data.to(self.device)
            hidden = m.model.embed_tokens(input_ids)
        else:
            hidden = input_data.to(self.device).to(torch.float16)

        seq_len = hidden.shape[1]
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
        causal_mask = self._causal_mask(seq_len, hidden.device, hidden.dtype)

        for idx in range(self.layer_start, self.layer_end + 1):
            layer = m.model.layers[idx]
            out = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=pos_ids,
            )
            # All HF decoder layers return a tuple; index 0 is always hidden states
            hidden = out[0] if isinstance(out, tuple) else out

        if self.is_last:
            hidden = m.model.norm(hidden)
            logits = m.lm_head(hidden)           # [1, seq_len, vocab_size]
            last_logits = logits[0, -1, :]        # [vocab_size]
            next_token = torch.argmax(last_logits) # greedy; temperature sampling in Day 4
            return next_token
        else:
            return hidden.cpu()  # send over network in float16 on CPU

    def _causal_mask(self, seq_len: int, device, dtype) -> torch.Tensor:
        """Upper-triangular mask for causal attention."""
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, return_tensors="pt")

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def vram_used_gb(self) -> float:
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / 1e9
        return 0.0