"""Hugging Face implementations of ``Fluency`` and ``Relevance``.

They need the ``lm`` extra (torch, transformers, sentence-transformers); models are
downloaded from the Hugging Face Hub on first use. A CUDA GPU is used when available.
Gated models (e.g. ``google/embeddinggemma-300m``) need ``HF_TOKEN`` and an accepted license.
"""

from collections.abc import Sequence

DEFAULT_FLUENCY_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_RELEVANCE_MODEL = "Snowflake/snowflake-arctic-embed-m-v1.5"
MISSING_EXTRA = "neural ranking needs the lm extra: pip install 'granama[lm]'"


def _torch():
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ImportError(MISSING_EXTRA) from exc
    transformers.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    return torch


def _device(torch, device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


class CausalLMFluency:
    """Fluency as log P(phrase + ".") under a causal language model, or with ``capitalize``
    as log P(Phrase): "Old west action" rather than "old west action." (on a set of known
    word orders, this picked the right order more often: 75% against 64%)."""

    def __init__(
        self,
        model: str = DEFAULT_FLUENCY_MODEL,
        device: str | None = None,
        batch_tokens: int = 20_000,
        capitalize: bool = False,
    ):
        torch = self._torch = _torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = _device(torch, device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device)
        self.model.eval()
        bos = self.tokenizer.bos_token_id
        self.start = [self.tokenizer.eos_token_id if bos is None else bos]
        self.period = self._encode(".")[-1]
        self.batch_tokens = batch_tokens  # lower it on out-of-memory errors
        self.capitalize = capitalize

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def log_prob(
        self, phrases: Sequence[str], context: str = "", complete: bool = True
    ) -> list[float]:
        torch = self._torch
        if not phrases:
            return []
        prefix = self.start + (self._encode(context) if context else [])
        if self.capitalize:
            phrases = [p[:1].upper() + p[1:] for p in phrases]
        encoded = self.tokenizer(list(phrases), add_special_tokens=False).input_ids
        end = [self.period] if complete and not self.capitalize else []
        ids = [[*seq, *end] for seq in encoded]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out = [0.0] * len(ids)
        head = self.model.get_output_embeddings()
        start = 0
        while start < len(order):
            # lengths are sorted, so the batch's last sequence is its longest
            stop = start + 1
            while (
                stop < len(order)
                and (stop - start + 1) * (len(prefix) + len(ids[order[stop]])) <= self.batch_tokens
            ):
                stop += 1
            batch = order[start:stop]
            width = len(prefix) + max(len(ids[i]) for i in batch)
            tokens = torch.zeros((len(batch), width), dtype=torch.long)
            attend = torch.zeros((len(batch), width), dtype=torch.long)
            scored = torch.zeros((len(batch), width), dtype=torch.bool)
            for row, i in enumerate(batch):
                seq = prefix + ids[i]
                tokens[row, : len(seq)] = torch.tensor(seq)
                attend[row, : len(seq)] = 1
                scored[row, len(prefix) : len(seq)] = True
            tokens, attend, scored = (t.to(self.device) for t in (tokens, attend, scored))
            with torch.no_grad():
                hidden = self.model.base_model(
                    input_ids=tokens, attention_mask=attend
                ).last_hidden_state
                # the hidden state at position t predicts token t + 1; project only the
                # scored positions, in chunks, since the vocabulary can be large
                mask = scored[:, 1:]
                states, targets = hidden[:, :-1][mask], tokens[:, 1:][mask]
                token_lp = torch.empty(len(states), device=self.device)
                for c in range(0, len(states), 1024):
                    chunk = slice(c, c + 1024)
                    logits = head(states[chunk]).float()
                    target_logits = logits.gather(1, targets[chunk, None])[:, 0]
                    token_lp[chunk] = target_logits - logits.logsumexp(-1)
                per_row = torch.zeros(mask.shape, device=self.device)
                per_row[mask] = token_lp
                for i, lp in zip(batch, per_row.sum(1).tolist(), strict=True):
                    out[i] = lp
            start = stop
        return out


class EmbeddingRelevance:
    """Relevance as the cosine similarity of sentence-transformers embeddings."""

    def __init__(
        self,
        model: str = DEFAULT_RELEVANCE_MODEL,
        device: str | None = None,
        prompt: str | None = None,
        half: bool = True,
        batch_size: int = 1024,
    ):
        """``prompt`` is one of the model's prompt names (e.g. "STS" for EmbeddingGemma);
        ``half`` uses float16 on GPU (turn it off for models that overflow, like Gemma)."""
        torch = _torch()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(MISSING_EXTRA) from exc

        self.device = _device(torch, device)
        self.model = SentenceTransformer(model, device=self.device, trust_remote_code=True)
        if half and self.device == "cuda":
            self.model.half()
        self.options = {
            "normalize_embeddings": True,
            "prompt_name": prompt,
            "batch_size": batch_size,
            "show_progress_bar": False,
        }

    def similarity(self, source: str, phrases: Sequence[str]) -> list[float]:
        if not phrases:
            return []
        source_embedding = self.model.encode([source], **self.options)[0]
        embeddings = self.model.encode(list(phrases), **self.options)
        return (embeddings @ source_embedding).tolist()
