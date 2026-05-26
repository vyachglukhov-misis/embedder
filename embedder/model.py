import math
import os
from pathlib import Path

import torch

# ─── WORKAROUND ─────────────────────────────────────────────────────────────
# В свежих transformers (>=4.49) AutoModel.from_pretrained проталкивает
# kwarg `dtype` в `cls.__init__` через **model_kwargs, но XLMRobertaModel
# его в сигнатуре не объявляет — падает TypeError. Накрываем монки-патчем
# до того, как FlagEmbedding попробует поднять bge-m3.
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel

_orig_xlmr_init = XLMRobertaModel.__init__

def _patched_xlmr_init(self, *args, **kwargs):
    kwargs.pop('dtype', None)
    return _orig_xlmr_init(self, *args, **kwargs)

XLMRobertaModel.__init__ = _patched_xlmr_init
# ────────────────────────────────────────────────────────────────────────────

from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE_DIR = Path(os.getenv("HF_HOME", "./model_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)

_model = None
_reranker = None


def get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        print("Загружаем bge-m3 на GPU...")
        _model = BGEM3FlagModel(
            'BAAI/bge-m3',
            use_fp16=True,
            device='cuda',
        )
        print("Модель готова")
    return _model


class GemmaReranker:
    """
    Прямая реализация bge-reranker-v2-gemma через transformers — обходит баг
    FlagLLMReranker.compute_score с tokenizer.prepare_for_model в свежих
    transformers (>=4.50, где этот метод был удалён).

    Модель — Gemma-2B, дообученная как cross-encoder. Скоринг: конструируется
    промпт «A: query / B: passage / Given a query A and a passage B,
    determine whether the passage contains an answer ... Yes or No», и берётся
    логит токена 'Yes' на последней позиции. Sigmoid → score в [0, 1].

    Контракт `compute_score(pairs, normalize=True) -> list[float]` совпадает
    с FlagReranker/FlagLLMReranker — drop-in замена для main.py.
    """

    SUFFIX_TEXT = (
        "Given a query A and a passage B, determine whether the passage contains "
        "an answer to the query by providing a prediction of either 'Yes' or 'No'."
    )

    def __init__(
        self,
        model_name: str = 'BAAI/bge-reranker-v2-gemma',
        max_length: int = 1024,
        batch_size: int = 8,
    ):
        print(f"Загружаем {model_name} на GPU (manual)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            # У Gemma нет pad_token — переиспользуем eos
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to('cuda').eval()

        self.max_length = max_length
        self.batch_size = batch_size

        # Предсчитываем suffix-промпт и id токена 'Yes' один раз
        self.suffix_ids: list[int] = self.tokenizer(
            self.SUFFIX_TEXT, add_special_tokens=False
        )['input_ids']
        self.yes_token_id: int = self.tokenizer(
            'Yes', add_special_tokens=False
        )['input_ids'][0]

        print("Реранкер готов")

    @torch.inference_mode()
    def compute_score(self, pairs, normalize: bool = True):
        """
        pairs: list[[query, passage]]. Поддерживается и одиночная пара [q, p].
        normalize: True → sigmoid(logit) ∈ [0, 1]; False → сырой логит.
        return: list[float] длины len(pairs).
        """
        if not pairs:
            return []
        # Одиночная пара [q, p] — оборачиваем, как FlagReranker
        if not isinstance(pairs[0], (list, tuple)):
            pairs = [pairs]

        all_scores: list[float] = []

        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i:i + self.batch_size]

            # Конструируем input_ids вручную: bos + 'A: query\nB: passage\n' + suffix.
            # Это нужно, чтобы suffix-промпт НЕ отрезался при truncation —
            # последний токен суффикса является сигналом для логита 'Yes'.
            seq_ids: list[list[int]] = []
            for query, passage in chunk:
                prefix_text = f"A: {query}\nB: {passage}\n"
                # бюджет для prefix = max_length − bos − suffix
                budget = self.max_length - 1 - len(self.suffix_ids)
                prefix_ids = self.tokenizer(
                    prefix_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max(budget, 1),
                )['input_ids']
                ids = [self.tokenizer.bos_token_id] + prefix_ids + self.suffix_ids
                seq_ids.append(ids)

            # Left-padding руками: pad слева, чтобы последний токен суффикса
            # у всех примеров был на позиции -1 в логитах.
            target_len = max(len(s) for s in seq_ids)
            pad_id = self.tokenizer.pad_token_id
            padded_ids: list[list[int]] = []
            attn_mask: list[list[int]] = []
            for s in seq_ids:
                n_pad = target_len - len(s)
                padded_ids.append([pad_id] * n_pad + s)
                attn_mask.append([0] * n_pad + [1] * len(s))

            input_ids = torch.tensor(padded_ids, dtype=torch.long, device='cuda')
            attention_mask = torch.tensor(attn_mask, dtype=torch.long, device='cuda')

            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            # Последний токен на позиции -1 (left-padded), его лог-распределение
            last_logits = logits[:, -1, :].float()
            yes_logits = last_logits[:, self.yes_token_id]
            all_scores.extend(yes_logits.cpu().tolist())

        if normalize:
            all_scores = [1.0 / (1.0 + math.exp(-s)) for s in all_scores]
        return all_scores


def get_reranker() -> GemmaReranker:
    global _reranker
    if _reranker is None:
        _reranker = GemmaReranker(
            model_name='BAAI/bge-reranker-v2-gemma',
            max_length=1024,
            batch_size=8,
        )
    return _reranker
