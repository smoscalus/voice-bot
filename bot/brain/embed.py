"""«Отпечатки смысла» (эмбеддинги) для поиска по памяти: какие воспоминания подходят к разговору.

Маленькая многоязычная модель (~120 МБ, скачивается сама при первом запуске), работает на процессоре
за миллисекунды и не тратит лимиты. Пока модель грузится или если не загрузилась — память ищется
просто по словам.
"""
from __future__ import annotations

import logging
import os
import threading

import numpy as np

log = logging.getLogger(__name__)

REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ONNX_FILE = "onnx/model_quint8_avx2.onnx"


class Embedder:
    def __init__(self) -> None:
        self.ready = False
        self._session = None
        self._tok = None
        self._inputs: list[str] = []
        self._lock = threading.Lock()
        threading.Thread(target=self._load, name="embedder-load", daemon=True).start()

    def _load(self) -> None:
        try:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
            tok.enable_truncation(128)
            tok.enable_padding()
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.log_severity_level = 3
            sess = ort.InferenceSession(hf_hub_download(REPO, ONNX_FILE), opts, providers=["CPUExecutionProvider"])
            self._tok, self._session = tok, sess
            self._inputs = [i.name for i in sess.get_inputs()]
            self.ready = True
            log.info("поиск по смыслу готов")
        except Exception as e:
            log.warning("поиск по смыслу недоступен (буду искать по словам): %s", e)

    def encode(self, texts: list[str]) -> np.ndarray | None:
        if not self.ready or not texts:
            return None
        with self._lock:
            enc = self._tok.encode_batch(texts)
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            out = self._session.run(None, feed)[0]
        v = (out * mask[..., None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


_EMB: Embedder | None = None


def get_embedder() -> Embedder:
    global _EMB
    if _EMB is None:
        _EMB = Embedder()
    return _EMB
