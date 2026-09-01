import logging
import os
import re
import shutil
import threading
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _has_sklearn = True
except Exception:
    _has_sklearn = False

try:
    from sentence_transformers import CrossEncoder
    _has_cross = True
except Exception:
    _has_cross = False

CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", "./chroma_db"))
COLLECTION = "documents"
MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = 700
CHUNK_OVERLAP = 80

_embedding_model = None
_cross_model = None

_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                CHROMA_DIR.mkdir(parents=True, exist_ok=True)
                try:
                    os.chmod(CHROMA_DIR, 0o777)
                except Exception:
                    pass
                _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def _reset_client(hard: bool = False):
    global _client
    with _client_lock:
        _client = None
        if hard:
            for child in CHROMA_DIR.glob("*"):
                try:
                    if child.is_file():
                        child.unlink()
                    else:
                        shutil.rmtree(child)
                except Exception:
                    pass
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(CHROMA_DIR, 0o777)
            except Exception:
                pass


def _collection():
    for attempt in range(2):
        try:
            client = _get_client()
            return client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
        except Exception as e:
            msg = str(e).lower()
            is_corrupt = "no such table" in msg or "acquire_write" in msg
            is_readonly = "readonly" in msg or "attempt to write a readonly" in msg
            if is_corrupt or is_readonly:
                if is_readonly:
                    for p in CHROMA_DIR.rglob("*"):
                        try:
                            os.chmod(p, 0o666)
                        except Exception:
                            pass
                    try:
                        os.chmod(CHROMA_DIR, 0o777)
                    except Exception:
                        pass
                _reset_client(hard=is_corrupt)
                if attempt == 0:
                    continue
            raise


def _with_retry(fn, *args, **kwargs):
    for attempt in range(2):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            if ("no such table" in msg or "acquire_write" in msg or "readonly" in msg) and attempt == 0:
                _reset_client(hard="no such table" in msg or "acquire_write" in msg)
                continue
            raise


def clear_all():
    for attempt in range(2):
        try:
            client = _get_client()
            try:
                client.reset()
            except Exception:
                try:
                    client.delete_collection(COLLECTION)
                except Exception:
                    pass
            client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
            return
        except Exception as e:
            msg = str(e).lower()
            if ("no such table" in msg or "acquire_write" in msg) and attempt == 0:
                _reset_client(hard=True)
                continue
            if "readonly" in msg and attempt == 0:
                for p in CHROMA_DIR.rglob("*"):
                    try:
                        os.chmod(p, 0o666)
                    except Exception:
                        pass
                try:
                    os.chmod(CHROMA_DIR, 0o777)
                except Exception:
                    pass
                _reset_client(hard=False)
                continue
            raise


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(MODEL_NAME)
    return _embedding_model


def _get_cross():
    global _cross_model
    if not _has_cross:
        return None
    if _cross_model is None:
        try:
            _cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            return None
    return _cross_model


def _ollama_client():
    import ollama
    key = os.environ.get("OLLAMA_API_KEY", "")
    if key:
        return ollama.Client(host=os.environ.get("OLLAMA_CLOUD_HOST", "https://ollama.com"),
                             headers={"Authorization": f"Bearer {key}"}, timeout=300)
    return ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"), timeout=600)


def _ollama_model():
    m = os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud")
    if os.environ.get("OLLAMA_API_KEY"):
        return re.sub(r"[-:]cloud$", "", m) or m
    return m


def extract_text(pdf_bytes: bytes) -> str:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages = [p.get_text() or "" for p in doc]
    finally:
        doc.close()
    raw = "\n".join(pages)
    raw = re.sub(r"\r\n", "\n", raw)
    lines = []
    for line in raw.split("\n"):
        s = line.strip()
        if re.match(r"^Top 200 Data Analytics Interview Q&A\s*(\|\s*Page.*)?$", s):
            continue
        if re.match(r"^Page \d+(\s*/\s*\d+)?$", s):
            continue
        if s == "�":
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"�", "-", cleaned)
    return cleaned.strip()


def _question_chunks(text: str):
    parts = re.split(r"\n(?=\d{1,3}\.\s+[A-Z])", text)
    parts = [p.strip() for p in parts if p.strip()]
    q_parts = [p for p in parts if re.match(r"^\d{1,3}\.\s", p)]
    if len(q_parts) < 10:
        return None
    chunks = []
    for part in parts:
        if not re.match(r"^\d{1,3}\.\s", part):
            if len(part) < 40:
                continue
            if chunks and len(part) < 120:
                continue
            chunks.append(part)
            continue
        if len(part) <= CHUNK_SIZE:
            chunks.append(part)
        else:
            cur = part
            while len(cur) > CHUNK_SIZE:
                cut = cur.rfind(". ", 0, CHUNK_SIZE)
                if cut == -1:
                    cut = cur.rfind(" ", 0, CHUNK_SIZE)
                if cut == -1:
                    cut = CHUNK_SIZE
                else:
                    cut += 1
                chunks.append(cur[:cut].strip())
                cur = cur[cut:].strip()
            if cur:
                chunks.append(cur)
    return chunks


def _recursive_chunks(text: str, size: int, overlap: int):
    seps = ["\n\n", "\n", ". ", " "]

    def split_rec(t: str, idx: int):
        if len(t) <= size:
            return [t]
        if idx >= len(seps):
            return [t[i:i + size] for i in range(0, len(t), size - overlap)]
        sep = seps[idx]
        parts = t.split(sep)
        out = []
        cur = ""
        for p in parts:
            cand = (cur + sep + p).strip() if cur else p
            if len(cand) <= size:
                cur = cand
            else:
                if cur:
                    out.extend(split_rec(cur, idx + 1))
                    cur = p
                else:
                    out.extend(split_rec(p, idx + 1))
        if cur:
            out.extend(split_rec(cur, idx + 1))
        return out

    raw = split_rec(text, 0)
    merged = []
    for r in raw:
        r = r.strip()
        if r and len(r) >= 60 and (not merged or r != merged[-1]):
            merged.append(r)
    return merged


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP, strategy: str = "auto"):
    text = text.strip()
    if not text:
        return []
    if strategy == "fixed":
        text = re.sub(r"\s+\n", "\n", text)
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            if end < n and not text[end].isspace() and not text[end - 1].isspace():
                sp = text.rfind(" ", start, end)
                if sp > start + size * 0.5:
                    end = sp
            piece = text[start:end].strip()
            if piece and (not chunks or piece != chunks[-1]):
                chunks.append(piece)
            if end >= n:
                break
            nxt = end - overlap
            if nxt <= start:
                nxt = start + max(1, size - overlap)
            start = nxt
        return [c for c in chunks if len(c) >= 60]
    if strategy == "recursive":
        return _recursive_chunks(text, size, overlap)
    qc = _question_chunks(text)
    if qc is not None:
        return [c for c in qc if len(c) >= 60]
    return chunk_text(text, size, overlap, strategy="fixed")


def _remove_existing_source(source: str):
    def _do():
        col = _collection()
        existing = col.get(where={"source": source}, include=["metadatas"])
        ids = existing.get("ids") or []
        if ids:
            col.delete(ids=ids)
    try:
        _with_retry(_do)
    except Exception:
        pass


def add_document(filename: str, pdf_bytes: bytes, strategy: str = "auto") -> int:
    text = extract_text(pdf_bytes)
    chunks = chunk_text(text, strategy=strategy)
    if not chunks:
        return 0
    _remove_existing_source(filename)
    emb = _get_embedding_model().encode(chunks, show_progress_bar=False).tolist()
    doc_id = str(uuid4())
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metas = [{"source": filename, "doc_id": doc_id, "index": i} for i in range(len(chunks))]

    def _do_add():
        _collection().add(ids=ids, documents=chunks, embeddings=emb, metadatas=metas)

    _with_retry(_do_add)
    return len(chunks)


def add_text_document(filename: str, text: str, strategy: str = "auto") -> int:
    text = text.strip()
    if not text:
        return 0
    chunks = chunk_text(text, strategy=strategy)
    if not chunks:
        return 0
    _remove_existing_source(filename)
    emb = _get_embedding_model().encode(chunks, show_progress_bar=False).tolist()
    doc_id = str(uuid4())
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metas = [{"source": filename, "doc_id": doc_id, "index": i} for i in range(len(chunks))]

    def _do_add():
        _collection().add(ids=ids, documents=chunks, embeddings=emb, metadatas=metas)

    _with_retry(_do_add)
    return len(chunks)


def _bm25_scores(query: str, docs: list[str]):
    if not _has_sklearn or not docs:
        return [0.0] * len(docs)
    try:
        vec = TfidfVectorizer().fit(docs + [query])
        dm = vec.transform(docs)
        qv = vec.transform([query])
        return cosine_similarity(qv, dm)[0].tolist()
    except Exception:
        return [0.0] * len(docs)


def _dedup(cands: list[dict]):
    seen = set()
    out = []
    for c in cands:
        key = c["text"].strip()[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def query(question: str, top_k: int = 3, use_hybrid: bool = False, use_rerank: bool = False, filters: dict | None = None, use_hyde: bool = False):
    def _do_query():
        col = _collection()
        if col.count() == 0:
            return []
        where = {"source": filters["source"]} if filters and filters.get("source") else None
        effective_q = question
        if use_hyde:
            try:
                hr = _ollama_client().generate(
                    model=_ollama_model(),
                    prompt=f"Write a short paragraph that would answer: {question}\nParagraph:",
                    options={"temperature": 0.3, "num_predict": 120},
                )
                hypo = hr.get("response", "").strip()
                if len(hypo) > 20:
                    effective_q = hypo
            except Exception:
                pass
        q_emb = _get_embedding_model().encode([effective_q], show_progress_bar=False).tolist()
        n_fetch = min(top_k * 4 if use_hybrid or use_rerank else top_k, col.count())
        res = col.query(query_embeddings=q_emb, n_results=n_fetch, where=where)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        cands = [{"text": d, "source": m.get("source"), "score": float(s), "meta": m} for d, m, s in zip(docs, metas, dists)]
        cands = _dedup(cands)
        if use_hybrid and _has_sklearn:
            try:
                all_data = col.get(include=["documents"], where=where)
                all_docs = all_data.get("documents") or []
                bm = _bm25_scores(effective_q, all_docs)
                bm_rank = sorted(range(len(bm)), key=lambda i: bm[i], reverse=True)[:n_fetch]
                doc_to_bm_rank = {all_docs[i]: r for r, i in enumerate(bm_rank)}
                vec_order = sorted(range(len(cands)), key=lambda i: cands[i]["score"])
                bm_order_scores = []
                for c in cands:
                    bm_order_scores.append(doc_to_bm_rank.get(c["text"], 999))
                bm_order = sorted(range(len(cands)), key=lambda i: bm_order_scores[i])
                rrf = {}
                for lst in (vec_order, bm_order):
                    for rank, idx in enumerate(lst):
                        rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank + 1)
                cands = sorted(cands, key=lambda c: rrf[cands.index(c)], reverse=True)
                cands = _dedup(cands)
            except Exception:
                pass
        if use_rerank:
            ce = _get_cross()
            if ce is not None:
                try:
                    pairs = [(question, c["text"]) for c in cands]
                    scores = ce.predict(pairs)
                    for c, s in zip(cands, scores):
                        c["score"] = float(s)
                    cands.sort(key=lambda x: x["score"], reverse=True)
                    cands = _dedup(cands)
                except Exception:
                    pass
        return cands[:top_k]

    return _with_retry(_do_query)


def get_overview(limit: int = 2):
    def _do():
        col = _collection()
        if col.count() == 0:
            return []
        data = col.get(limit=limit, include=["documents", "metadatas"])
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        return [{"text": d, "source": m.get("source"), "score": 0.0} for d, m in zip(docs, metas)]

    return _with_retry(_do)


def list_documents():
    def _do():
        col = _collection()
        if col.count() == 0:
            return []
        data = col.get(include=["metadatas"])
        metas = data.get("metadatas") or []
        seen = {}
        for m in metas:
            src = m.get("source", "unknown")
            seen[src] = seen.get(src, 0) + 1
        return [{"source": k, "chunks": v} for k, v in seen.items()]

    return _with_retry(_do)


def document_count() -> int:
    def _do():
        return _collection().count()

    return _with_retry(_do)


def evaluate(question: str, answer: str, sources: list[dict]):
    if not sources:
        return {"faithfulness": 0, "relevance": 0, "citation_count": 0}
    try:
        import numpy as np
        q_emb = _get_embedding_model().encode([question])[0]
        a_emb = _get_embedding_model().encode([answer])[0] if answer else q_emb * 0
        rel = float(np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-9))
        src_text = " ".join(s["text"] for s in sources[:3])
        s_emb = _get_embedding_model().encode([src_text])[0]
        faith = float(np.dot(a_emb, s_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(s_emb) + 1e-9))
        return {"faithfulness": round(max(0, faith), 3), "relevance": round(max(0, rel), 3), "citation_count": len(sources)}
    except Exception:
        return {"faithfulness": 0.5, "relevance": 0.5, "citation_count": len(sources)}
