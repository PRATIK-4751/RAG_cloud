import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

# prefer a local .env (gitignored) so the API key never lives in source
load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import rag
import ollama

app = FastAPI(title="Local RAG")

# Ollama Cloud (hosted) is used when OLLAMA_API_KEY is set, else local Ollama.
CLOUD_HOST = os.environ.get("OLLAMA_CLOUD_HOST", "https://ollama.com")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
LOCAL_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud")
# the cloud catalog names the model without the "-cloud"/":cloud" qualifier
CLOUD_MODEL = re.sub(r"[-:]cloud$", "", OLLAMA_MODEL) or OLLAMA_MODEL
FALLBACK_MODELS = [OLLAMA_MODEL, "gemma4:31b-cloud"]


def _ollama():
    if OLLAMA_API_KEY:
        return ollama.Client(host=CLOUD_HOST, headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}, timeout=300)
    return ollama.Client(host=LOCAL_HOST, timeout=600)


def _model_name(m):
    return CLOUD_MODEL if OLLAMA_API_KEY else m


class HistoryTurn(BaseModel):
    role: str
    content: str


class Question(BaseModel):
    question: str
    top_k: int = 3
    history: list[HistoryTurn] | None = None
    use_hybrid: bool = False
    use_hyde: bool = False
    use_rerank: bool = False
    use_rewrite: bool = False
    filters: dict | None = None
    stream: bool = False


GREETING_RE = re.compile(r"^\s*(hi+!?|hello!?|hey!?|hii+!?|good\s*(morning|afternoon|evening)\s*!?)\s*$", re.I)


def is_greeting(q: str) -> bool:
    return bool(GREETING_RE.match(q.strip())) and len(q.strip()) < 30


def is_overview_query(q: str) -> bool:
    t = re.sub(r"[^\w\s]", " ", q.lower())
    t = re.sub(r"\s+", " ", t).strip()
    if "summariz" in t or "overview" in t:
        return True
    has_what = "what" in t
    has_about = "about" in t
    has_doc = any(w in t for w in ["document", "pdf", "file", "this"])
    if has_what and has_about and has_doc:
        return True
    return bool(re.search(r"what.*about", t))


def _try_generate(prompt: str, options: dict | None = None, stream: bool = False):
    client = _ollama()
    for model in FALLBACK_MODELS:
        m = _model_name(model)
        try:
            if stream:
                return client.generate(model=m, prompt=prompt, options=options or {}, stream=True)
            text = client.generate(model=m, prompt=prompt, options=options or {}).get("response", "").strip()
            if text:
                return text
        except Exception:
            continue
    return None


def rewrite_query(q: str, history: list[HistoryTurn] | None) -> str:
    if not history:
        return q
    try:
        hist = "\n".join(f"{h.role}: {h.content}" for h in history[-4:])
        r = _try_generate(
            prompt=f"Rewrite the last user question to be standalone for retrieval, using history.\nHistory:\n{hist}\nQuestion: {q}\nRewritten:",
            options={"temperature": 0.2, "num_predict": 60},
        )
        if r:
            rw = r.strip().split("\n")[0].strip()
            return rw if len(rw) > 5 else q
    except Exception:
        pass
    return q


def _clean_lines(text: str):
    lines = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        latin = len(re.findall(r"[A-Za-z0-9 ]", line))
        dev = len(re.findall(r"[^\x00-\x7F]", line))
        if dev > latin and latin < 3:
            continue
        lines.append(line)
    return lines


def _clean_sentence(s: str) -> str:
    s = re.sub(r"\s*\|\s*", " ", s)
    s = re.sub(r"\S+@\S+", "", s)
    s = re.sub(r"\+91[-\s\d]+", "", s)
    s = s.replace("•", "·").replace("�", " ")
    return re.sub(r"\s{2,}", " ", s).strip()


def _sentences(text: str):
    # drop non-Latin-heavy lines (mojibake Devanagari) before splitting
    lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            continue
        latin = len(re.findall(r"[A-Za-z0-9 ]", s))
        nonlatin = len(re.findall(r"[^\x00-\x7F]", s))
        if nonlatin > latin and latin < 4:
            continue
        lines.append(s)
    cleaned = _clean_sentence(" ".join(lines))

    raw = [p.strip() for p in re.split(r"(?<=[.!?])\s+", cleaned) if p.strip()]
    parts = []
    carry = ""
    for p in raw:
        # a short fragment (a name/heading) belongs to the sentence that follows it
        if not parts and len(p) < 25:
            carry = p + " "
            continue
        parts.append((carry + p) if carry else p)
        carry = ""
    if carry:
        parts.append(carry.strip())
    return [p for p in parts if len(p) >= 10]


_HEADER_WORDS = {"secondary", "school", "examination", "marks", "statement", "certificate", "board", "central", "education", "resume", "summary", "certify", "course-a", "scholastic", "achievement", "positional", "grade", "theory", "internal", "assessment", "practical"}


def _synthesize(question: str, results: list[dict]) -> str:
    stop = {"is", "the", "a", "an", "what", "who", "can", "does", "do", "are", "did", "of", "in", "and", "to", "for", "get", "got", "good", "better", "best"}
    q_tokens = set(re.findall(r"\w+", question.lower())) - stop

    best_chunk = None
    best_chunk_score = -1
    for r in results:
        sents = _sentences(r["text"])
        if not sents:
            continue
        score = sum(len(q_tokens & set(re.findall(r"\w+", s.lower()))) for s in sents)
        if score > best_chunk_score:
            best_chunk_score = score
            best_chunk = (r, sents)

    if best_chunk is None:
        return "I could not find a clear answer in the uploaded documents."

    r, sents = best_chunk

    def is_boilerplate(s):
        low = s.lower()
        return sum(1 for w in _HEADER_WORDS if w in low) >= 2

    scored = []
    for s in sents:
        if is_boilerplate(s):
            continue
        sc = len(q_tokens & set(re.findall(r"\w+", s.lower())))
        scored.append((sc, s))
    scored.sort(key=lambda x: x[0], reverse=True)

    name = next((w for w in re.findall(r"[A-Z][a-zA-Z]+", question) if w.lower() not in stop), None)
    if name:
        named = [s for sc, s in scored if sc and name.lower() in s.lower()]
        if named:
            named.sort(key=len, reverse=True)
            body = named[0].strip()
            if body[-1] not in ".!?":
                body += "."
            return body + " [1]"
    # fall back to most informative sentence (longest)
    informative = [s for sc, s in scored if sc]
    if not informative:
        informative = [s for s in sents if not is_boilerplate(s)]
    if informative:
        informative.sort(key=len, reverse=True)
        body = informative[0][:650].strip()
        if body[-1] not in ".!?":
            body += "."
        return body + " [1]"

    return "I could not find a clear answer in the uploaded documents."


@app.get("/health")
def health():
    return {"status": "ok", "documents": rag.document_count()}


@app.get("/documents")
def documents():
    return {"documents": rag.list_documents(), "total_chunks": rag.document_count()}


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.post("/upload")
def upload(files: list[UploadFile] = File(default=None), file: UploadFile = File(default=None), strategy: str = Form(default="auto")):
    incoming: list[UploadFile] = []
    if files:
        incoming.extend(files)
    if file is not None:
        incoming.append(file)
    if not incoming:
        raise HTTPException(status_code=400, detail="no files provided")
    if strategy not in ("auto", "fixed", "recursive"):
        strategy = "auto"
    total = 0
    per = []
    for f in incoming:
        data = f.file.read()
        name = f.filename or "untitled"
        low = name.lower()
        try:
            if low.endswith(".pdf"):
                n = rag.add_document(name, data, strategy=strategy)
            elif low.endswith((".txt", ".md", ".csv")):
                n = rag.add_text_document(name, data.decode("utf-8", errors="ignore"), strategy=strategy)
            else:
                try:
                    n = rag.add_document(name, data, strategy=strategy)
                except Exception:
                    n = rag.add_text_document(name, data.decode("utf-8", errors="ignore"), strategy=strategy)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"{name}: could not read ({e})")
        total += n
        per.append({"filename": name, "chunks": n})
    return {"files": per, "total_chunks": total}


def retrieve_for(q: Question):
    query_text = rewrite_query(q.question, q.history) if q.use_rewrite else q.question
    is_overview = is_overview_query(query_text)
    use_hybrid = q.use_hybrid

    if is_overview:
        overview = rag.get_overview(limit=2)
        semantic = rag.query(query_text, top_k=2, use_hybrid=use_hybrid, use_rerank=q.use_rerank, filters=q.filters, use_hyde=q.use_hyde)
        seen = set()
        merged = []
        for r in overview + semantic:
            k = r["text"][:80]
            if k in seen:
                continue
            seen.add(k)
            merged.append(r)
        results = merged[: q.top_k]
    else:
        results = rag.query(query_text, top_k=q.top_k, use_hybrid=use_hybrid, use_rerank=q.use_rerank, filters=q.filters, use_hyde=q.use_hyde)
    return results, query_text, is_overview


def _context_block(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        body = "\n".join(_clean_lines(r["text"]))
        if not body.strip():
            continue
        parts.append(f"[Source {i}] ({r.get('source', 'unknown')})\n{body}")
    return "\n\n".join(parts)


def _build_prompt(question: str, results: list[dict], is_overview: bool, history: list[HistoryTurn] | None):
    hist_block = ""
    if history:
        hist_block = "\n".join(f"{h.role}: {h.content}" for h in history[-4:]) + "\n"
    context = _context_block(results)
    instruction = (
        "You are a grounded document assistant. Answer the question in natural, "
        "conversational sentences using ONLY the context below. Do not invent facts. "
        "If the context has the answer, give it concisely (a few sentences) and cite the "
        "source like [1]. If the context does NOT contain the answer, say exactly: "
        "I don't have information about that in the uploaded documents. Do not dump raw "
        "text verbatim; paraphrase into full sentences."
    )
    if is_overview:
        instruction = (
            "You are a grounded document assistant. Summarize in a few natural sentences "
            "what the uploaded documents are about, using ONLY the context below. Mention "
            "the document titles and main topics. Cite like [1]. Do not add anything not in context."
        )
    return (
        f"{hist_block}Context:\n\n{context}\n\n"
        f"{instruction}\n\nQuestion: {question}\nAnswer:"
    )


@app.get("/debug")
def debug():
    import urllib.request
    import json as _json
    if OLLAMA_API_KEY:
        host = CLOUD_HOST
        headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
        mode = "ollama-cloud"
    else:
        host = LOCAL_HOST
        headers = {}
        mode = "ollama-local"
    info = {"mode": mode, "ollama_host": host, "raw_chars": "", "models": [], "reachable": False, "error": None,
            "configured_model": OLLAMA_MODEL, "effective_model": _model_name(OLLAMA_MODEL)}
    try:
        req = urllib.request.Request(host + "/api/tags", headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            info["reachable"] = True
            info["raw_chars"] = resp.read().decode()
            info["models"] = [m.get("name") for m in (_json.loads(info["raw_chars"]).get("models") or [])]
    except Exception as e:
        info["error"] = str(e)
    return info


@app.post("/ask")
def ask(q: Question):
    if is_greeting(q.question):
        return {"answer": "Hi! I'm your document assistant. Ask me about your uploaded PDFs, or ask to summarize a document.", "sources": [], "rewritten": None, "evaluation": {"faithfulness": 1, "relevance": 1, "citation_count": 0}, "mode": "llm"}

    results, query_text, is_overview = retrieve_for(q)
    if not results:
        raise HTTPException(status_code=404, detail="no documents indexed yet")

    prompt = _build_prompt(q.question, results, is_overview, q.history)
    answer = _try_generate(prompt, options={"temperature": 0.2, "top_p": 0.9, "num_predict": 350})
    mode = "llm"
    if not answer or len(answer.strip()) < 15:
        answer = _synthesize(q.question, results)
        mode = "offline-fallback"

    return {"answer": answer, "sources": results, "rewritten": query_text if query_text != q.question else None, "evaluation": rag.evaluate(q.question, answer, results), "mode": mode}


@app.post("/ask/stream")
def ask_stream(q: Question):
    if is_greeting(q.question):
        msg = "Hi! I'm your document assistant. Ask me about your uploaded PDFs, or ask to summarize a document."

        def greet_stream():
            yield f"data: {json.dumps({'token': msg})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"
        return StreamingResponse(greet_stream(), media_type="text/event-stream")

    results, query_text, is_overview = retrieve_for(q)
    if not results:
        raise HTTPException(status_code=404, detail="no documents indexed yet")

    prompt = _build_prompt(q.question, results, is_overview, q.history)

    def token_stream():
        stream = _try_generate(prompt, options={"temperature": 0.2, "num_predict": 400}, stream=True)
        if stream is not None and not isinstance(stream, str):
            try:
                got = False
                for chunk in stream:
                    t = chunk.get("response", "")
                    if t:
                        got = True
                        yield f"data: {json.dumps({'token': t})}\n\n"
                if got:
                    yield f"data: {json.dumps({'done': True, 'sources': results, 'mode': 'llm', 'evaluation': rag.evaluate(q.question, '', results)})}\n\n"
                    return
            except Exception:
                pass

        fallback = _synthesize(q.question, results)
        yield f"data: {json.dumps({'token': fallback})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': results, 'mode': 'offline-fallback'})}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


@app.post("/evaluate")
def evaluate(payload: dict):
    q = payload.get("question", "")
    a = payload.get("answer", "")
    src = payload.get("sources") or rag.query(q, top_k=3)
    return rag.evaluate(q, a, src)


@app.post("/clear")
def clear():
    try:
        rag.clear_all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"clear failed: {e}")
    return {"status": "cleared"}
