# server.py
import os
import json
import time
from typing import List, Tuple
import re
import numpy as np
import faiss
import requests

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from db import fetch_all

# -----------------------------------------
# 환경
# -----------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

PORT = int(os.getenv("FLASK_PORT", "5001"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index.bin")
IDS_PATH = os.getenv("FAISS_IDS_PATH", "./faiss_ids.npy")
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.55"))
USE_HF_TRANSLATOR = os.getenv("USE_HF_TRANSLATOR", "false").lower() == "true"

# -----------------------------------------
# Flask
# -----------------------------------------
app = Flask(__name__)
cors_origins = ALLOWED_ORIGINS if ALLOWED_ORIGINS == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",")]
CORS(app, resources={r"/*": {"origins": cors_origins}})

# -----------------------------------------
# 전역 Lazy 리소스
# -----------------------------------------
_model: SentenceTransformer = None
_index: faiss.Index = None
_ids: np.ndarray = None
_translator = None

# -----------------------------------------
# 한국어 답변 프롬프트
# -----------------------------------------
KO_INSTRUCTION = """당신은 한국 가사/이혼 판례를 근거로 답하는 법률 조력자입니다.
한국어만 사용하세요(영어·한자·중국어 금지). 근거 없는 추론 금지.
반드시 '[핵심 요지]' 번호 목록으로 3~5개 요약하고, 보이면 위자료/양육권/재산분할 기준을 구체 근거와 함께 적으세요.
참고한 판례의 사건번호/청크가 있으면 괄호로 표기하세요. 모호하면 '근거 부족'이라고 명시하세요.
"""

def build_llm_prompt(question: str, contexts: List[dict]) -> str:
    ctx_text = "\n\n".join(
        [f"[판례ID {c['precedent_id']} / 사건번호 {c.get('case_no','-')} / 청크 {c['chunk_index']}]\n{c['text']}" for c in contexts]
    )
    return f"""### 역할
{KO_INSTRUCTION}

### 질문
{question}

### 참고 판례
{ctx_text}

### 출력 형식
- 한국어만 사용
- [핵심 요지]: 번호 목록 3~5개
- [근거]: 인용한 판례ID/사건번호/청크를 괄호로 간단히 표기
- [주의]: 근거 불충분 시 '근거 부족' 명시

### 답변
"""

# -----------------------------------------
# 로딩 및 유틸
# -----------------------------------------
def ensure_loaded():
    global _model, _index, _ids, _translator
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    if _index is None:
        _index = faiss.read_index(FAISS_INDEX_PATH)
    if _ids is None:
        _ids = np.load(IDS_PATH)
    if _translator is None:
        if USE_HF_TRANSLATOR:
            try:
                _translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ko")
            except Exception:
                _translator = "ollama"
        else:
            _translator = "ollama"

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def search_similar(query: str, top_k: int = 3) -> Tuple[List[int], List[float]]:
    ensure_loaded()
    if _index is None or _index.ntotal <= 0:
        return [], []
    q = _model.encode([query]).astype("float32")
    q = l2_normalize(q)
    sims, idxs = _index.search(q, min(top_k, _index.ntotal))
    idxs, sims = idxs[0], sims[0]
    hit_ids = [int(_ids[i]) for i in idxs if i >= 0]
    hit_scores = [float(s) for s in sims[: len(hit_ids)]]
    return hit_ids, hit_scores

def fetch_chunks_with_meta(ids: List[int]) -> List[dict]:
    if not ids: return []
    placeholders = ",".join(["?"] * len(ids))
    rows = fetch_all(
        f"""
        SELECT c.id, c.precedent_id, c.chunk_index, c.text,
               p.case_no, p.court, p.judgment_date
        FROM precedent_chunks c
        JOIN precedents p ON p.id = c.precedent_id
        WHERE c.id IN ({placeholders})
        """,
        tuple(ids),
    )
    order = {i: idx for idx, i in enumerate(ids)}
    rows_sorted = sorted(rows, key=lambda r: order.get(r[0], 10**9))
    return [
        {"id": r[0], "precedent_id": r[1], "chunk_index": r[2], "text": r[3],
         "case_no": r[4], "court": r[5], "judgment_date": str(r[6])}
        for r in rows_sorted
    ]

def call_ollama(prompt: str, max_tokens: int = 700) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens}},
            timeout=120,
        )
        if resp.status_code == 200:
            return (resp.json().get("response") or "").strip()
        return f"[오류] Ollama 응답 코드 {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"[오류] Ollama 연결 실패: {e}"

_hangul_re = re.compile(r"[가-힣]")
def ensure_korean(text: str) -> str:
    if not text: return text
    ratio = len(_hangul_re.findall(text)) / max(len(text), 1)
    if ratio >= 0.25: return text
    prompt = f"다음 답변을 한국어로 자연스럽게 재작성하세요.\n{text}\n\n한국어:"
    fixed = call_ollama(prompt, max_tokens=800)
    return fixed if fixed else text

def grade(score_pct: float) -> str:
    if score_pct >= 80: return "green"
    if score_pct >= 60: return "yellow"
    return "red"

# -----------------------------------------
# Explanation/Highlight (생략 부분 동일)
# -----------------------------------------
from rag_explain import build_explanation_json, extract_highlights  # ✅ 이미 별도 모듈이면 그대로 유지

# -----------------------------------------
# 라우트
# -----------------------------------------
@app.route("/health")
def health():
    ensure_loaded()
    vecs = int(_index.ntotal) if _index else 0
    return jsonify({
        "ok": True, "vectors": vecs, "llm": LLM_MODEL,
        "embed_model": EMBEDDING_MODEL, "translator": ("hf" if _translator != "ollama" else "ollama")
    })

@app.route("/rag", methods=["POST"])
def rag():
    t_all0 = time.perf_counter()
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    top_k = int(data.get("top_k", 3))

    if not question:
        return jsonify({"error": "question is required"}), 400

    # 1️⃣ 검색
    t_s0 = time.perf_counter()
    hit_ids, hit_scores = search_similar(question, top_k)
    kept = [(i, s) for i, s in zip(hit_ids, hit_scores) if s >= MIN_SCORE]
    t_s1 = time.perf_counter()

    if not kept:
        return jsonify({
            "answer": "유사한 판례를 찾지 못했습니다.",
            "references": [], "scores": [], "grades": [],
            "perf": {"search_ms": round((t_s1 - t_s0)*1000,1), "total_ms": round((time.perf_counter()-t_all0)*1000,1)}
        })

    kept_ids, kept_scores = zip(*kept)
    contexts = fetch_chunks_with_meta(list(kept_ids))

    # 2️⃣ LLM 본답변
    prompt = build_llm_prompt(question, contexts)
    t_llm0 = time.perf_counter()
    answer_primary = call_ollama(prompt, max_tokens=900)
    t_llm1 = time.perf_counter()

    # 3️⃣ 후처리
    answer_kor = ensure_korean(answer_primary)
    scores_pct = [round(s * 100, 2) for s in kept_scores]
    grades = [grade(p) for p in scores_pct]
    refs = [
        {"id": c["id"], "precedent_id": c["precedent_id"], "chunk_index": c["chunk_index"],
         "case_no": c["case_no"], "court": c["court"], "judgment_date": c["judgment_date"]}
        for c in contexts
    ]

    # 4️⃣ 설명 및 하이라이트
    t_ex0 = time.perf_counter()
    explanation = build_explanation_json(question, contexts)
    highlights = extract_highlights(question, contexts)
    t_ex1 = time.perf_counter()

    # 5️⃣ 응답
    resp = {
        "answer": answer_kor,
        "answer_en": answer_primary,
        "references": refs,
        "scores": scores_pct,
        "grades": grades,
        "explanation": {
            "reasoning": explanation.get("reasoning", ""),
            "factors": explanation.get("factors", []),
            "highlights": highlights,
        },
        "처리시간": {
            "검색단계_초": round(t_s1 - t_s0, 3),
            "LLM단계_초": round(t_llm1 - t_llm0, 3),
            "설명단계_초": round(t_ex1 - t_ex0, 3),
            "총소요_초": round(time.perf_counter() - t_all0, 3),
        },
    }


    return Response(json.dumps(resp, ensure_ascii=False, indent=2), mimetype="application/json")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
