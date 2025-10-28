# server.py
import os
import json
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

OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index.bin")
IDS_PATH = os.getenv("FAISS_IDS_PATH", "./faiss_ids.npy")

MIN_SCORE = float(os.getenv("MIN_SCORE", "0.55"))  # 0~1
USE_HF_TRANSLATOR = os.getenv("USE_HF_TRANSLATOR", "true").lower() == "true"

# -----------------------------------------
# Flask
# -----------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# -----------------------------------------
# 전역 Lazy 리소스
# -----------------------------------------
_model: SentenceTransformer = None
_index: faiss.Index = None
_ids: np.ndarray = None
_translator = None  # HF translator instance or "ollama"

# -----------------------------------------
# 한국어 강제 지침 / 프롬프트 구성
# -----------------------------------------
KO_INSTRUCTION = """당신은 한국 가사/이혼 판례를 근거로 답하는 법률 조력자입니다.
한국어만 사용하세요(영어·한자·중국어 금지). 근거 없는 추론 금지.
반드시 '[핵심 요지]' 번호 목록으로 3~5개 요약하고, 보이면 위자료/양육권/재산분할 기준을 구체 근거와 함께 적으세요.
참고한 판례의 사건번호/청크가 있으면 괄호로 표기하세요. 모호하면 '근거 부족'이라고 명시하세요.
"""

def build_llm_prompt(question: str, contexts: List[dict]) -> str:
    ctx_text = "\n\n".join(
        [
            f"[판례ID {c['precedent_id']} / 사건번호 {c.get('case_no','-')} / 청크 {c['chunk_index']}]\n{c['text']}"
            for c in contexts
        ]
    )
    return f"""### 역할
{KO_INSTRUCTION}

### 질문
{question}

### 참고 판례 (인용 가능 구간)
{ctx_text}

### 출력 형식
- 한국어만 사용
- [핵심 요지]: 번호 목록 3~5개
- [근거]: 인용한 판례ID/사건번호/청크를 괄호로 간단히 표기
- [주의]: 근거 불충분 시 '근거 부족' 명시

### 답변
"""

# -----------------------------------------
# 유틸
# -----------------------------------------
def ensure_loaded():
    """임베딩/FAISS/IDs/번역기 로딩"""
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
            except Exception as e:
                print("[WARN] HF 번역기 로드 실패, Ollama 번역으로 폴백:", e)
                _translator = "ollama"
        else:
            _translator = "ollama"

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def search_similar(query: str, top_k: int = 3) -> Tuple[List[int], List[float]]:
    """질문→임베딩→FAISS: 상위 K개 id + 코사인 유사도(0~1)"""
    ensure_loaded()
    q = _model.encode([query]).astype("float32")
    q = l2_normalize(q)
    k = min(int(top_k), _index.ntotal if _index is not None else 0)
    if k <= 0:
        return [], []
    sims, idxs = _index.search(q, k)  # faiss는 내적. 우리가 L2정규화했으니 코사인과 동일
    idxs, sims = idxs[0], sims[0]
    hit_ids = [int(_ids[i]) for i in idxs if i >= 0]
    hit_scores = [float(s) for s in sims[: len(hit_ids)]]
    return hit_ids, hit_scores

def fetch_chunks_with_meta(ids: List[int]) -> List[dict]:
    """청크 id → 텍스트+메타(사건번호/법원/선고일)"""
    if not ids:
        return []
    placeholders = ",".join(["?"] * len(ids))
    rows = fetch_all(
        f"""
        SELECT
            c.id, c.precedent_id, c.chunk_index, c.text,
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
        {
            "id": r[0],
            "precedent_id": r[1],
            "chunk_index": r[2],
            "text": r[3],
            "case_no": r[4],
            "court": r[5],
            "judgment_date": str(r[6]),
        }
        for r in rows_sorted
    ]

def call_ollama(prompt: str, max_tokens: int = 700) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=120,
        )
        if resp.status_code == 200:
            return (resp.json().get("response") or "").strip()
        return f"[오류] Ollama 응답 코드 {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"[오류] Ollama 연결 실패: {e}"

def translate_via_ollama(text: str) -> str:
    if not text.strip():
        return text
    prompt = f"""아래 영어 문장을 한국어로 정확하고 자연스럽게 번역하세요.
- 법률 용어는 자연스러운 한국어로
- 요약/삭제 없이 의미 보존

[영문]
{text}

[한국어 번역]"""
    out = call_ollama(prompt, max_tokens=800)
    return out if out else text

def translate_to_korean(text: str) -> str:
    """HF 번역 우선 → 실패 시 Ollama → 마지막 폴백은 원문"""
    if not text or not text.strip():
        return text
    ensure_loaded()
    try:
        if isinstance(_translator, str) and _translator == "ollama":
            return translate_via_ollama(text)
        result = _translator(text, max_length=2048)
        return result[0]["translation_text"]
    except Exception as e:
        print("[WARN] 번역 실패, Ollama 폴백 시도:", e)
        try:
            return translate_via_ollama(text)
        except Exception as e2:
            print("[WARN] 번역 최종 실패, 원문 유지:", e2)
            return text

_hangul_re = re.compile(r"[가-힣]")
def ensure_korean(text: str) -> str:
    """한글 비율이 낮으면 한국어 재작성 요청"""
    if not text:
        return text
    m = _hangul_re.findall(text)
    ratio = len(m) / max(len(text), 1)
    if ratio >= 0.25:  # 한글이 어느 정도 있으면 통과
        return text
    # 한국어 재작성 프롬프트
    prompt = f"""다음 답변을 한국어로만, 자연스럽게 재작성하세요. 한자/영어 금지.
원문:
{text}

한국어:"""
    fixed = call_ollama(prompt, max_tokens=800)
    return fixed if fixed else text

def grade(score_pct: float) -> str:
    if score_pct >= 80:
        return "green"
    if score_pct >= 60:
        return "yellow"
    return "red"

# -----------------------------------------
# 라우트
# -----------------------------------------
@app.route("/health")
def health():
    try:
        ensure_loaded()
        return jsonify({
            "ok": True,
            "vectors": int(_index.ntotal),
            "llm": LLM_MODEL,
            "embed_model": EMBEDDING_MODEL,
            "translator": ("hf" if _translator != "ollama" else "ollama")
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/rag", methods=["POST"])
def rag():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    top_k = int(data.get("top_k", 3))

    if not question:
        return jsonify({"error": "question is required"}), 400

    # 1) 검색
    hit_ids, hit_scores = search_similar(question, top_k=top_k)
    kept = [(i, s) for i, s in zip(hit_ids, hit_scores) if s >= MIN_SCORE]

    if not kept:
        body = {
            "answer": "유사한 판례를 찾지 못했습니다. 사건 요약을 조금 더 구체적으로 입력해 주세요.",
            "references": [],
            "scores": [],
            "grades": []
        }
        return Response(json.dumps(body, ensure_ascii=False, indent=2), mimetype="application/json")

    kept_ids, kept_scores = zip(*kept)
    contexts = fetch_chunks_with_meta(list(kept_ids))

    # 2) 프롬프트 구성 (한국어 강제)
    prompt = build_llm_prompt(question, contexts)

    # 3) LLM 호출(비스트리밍)
    answer_primary = call_ollama(prompt, max_tokens=900)

    # 4) 만약 영어/혼종이면 한국어로 강제 보정
    answer_kor = ensure_korean(answer_primary)

    # 5) (선택) 영어 원문도 보고 싶으면 역번역: 한국어→영어 (여긴 기본 비활성)
    #    지금은 클라이언트에서 디버깅 용도로 answer_en 표시가 필요할 수 있으니,
    #    한국어 보정 전 원문을 answer_en로 둠
    answer_en = answer_primary

    # 6) 시각화 필드
    scores_pct = [round(s * 100, 2) for s in kept_scores]
    grades = [grade(p) for p in scores_pct]
    refs = [
        {
            "id": c["id"],
            "precedent_id": c["precedent_id"],
            "chunk_index": c["chunk_index"],
            "case_no": c["case_no"],
            "court": c["court"],
            "judgment_date": c["judgment_date"],
        }
        for c in contexts
    ]

    resp = {
        "answer": answer_kor,      # ✅ 최종 한국어
        "answer_en": answer_en,    # ⛳️ 디버깅/비교용 원문
        "references": refs,
        "scores": scores_pct,
        "grades": grades
    }

    return Response(json.dumps(resp, ensure_ascii=False, indent=2), mimetype="application/json")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=(os.getenv("ENV") == "dev"))
