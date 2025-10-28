# build_index.py
import os, json
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from db import fetch_all

load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index.bin")
IDS_PATH = os.getenv("FAISS_IDS_PATH", "./faiss_ids.npy")

def l2_normalize(x: np.ndarray) -> np.ndarray:
    # cosine = dot on normalized vectors
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def main():
    # 1) 판례 청크 로드 (id, precedent_id, chunk_index, text)
    rows = fetch_all(
        "SELECT id, precedent_id, chunk_index, text "
        "FROM precedent_chunks ORDER BY precedent_id, chunk_index"
    )
    if not rows:
        raise RuntimeError("precedent_chunks 가 비어있음")

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    texts = [r[3] for r in rows]

    print(f"[INDEX] load chunks: {len(texts)}")

    # 2) 임베딩
    print(f"[INDEX] load model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True)
    emb = emb.astype("float32")
    emb = l2_normalize(emb)

    # 3) FAISS (cosine via InnerProduct on normalized vectors)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    print(f"[INDEX] faiss added: {index.ntotal} vectors")

    # 4) 저장
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(IDS_PATH, ids)
    # 메타 요약(옵션): id→텍스트 앞부분
    meta_preview = [{"id": int(_id), "preview": texts[i][:120]} for i, _id in enumerate(ids)]
    with open("faiss_meta_preview.json", "w", encoding="utf-8") as f:
        json.dump(meta_preview, f, ensure_ascii=False, indent=2)

    print(f"[INDEX] saved: {FAISS_INDEX_PATH}, {IDS_PATH}, faiss_meta_preview.json")

if __name__ == "__main__":
    main()
