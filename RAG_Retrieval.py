import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from Paths import BASE, MODELS_DIR, DATA_DIR, rp
def load_corpus(jsonl_path: Path) -> List[Dict[str, Any]]:
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            meta = obj.get("metadata", {}) or {}
            docs.append({
                "id": obj.get("id"),
                "text": obj.get("text", "") or "",
                "title": meta.get("title", "") or "",
                "source_uri": meta.get("source_uri", "") or "",
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "element_type": meta.get("element_type", "") or "",
            })
    return docs

def make_snippet(text: str, maxlen: int = 420) -> str:
    s = (text or "").strip().replace("\n", " ")
    if maxlen is None:
        maxlen = 420
    # <=0 表示不截斷
    if maxlen <= 0 or len(s) <= maxlen:
        return s
    # 在 maxlen 之前找最近的空白來斷句，避免把單字切成一半
    cut = s.rfind(" ", 0, maxlen)
    if cut == -1:
        cut = maxlen
    return s[:cut] + " ..."

_token_pat = re.compile(r"[a-z0-9][a-z0-9\-]+|[a-z0-9]", flags=re.IGNORECASE)

def _tokenize(s: str):
    return [t.lower() for t in _token_pat.findall(s or "")]

class BM25:
    def __init__(self, docs: List[Dict[str, Any]]):
        self.N = len(docs)
        self.docs = docs
        self.doc_tokens = [_tokenize(d["text"]) for d in docs]
        self.df = {}
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = (sum(len(t) for t in self.doc_tokens) / max(1, self.N)) if self.N else 0.0
        self.idf = {t: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for t, df in self.df.items()}

    def search(self, query: str, topk: int = 5, k1: float = 1.5, b: float = 0.75):
        q_tokens = _tokenize(query)
        scores = []
        for i, toks in enumerate(self.doc_tokens):
            if not toks:
                continue
            dl = len(toks)
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for q in q_tokens:
                f = tf.get(q, 0)
                if f == 0:
                    continue
                idf = self.idf.get(q, 0.0)
                denom = f + k1 * (1 - b + b * (dl / (self.avgdl or 1)))
                score += idf * (f * (k1 + 1)) / (denom or 1.0)
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        return scores[:topk]

def try_load_faiss(index_path: Optional[Path]):
    if not index_path or not index_path.exists():
        return None
    try:
        import faiss  # type: ignore
        return faiss.read_index(str(index_path))
    except Exception as e:
        print(f"[WARN] 無法載入 FAISS：{e}")
        return None

def try_load_embedder(model_name_or_path: Optional[str]):
    if not model_name_or_path:
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer(model_name_or_path, device="cpu")
        return model
    except Exception as e:
        print(f"[WARN] 無法載入嵌入模型：{e}")
        return None

def faiss_search(query: str,
                 index,
                 embedder,
                 topk: int = 5,
                 query_prefix: str = "query: ") -> List[tuple]:
    # 以 E5/SBERT 系列為例，常見 query 用 "query: " 前綴（若當初建庫沒用，可改成空字串）
    q = query_prefix + query if query_prefix else query
    vec = embedder.encode([q], normalize_embeddings=True)
    import numpy as np  # sentence_transformers 會帶入 numpy
    D, I = index.search(np.asarray(vec, dtype="float32"), topk)
    # 回傳 [(score, doc_idx), ...]；這裡用 -distance 作為相似度（L2 距離越小越相似）
    hits = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        # 注意：假設 index 與 docs 的順序一致（最常見做法）
        hits.append((-float(dist), int(idx)))
    return hits

def normalize(scores: List[tuple]) -> Dict[int, float]:
    """
    將 [(score, idx), ...] normalize 到 0~1，回傳 {idx: norm_score}
    """
    if not scores:
        return {}
    vals = [s for s, _ in scores]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {idx: 1.0 for s, idx in scores}
    return {idx: (s - lo) / (hi - lo) for s, idx in scores}

def hybrid_merge(bm25_res: List[tuple], vec_res: List[tuple], w_bm25=0.5, w_vec=0.5, topk=5):
    nb = normalize(bm25_res)
    nv = normalize(vec_res)
    keys = set(nb) | set(nv)
    fused = []
    for k in keys:
        fused.append((w_bm25 * nb.get(k, 0.0) + w_vec * nv.get(k, 0.0), k))
    fused.sort(reverse=True)
    return fused[:topk]

def compose_augmented_prompt(snippet_chars: int, user_query: str, docs: List[Dict[str, Any]], ranked: List[tuple]) -> str:
    lines = []
    for rank, (score, idx) in enumerate(ranked, start=1):
        if idx < 0 or idx >= len(docs):
            continue
        d = docs[idx]
        pages = ""
        if d.get("page_start") is not None:
            pages = f'{d.get("page_start")}-{d.get("page_end")}'
        lines.append(f'[[Context {rank} | {Path(d.get("source_uri","")).name}]]')
        lines.append(make_snippet(d.get("text",""), maxlen=snippet_chars))
    ctx_block = "\n".join(lines)

    return f"""
# 角色
你是專業的技術研究助理。請根據「已檢索到的內容」回答以及你「另外的理解」回答，分為兩部分；若內容不足300字，請明確說明缺少資訊並提出可追問的關鍵點。避免編造。

# 已檢索到的內容
{ctx_block}

# 使用者問題
{user_query}

# 回答要求
- 優先使用檢索內容中的觀點與數據，必要時可整合多段內容。
- 若不同段落有衝突，請指出衝突並給出最合理的解釋。
    """

def format_references(docs: List[Dict[str, Any]], ranked: List[tuple]) -> str:
    lines = ["[參考資料]"]
    for rank, (score, idx) in enumerate(ranked, start=1):
        if idx < 0 or idx >= len(docs):
            continue
        d = docs[idx]
        pages = ""
        if d.get("page_start") is not None:
            pages = f'{d.get("page_start")}-{d.get("page_end")}'
        lines.append(f"{rank}. {Path(d.get('source_uri','')).name} | score={score:.4f}")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Simple RAG Retriever -> Augmented Prompt")
    parser.add_argument("--data", type=str, default="./VectorDatabase/metadata.jsonl", help="metadata.jsonl 路徑")
    parser.add_argument("--index", type=str, default="/VectorDatabase/index.faiss", help="index.faiss 路徑（可選）")
    parser.add_argument("--embedder", type=str, default= MODELS_DIR/"intfloat-multilingual-e5-base", help="SentenceTransformer 模型")
    parser.add_argument("--topk", type=int, default=10, help="擷取前幾筆片段")
    parser.add_argument("--snippet", type=int, default=480, help="每段 snippet 的最大字元數；<=0 表示不截斷")
    parser.add_argument("--hybrid", action="store_true", help="啟用 Hybrid（BM25 + 向量）融合")
    parser.add_argument("--query_prefix", type=str, default="query: ", help="查詢前綴（E5 常用 'query: '）")
    parser.add_argument("--save", type=str, default="augmented_prompt.txt", help="輸出檔名")
    args = parser.parse_args()

    docs = load_corpus(Path(args.data))
    if not docs:
        raise SystemExit("沒有載入到任何文件片段，請確認 metadata.jsonl")

    # BM25
    bm25 = BM25(docs)

    # 可選：FAISS + Embedder
    index = try_load_faiss(Path(args.index)) if args.index else None
    embedder = try_load_embedder(args.embedder) if index is not None else None

    # 使用者輸入查詢
    try:
        user_query = input("請輸入你的查詢（prompt）：").strip()
    except EOFError:
        user_query = ""
    if not user_query:
        raise SystemExit("未輸入查詢，結束。")

    # 取得候選結果
    bm25_res = bm25.search(user_query, topk=args.topk)
    vec_res = []
    if index is not None and embedder is not None:
        try:
            vec_res = faiss_search(user_query, index, embedder, topk=args.topk, query_prefix=args.query_prefix)
        except Exception as e:
            print(f"[WARN] 向量檢索失敗，改用 BM25：{e}")

    if args.hybrid and vec_res:
        ranked = hybrid_merge(bm25_res, vec_res, w_bm25=0.5, w_vec=0.5, topk=args.topk)
    else:
        ranked = vec_res if vec_res else bm25_res

    # 組裝 Augmented Prompt
    aug = compose_augmented_prompt(args.snippet, user_query, docs, ranked)

    # 顯示參考資料
    print("\n" + format_references(docs, ranked) + "\n")
    print("=== Augmented Prompt ===")
    print(aug)

    # 存檔
    out_path = Path(args.save)
    out_path.write_text(aug, encoding="utf-8")
    print(f"\n已輸出：{out_path.resolve()}")

if __name__ == "__main__":
    main()