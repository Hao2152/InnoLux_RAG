import argparse
import json
import math
import re, unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from Paths import BASE, MODELS_DIR, DATA_DIR, rp

# ========================
# 句子級 re-rank + 呈現工具
# ========================
_GLOBAL_EMBEDDER = None   # 由 try_load_embedder 設定，compose_augmented_prompt 可取用

# 常見中英標點斷句；保留標點後換行或空白做切分
_SENT_SPLIT = re.compile(r'(?<=[。！？…；：!?;:])\s+|(?<=[\.\?\!])\s+')
def looks_like_bib_by_snippet(text: str, user_query: str, max_chars: int = 240) -> bool:
    """
    以『挑出的最佳片段（snippet）』來判斷是否像參考文獻。
    只要 snippet 不像文獻，就保留該候選，避免因 chunk 內少量 [n]/vol./pp. 誤殺。
    """
    try:
        _min = 60
        _max = max(120, min(max_chars or 240, 490))
        snippet = _pick_best_sentence(
            user_query, text or "", embedder=_GLOBAL_EMBEDDER,
            min_chars=_min, max_chars=_max
        )
        return looks_like_bibliography(snippet)
    except Exception:
        # 出錯時退回 chunk 判斷，寧可誤擋也不要漏網
        return looks_like_bibliography(text)
    
def _split_sentences(text: str) -> list[str]:
    s = (text or "").strip().replace("\n", " ")
    if not s:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(s) if p and p.strip()]
    # 若整段幾乎沒有標點，退回粗切：每 ~100 字切一刀，避免只拿到片語
    if len(parts) <= 1 and len(s) > 140:
        step = 100
        parts = [s[i:i+step].strip() for i in range(0, len(s), step)]
    return parts

_token_pat = re.compile(r"[a-z0-9][a-z0-9\-]+|[a-z0-9]", flags=re.IGNORECASE)

def _tokenize(s: str):
    return [t.lower() for t in _token_pat.findall(s or "")]

def _grow_span_around(sents: list[str], center: int, *, min_chars: int, max_chars: int) -> str:
    """從 center 句開始，向左右擴張直到達到 min_chars（不超過 max_chars）。"""
    if not sents:
        return ""
    left = center - 1
    right = center + 1
    result = sents[center].strip()

    # 交替左右擴張，直到達到最小字元數或沒有空間
    turn_right = True
    while len(result) < min_chars and (left >= 0 or right < len(sents)):
        added = False
        if turn_right and right < len(sents):
            cand = (result + " " + sents[right].strip()) if result else sents[right].strip()
            if len(cand) <= max_chars:
                result = cand
                right += 1
                added = True
        elif not turn_right and left >= 0:
            cand = (sents[left].strip() + " " + result) if result else sents[left].strip()
            if len(cand) <= max_chars:
                result = cand
                left -= 1
                added = True
        turn_right = not turn_right
        if not added:
            break

    if len(result) > max_chars:
        result = result[:max_chars-1].rstrip() + "…"
    return result

def _simple_tokens(t: str) -> list[str]:
    # 英數與中日韓統一表意文字做簡單分詞
    return [w for w in re.split(r'[^0-9A-Za-z\u4e00-\u9fff]+', (t or "").lower()) if w]

def make_snippet(text: str, maxlen: int = 420) -> str:
    """簡單摘要：壓縮空白、限制最大長度並加上省略號。"""
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    if maxlen and maxlen > 0 and len(t) > maxlen:
        return t[:maxlen-1].rstrip() + "…"
    return t

def _extract_query_terms(q: str) -> list[str]:
    """從查詢中抽取要高亮的詞：
    - 英數詞（長度>=2）
    - 連續的中日韓表意字串（長度>=1）
    """
    if not q:
        return []
    en = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", q)
    cjk = re.findall(r"[\u4e00-\u9fff]{1,}", q)
    terms = set([w for w in en if len(w) >= 2]) | set([s.strip() for s in cjk if s.strip()])
    return sorted(terms, key=len, reverse=True)  # 先配對長詞，避免被短詞吃掉

def mask_snippet(snippet: str, query: str, left: str = "【", right: str = "】") -> str:
    """把與 query 相符的詞加上標記（預設【】）。"""
    terms = _extract_query_terms(query)
    if not snippet or not terms:
        return snippet
    patt = "|".join(re.escape(t) for t in terms)
    try:
        return re.sub(patt, lambda m: f"{left}{m.group(0)}{right}", snippet, flags=re.IGNORECASE)
    except re.error:
        return snippet

# --- 關鍵詞偏置（更貼合“怎麼選/定義/標註”類問題） ---
_BONUS_TERMS = {"define","definition","ground truth","annotation","label","mask","gt",
                "標註","遮罩","定義","選","準則","規範","標準"}

def _kw_bonus(sent: str) -> float:
    s = (sent or "").lower()
    return 0.05 if any(k in s for k in _BONUS_TERMS) else 0.0

def _pick_best_sentence(user_query: str, text: str, embedder=None, *, min_chars: int = 80, max_chars: int = 420) -> str:
    """回傳『至少是完整一句、長度夠』的參考片段。"""
    sents = _split_sentences(text)
    if not sents:
        return make_snippet(text, maxlen=max_chars)

    best_idx = 0
    try:
        if embedder is not None:
            qv = embedder.encode(["query: " + (user_query or "")], normalize_embeddings=True)[0]
            X = embedder.encode(["passage: " + s for s in sents], normalize_embeddings=True)
            sims = (X @ qv).tolist()
            sims = [v + _kw_bonus(sents[i]) for i, v in enumerate(sims)]  # 小額偏置
            best_idx = max(range(len(sents)), key=lambda i: sims[i])
        else:
            qtok = set(_simple_tokens(user_query))
            def score(sent: str):
                stok = set(_simple_tokens(sent))
                overlap = sum(1 for t in qtok if t in stok)
                penalty = 0.5 if len(sent) < 12 or len(stok) <= 2 else 1.0
                return overlap * penalty + _kw_bonus(sent)
            best_idx = max(range(len(sents)), key=lambda i: score(sents[i]))
    except Exception:
        return make_snippet(text, maxlen=max_chars)

    best = sents[best_idx].strip()
    if len(best) < min_chars or len(_simple_tokens(best)) <= 2:
        return _grow_span_around(sents, best_idx, min_chars=min_chars, max_chars=max_chars)
    return best if len(best) <= max_chars else (best[:max_chars-1].rstrip() + "…")

# ========================
# 語料與檢索
# ========================

def load_corpus(jsonl_path: Path) -> List[Dict[str, Any]]:
    """讀取 metadata.jsonl，統一成檢索端需要的欄位。"""
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
                "source_uri": meta.get("source_uri", "") or "",
                # 新版欄位（RAG_Chunking 方案A）
                "page_no": meta.get("page_no"),
                "chunk_no": meta.get("chunk_no"),
                "piece_no": meta.get("piece_no"),
                "doc_name": meta.get("doc_id", "") or "",
                "checksum": meta.get("checksum", "") or "",
                # 舊版欄位（若存在則保留）
                "title": meta.get("title", "") or "",
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "element_type": meta.get("element_type", "") or "",
            })
    return docs

class BM25:
    """簡化版 BM25；回傳 (score, doc_id) 以便與向量結果合流。"""
    def __init__(self, docs: List[Dict[str, Any]]):
        self.N = len(docs)
        self.docs = docs
        self.doc_ids = [d["id"] for d in docs]
        self.doc_tokens = [_tokenize(d["text"]) for d in docs]
        self.df: Dict[str, int] = {}
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = (sum(len(t) for t in self.doc_tokens) / max(1, self.N)) if self.N else 0.0
        self.idf = {t: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for t, df in self.df.items()}

    def search(self, query: str, topk: int = 5, k1: float = 1.5, b: float = 0.75):
        q_tokens = _tokenize(query)
        scores: List[tuple] = []
        for i, toks in enumerate(self.doc_tokens):
            if not toks:
                continue
            dl = len(toks)
            tf: Dict[str, int] = {}
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
                scores.append((score, self.doc_ids[i]))  # ← 回傳 doc_id
        scores.sort(reverse=True)
        return scores[:topk]

# ---------- 載入 FAISS / Embedder ----------

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
        global _GLOBAL_EMBEDDER
        _GLOBAL_EMBEDDER = model
        return model
    except Exception as e:
        print(f"[WARN] 無法載入嵌入模型：{e}")
        return None

# ---------- 向量檢索 ----------

def faiss_search(query: str,
                 index,
                 embedder,
                 topk: int = 5,
                 query_prefix: str = "query: ") -> List[tuple]:
    """以向量檢索回傳 (score, doc_id)。自動判斷 index metric。"""
    q = query_prefix + query if query_prefix else query
    vec = embedder.encode([q], normalize_embeddings=True)
    import numpy as np, faiss  # type: ignore
    D, I = index.search(np.asarray(vec, dtype="float32"), topk)

    metric = getattr(index, "metric_type", None)
    hits: List[tuple] = []
    for dist, doc_id in zip(D[0], I[0]):
        if doc_id < 0:
            continue
        # METRIC_INNER_PRODUCT：越大越好；L2：距離越小越好
        if metric == faiss.METRIC_INNER_PRODUCT:
            score = float(dist)
        else:
            score = -float(dist)
        hits.append((score, int(doc_id)))
    return hits

# ========================
# 融合、過濾與展示
# ========================

def normalize(scores: List[tuple]) -> Dict[int, float]:
    """將 [(score, id), ...] normalize 到 0~1，回傳 {id: norm_score}"""
    if not scores:
        return {}
    vals = [s for s, _ in scores]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {idx: 1.0 for s, idx in scores}
    return {idx: (s - lo) / (hi - lo) for s, idx in scores}

def hybrid_merge(bm25_res: List[tuple], vec_res: List[tuple], w_bm25=0.3, w_vec=0.7, topk=5):
    nb = normalize(bm25_res)
    nv = normalize(vec_res)
    keys = set(nb) | set(nv)
    fused = []
    for k in keys:
        fused.append((w_bm25 * nb.get(k, 0.0) + w_vec * nv.get(k, 0.0), k))
    fused.sort(reverse=True)
    return fused[:topk]

# 參考文獻頁面/文獻列表的偵測
def looks_like_bibliography(t: str) -> bool:
    if not t:
        return False
    s = unicodedata.normalize("NFKC", t.strip())
    sl = s.lower()

    # 統一小寫的關鍵詞（都比對 sl）
    ref_kw = (
        "references", "bibliography", "acknowledg",
        "doi:", "springer", "elsevier",
        "proc.", "in proceedings", "conf.", "journal", "vol.", "no.", "pp.", "pages"
    )
    if any(k in sl for k in ref_kw):
        return True

    # 引用標記 [1] / [23]
    if re.search(r"\[\d{1,3}\]", s):
        return True

    # 多作者 + 年份樣式（Borji A, Cheng M-M, ... 2015）
    if re.search(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*,\s*[A-Z](?:\.[A-Z])?\.){2,}.*\b(19|20)\d{2}\b", s):
        return True

    # 頁碼區間（478–487 / 478-487）+ 多逗號
    if re.search(r"\b\d{2,4}\s*[–-]\s*\d{2,4}\b", s) and s.count(",") >= 3:
        return True

    # 短段且包含引用
    if len(s) < 400 and re.findall(r"\[\d{1,3}\]", s):
        return True

    return False

def page_key(d: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    fname = Path(d.get("source_uri", "")).name
    return (fname, d.get("page_no"))

def compose_augmented_prompt(snippet_chars: int, user_query: str, docs: List[Dict[str, Any]], ranked: List[tuple], id2row: Dict[int, int], highlight: bool = False) -> str:
    lines = []
    for rank, (score, doc_id) in enumerate(ranked, start=1):
        row = id2row.get(doc_id)
        if row is None or row < 0 or row >= len(docs):
            continue
        d = docs[row]
        fname = Path(d.get("source_uri", "")).name
        page = d.get("page_no")
        tag = f"p{page}" if page is not None else ""
        lines.append(f"[[Context {rank} | {fname} {tag}]]")
        _min_chars = max(50, min(160, (snippet_chars or 420)//3))
        snippet = _pick_best_sentence(user_query, d.get("text", ""), embedder=_GLOBAL_EMBEDDER, min_chars=_min_chars, max_chars=snippet_chars)
        snippet = mask_snippet(snippet, user_query) if highlight else snippet
        lines.append(snippet)

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

def format_references(docs: List[Dict[str, Any]], ranked: List[tuple], id2row: Dict[int, int]) -> str:
    lines = ["[參考資料]"]
    for rank, (score, doc_id) in enumerate(ranked, start=1):
        row = id2row.get(doc_id)
        if row is None or row < 0 or row >= len(docs):
            continue
        d = docs[row]
        fname = Path(d.get('source_uri','')).name
        page = d.get('page_no')
        tag = f"p{page}" if page is not None else ""
        lines.append(f"{rank}. {fname} {tag} | score={score:.4f}")
    return "\n".join(lines)

# ========================
# Query Expansion（選配）
# ========================

def expand_queries(user_query: str) -> List[str]:
    uq = user_query.strip()
    if not uq:
        return []
    extras = [
        f"{uq} 標註 準則", f"{uq} 定義", f"{uq} 選取 規範",
        "salient object annotation criteria",
        "how ground truth masks are defined in salient object detection",
        "dataset annotation guideline salient object",
        "GT labeling rule salient"
    ]
    # 去重並保持順序
    seen = set([uq])
    out = [uq]
    for x in extras:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

# ========================
# CLI
# ========================

def main():
    DEFAULT_MODEL_DIR = (MODELS_DIR / "intfloat-multilingual-e5-base")
    parser = argparse.ArgumentParser(description="RAG Retriever -> Augmented Prompt（id 映射＋句子級 snippet＋過濾/擴充/多樣化/高亮）")
    parser.add_argument("--data", type=str, default="", help="metadata.jsonl 路徑")
    parser.add_argument("--index", type=str, default="", help="index.faiss 路徑（可選）")
    parser.add_argument("--embedder", type=str, default=str(DEFAULT_MODEL_DIR), help="SentenceTransformer 模型（與建庫一致）")
    parser.add_argument("--topk", type=int, default=5, help="擷取前幾筆片段")
    parser.add_argument("--snippet", type=int, default=480, help="每段 snippet 的最大字元數；<=0 表示不截斷")
    parser.add_argument("--hybrid", action="store_true", help="啟用 Hybrid（BM25 + 向量）融合")
    parser.add_argument("--w-bm25", type=float, default=0.3, help="Hybrid 權重：BM25（預設 0.3）")
    parser.add_argument("--w-vec", type=float, default=0.7, help="Hybrid 權重：向量（預設 0.7）")
    parser.add_argument("--query_prefix", type=str, default="query: ", help="查詢前綴（E5 常用 'query: '）")
    parser.add_argument("--save", type=str, default="", help="輸出檔名")
    parser.add_argument("--allow-bib", action="store_true", help="允許參考文獻類片段（預設過濾掉）")
    parser.add_argument("--page-diverse", action="store_true", help="避免同一頁佔滿（保留每頁最多一段）")
    parser.add_argument("--highlight", action="store_true", help="在輸出中對命中詞高亮（對 LLM 可關閉）")
    parser.add_argument("--qe", action="store_true", help="開啟查詢擴充（multi-query）")
    parser.add_argument("--qe-topn", type=int, default=3, help="每個子查詢保留前 N 筆（預設 3）")
    args = parser.parse_args()


    docs = load_corpus(Path(args.data))
    if not docs:
        raise SystemExit("沒有載入到任何文件片段，請確認 metadata.jsonl")

    # 建立 id→row 對應（健全化 2）
    id2row = {d["id"]: i for i, d in enumerate(docs) if d.get("id") is not None}

    # 檢索器
    bm25 = BM25(docs)
    index = try_load_faiss(Path(args.index)) if args.index else None
    embedder = try_load_embedder(args.embedder) if index is not None else None

    # 使用者輸入查詢
    try:
        user_query = input("請輸入你的查詢（prompt）：").strip()
    except EOFError:
        user_query = ""
    if not user_query:
        raise SystemExit("未輸入查詢，結束。")

    # 檢索（單一或查詢擴充）
    def search_one(q: str, topn: int) -> List[tuple]:
        bm25_res = bm25.search(q, topk=topn)
        vec_res: List[tuple] = []
        if index is not None and embedder is not None:
            try:
                vec_res = faiss_search(q, index, embedder, topk=topn, query_prefix=args.query_prefix)
            except Exception as e:
                print(f"[WARN] 向量檢索失敗，改用 BM25：{e}")
        if args.hybrid and vec_res:
            return hybrid_merge(bm25_res, vec_res, w_bm25=args.w_bm25, w_vec=args.w_vec, topk=topn)
        return vec_res if vec_res else bm25_res

    if args.qe:
        candidates: Dict[int, float] = {}
        for q in expand_queries(user_query):
            fused = search_one(q, args.qe_topn)
            # 以每個子查詢內的 normalize 分數累加
            for doc_id, score in normalize(fused).items():
                candidates[doc_id] = candidates.get(doc_id, 0.0) + score
        ranked = sorted([(s, i) for i, s in candidates.items()], reverse=True)[:max(args.topk, 1)]
    else:
        ranked = search_one(user_query, args.topk)

    # 過濾：移除不在 id2row 的 id（避免索引/語料不一致）
    ranked = [(s, i) for (s, i) in ranked if i in id2row]
    # 參考文獻過濾（預設啟用；--allow-bib 可關閉）
    if not args.allow_bib:
        ranked = [(s, i) for (s, i) in ranked if not looks_like_bib_by_snippet(docs[id2row[i]]["text"], user_query, args.snippet)]
    # 頁面多樣化（可選）
    if args.page_diverse:
        seen_pages = set()
        unique = []
        for s, i in ranked:
            d = docs[id2row[i]]
            key = page_key(d)
            if key in seen_pages:
                continue
            seen_pages.add(key)
            unique.append((s, i))
        ranked = unique
    # 取前 topk
    ranked = ranked[:args.topk]

    if not ranked:
        print("[INFO] 查無結果，請嘗試更換關鍵字或放寬條件。")
        return

    # 組裝 Augmented Prompt
    aug = compose_augmented_prompt(args.snippet, user_query, docs, ranked, id2row, highlight=args.highlight)

    # 顯示參考資料
    print("\n" + format_references(docs, ranked, id2row) + "\n")
    print("=== Augmented Prompt ===")
    print(aug)

    # 存檔
    out_path = Path(args.save)
    out_path.write_text(aug, encoding="utf-8")
    print(f"\n已輸出：{out_path.resolve()}")

if __name__ == "__main__":
    main()
