import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 以上兩行要在載入模型之前

import re
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
import fitz  # PyMuPDF

import torch
from transformers import AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging
from Paths import BASE, MODELS_DIR, DATA_DIR, rp
Model_Dir = MODELS_DIR / "intfloat-multilingual-e5-base"
model = SentenceTransformer(str(Model_Dir), device="cpu")
#關閉huggingface提醒
hf_logging.set_verbosity_error()
_WS = re.compile(r'\s+')

# ---------- 小工具 ----------
def normalize_text_no_newline(s: str) -> str:
    """將所有換行（實際換行與字面 '\\n'）改成空白，並壓縮多個空白。"""
    if not isinstance(s, str):
        s = str(s)
    # 先統一換行符號
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    # 把字面 "\n" 與實際換行都變空白
    s = s.replace('\\n', ' ').replace('\n', ' ')
    # 壓縮連續空白（含 tab）
    s = _WS.sub(' ', s).strip()
    return s

def split_if_overlong(text: str, tokenizer: AutoTokenizer, model_name: str,
                      margin: int = 32, prefix: str = "passage: ", overlap_ratio: float = 0.17) -> list[str]:
    """
    若 text 編成 token 後超過模型上限，切成多段（含重疊），確保每段 <= 可用預算。
    overlap_ratio 建議 0.15~0.20
    """
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    max_len = getattr(cfg, "max_position_embeddings", 512) or 512

    # E5/BGE 會加前綴，再預留一些安全邊界
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    budget = max(128, max_len - margin - len(prefix_ids))

    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return [text]

    pieces = []
    step = max(1, int(budget * (1 - overlap_ratio)))  # 例如 0.83 * budget
    for i in range(0, len(ids), step):
        sub_ids = ids[i:i+budget]
        pieces.append(tokenizer.decode(sub_ids, skip_special_tokens=True))
    return pieces

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()

def normalize_text(t: str) -> str:
    t = t.replace("\r", "")
    t = re.sub(r"-\n(?=\w)", "", t)        # 英文連字斷行
    t = re.sub(r"[ \t]+", " ", t)          # 多餘空白
    t = re.sub(r"\n{3,}", "\n\n", t)       # 多餘空行
    t = re.sub(r"[ \t]*\n[ \t]*", "\n", t) # 行尾空白
    # 去掉純頁碼行
    t = "\n".join([ln for ln in t.split("\n") if not re.fullmatch(r"\d{1,4}", ln.strip())])
    return t.strip()

def pdf_pages_text(pdf_path: Path) -> List[str]:
    doc = fitz.open(pdf_path.as_posix())
    texts = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        txt = page.get_text("text") or ""
        texts.append(normalize_text(txt))
    doc.close()
    return texts

def chunk_by_token(tokenizer: AutoTokenizer, page_text: str, chunk_tokens=700, overlap=120) -> List[str]:
    if not page_text.strip():
        return []
    paras = [p.strip() for p in re.split(r"\n{2,}", page_text) if p.strip()]
    chunks, buf, cur = [], [], 0
    for p in paras:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if cur + len(ids) <= chunk_tokens:
            buf.append(p); cur += len(ids)
        else:
            if buf:
                chunks.append("\n\n".join(buf))
            if overlap > 0 and chunks:
                last = chunks[-1]
                last_ids = tokenizer.encode(last, add_special_tokens=False)
                keep_ids = last_ids[max(0, len(last_ids)-overlap):]
                keep_txt = tokenizer.decode(keep_ids)
                buf = [keep_txt, p]
                cur = len(tokenizer.encode(keep_txt, add_special_tokens=False)) + len(ids)
            else:
                buf, cur = [p], len(ids)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks

class E5Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_chunks(self, texts: List[str]) -> np.ndarray:
        prep = [f"passage: {t}" for t in texts]
        vecs = self.model.encode(
            prep, batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True
        )
        return vecs.astype(np.float32)

def load_existing(index_dir: Path):
    idx_path = index_dir / "index.faiss"
    meta_path = index_dir / "metadata.jsonl"
    cfg_path = index_dir / "config.json"

    if not (idx_path.exists() and meta_path.exists() and cfg_path.exists()):
        return None

    index = faiss.read_index(str(idx_path))
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {"index": index, "metas": metas, "cfg": cfg}

def save_index(index_dir: Path, index, metas: List[Dict[str, Any]], cfg: Dict[str, Any]):
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    with open(index_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for rec in metas:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(index_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def add_pdf_to_db(pdf_path: Path, index_dir: Path, model_name: str, chunk_tokens: int, overlap: int):

    checksum = sha256_file(pdf_path)  # 你原本就有的工具函式，會回傳 "sha256:xxxxx"
    existed = load_existing(index_dir)  # 你原本就有的讀取函式
    if existed and any(
        (rec.get("metadata", {}) or {}).get("checksum") == checksum
        for rec in existed["metas"]
    ):
        print(f"[SKIP] {pdf_path.name} 已存在（checksum 命中）")
        return
    

    emb = E5Embedder(model_name)
    tokenizer = emb.tokenizer

    # 讀 PDF -> 分頁文字 -> chunk
    pages = pdf_pages_text(pdf_path)
    chunks, metas = [], []
    checksum = sha256_file(pdf_path)
    running_local_id = 0
    for pageno, page_text in enumerate(pages, start=1):
        page_chunks = chunk_by_token(tokenizer, page_text, chunk_tokens, overlap)
        for ch in page_chunks:
            metas.append({
                "text": normalize_text_no_newline(ch),
                "metadata": {
                    "doc_id": pdf_path.name,
                    "checksum": checksum,
                    "source_uri": str(pdf_path.resolve())
                }
            })
            chunks.append(ch)
            running_local_id += 1

    if not chunks:
        print(f"[警告] {pdf_path.name} 沒有可用文字，略過。")
        return

    # 產生向量
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    safe_chunks = []
    for c in chunks:
        safe_chunks.extend(
            split_if_overlong(
                c, tokenizer, model_name,
                margin=32, prefix="passage: ", overlap_ratio=0.17
            )
        )
    chunks = safe_chunks

    # （可選）除錯：看切完後的最長 token 長度（含前綴與特殊符號），應 ≤ 512
    if chunks:
        max_len = max(len(tokenizer.encode("passage: " + c)) for c in chunks)
        print(f"最長序列長度（含前綴/特殊符號）：{max_len}")

    print(f"正在產生向量：{len(chunks)} 個 chunks（{emb.model_name} @ {emb.device}）")


    vecs = emb.embed_chunks(chunks)   # 若 embed_chunks 內部已加前綴，就用這行
    dim = vecs.shape[1]

    # 載入或建立索引
    existed = load_existing(index_dir)
    if existed is None:
        # 新建
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        all_metas = []
        cfg = {"model_name": emb.model_name, "dim": dim, "created_by": "pdf_embed_single.py"}
        next_id = 0
    else:
        index = existed["index"]
        all_metas = existed["metas"]
        cfg = existed["cfg"]
        if cfg["model_name"] != emb.model_name or cfg["dim"] != dim:
            raise ValueError(f"索引模型不一致：index 用 {cfg['model_name']} (dim={cfg['dim']}), 目前為 {emb.model_name} (dim={dim})")
        # 從既有 metadata 推算下一個 id
        existing_ids = [rec["id"] for rec in all_metas] if all_metas else []
        next_id = (max(existing_ids) + 1) if existing_ids else 0

    # 指派全域 id 並寫入
    ids = np.arange(next_id, next_id + len(vecs), dtype=np.int64)
    index.add_with_ids(vecs, ids)

    # 合併 metadata（加上 id）
    for i, rec in zip(ids.tolist(), metas):
        rec["id"] = i
        all_metas.append(rec)

    # 落盤
    save_index(index_dir, index, all_metas, cfg)

    # ---- Terminal Summary ----
    print("\n嵌入完成")
    print(f"PDF：{pdf_path.name}")
    print(f"新增 chunks：{len(vecs)}  筆（id 範圍：{ids[0]} ~ {ids[-1]}）")
    print(f"向量維度：{dim}")
    print(f"資料庫位置：{index_dir.resolve()}")
    print(f"目前索引總筆數：{index.ntotal}")

def main():
    import argparse
    from pathlib import Path
    import glob, re

    ap = argparse.ArgumentParser()
    # 說明文字改一下，但參數仍然只有一個 --pdf（保持相容）
    ap.add_argument("--pdf", default="D:/Py/Paper(RAG6)", type=str, help="PDF 檔路徑、資料夾、通配路徑（例如 D:\\docs\\*.pdf），也可用逗號分隔多個檔案")
    ap.add_argument("--index_dir", default="D:/Py/VectorDatabase", type=str, help="FAISS 索引與 JSONL 存放目錄")
    ap.add_argument("--model_name", default=MODELS_DIR/"multilingual-e5-base", type=str, help="embedding 模型")
    ap.add_argument("--chunk_tokens", default=480, type=int, help="每個 chunk 的最大 token 數")
    ap.add_argument("--overlap", default=80, type=int, help="chunk 之間的重疊 token")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    target = Path(args.pdf)

    # 蒐集要處理的 PDF 清單（僅在 main 做，其他函式不用改）
    pdf_list = []

    # 1) 若是資料夾：抓該夾內所有 .pdf（不遞迴；要遞迴可改成 rglob("*.pdf")）
    if target.is_dir():
        pdf_list = sorted([p for p in target.glob("*.pdf") if p.is_file()])

    else:
        # 2) 支援逗號分隔與萬用字元
        #    例如 --pdf "D:\a.pdf,D:\b.pdf" 或 --pdf "D:\docs\*.pdf"
        tokens = [t.strip() for t in re.split(r"[,\u3001;]", args.pdf) if t.strip()]
        for t in tokens:
            if any(ch in t for ch in "*?[]"):
                for g in glob.glob(t):
                    p = Path(g)
                    if p.suffix.lower() == ".pdf" and p.is_file():
                        pdf_list.append(p)
            else:
                p = Path(t)
                if p.is_dir():
                    pdf_list.extend(sorted([q for q in p.glob("*.pdf") if q.is_file()]))
                else:
                    pdf_list.append(p)

    # 去重、過濾不存在或副檔名不是 .pdf 的
    uniq = []
    seen = set()
    for p in pdf_list:
        try:
            p = p.resolve()
        except Exception:
            p = Path(p)
        if p.exists() and p.suffix.lower() == ".pdf" and p not in seen:
            uniq.append(p)
            seen.add(p)
        else:
            print(f"[WARN] 略過：{p}（不存在或非 PDF）")
    pdf_list = uniq

    if not pdf_list:
        raise FileNotFoundError(f"沒有可處理的 PDF，請確認 --pdf 的路徑/樣式：{args.pdf}")

    # 逐一處理
    print(f"將處理 {len(pdf_list)} 份 PDF：")
    for i, pdf_path in enumerate(pdf_list, 1):
        print(f"\n[ {i}/{len(pdf_list)} ] {pdf_path}")
        # 這裡呼叫你原本的單檔處理函式，不需要改其他程式碼
        add_pdf_to_db(pdf_path, index_dir, args.model_name, args.chunk_tokens, args.overlap)

    print("Done!")

if __name__ == "__main__":
    main()
