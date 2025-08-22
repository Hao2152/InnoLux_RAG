"""
PDF檔案名稱可中/英文
PDF資料夾路徑以及VectorDatabase資料夾路徑不可中文
PDF內容可中文
"""
from __future__ import annotations
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import RAG_Chunking
import RAG_Retrieval
import re
import sys
import threading
import queue
from pathlib import Path
import traceback
import importlib.util
import json
from Paths import BASE, MODELS_DIR, DATA_DIR, rp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 預設參數（檢索頁面）
DEFAULT_SNIPPET_CHARS = 420
W_BM25 = 0.5
W_VEC = 0.5

# 固定模組路徑
RETRIEVAL_DEFAULT = Path(BASE/"RAG_Retrieval.py").resolve()
CHUNKING_DEFAULT = Path(BASE/"RAG_Chunking.py").resolve()
model_path = str(MODELS_DIR / "intfloat-multilingual-e5-base")
# ---- Path 檢查工具 ----
_CJK_REGEX = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF]")

def has_non_ascii_or_cjk(s: str) -> bool:
    s = str(s)
    return any(ord(ch) > 127 for ch in s) or bool(_CJK_REGEX.search(s))

# =============== 工具：動態載入 .py 模組 ===============
def load_module_from_path(py_path: Path, module_name: str):
    """從檔案路徑載入 Python 模組。"""
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"無法載入模組：{py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# =============== GUI 主體 ===============
class RAGApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAG（Retrieval & Embedding）")
        self.geometry("980x800")
        header = tk.Frame(self)
        header.pack(side="top", fill="x")


        logo_path = Path(__file__).resolve().parent / "logo.png"
        try:
            self.logo_img = tk.PhotoImage(file=str(logo_path))
        except Exception as e:
            print("[badge] load failed:", e)
            self.logo_img = None

        # 3) 顯示Innolux
        if self.logo_img:
            self.badge = tk.Label(header, image=self.logo_img, bd=0, highlightthickness=0)
            self.badge.pack(side="left", padx=4, pady=2)
            self.badge.lift()  # 保證在最上層
        
        # 狀態
        self.mod_chunking = None  # RAG_Chunking 模組
        self.mod_retrieval = None  # RAG_Retrieval 模組
        self.log_queue = queue.Queue()

        # Notebook（兩個分頁）
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.tab_search = ttk.Frame(nb)
        self.tab_embed = ttk.Frame(nb)
        nb.add(self.tab_search, text="Retrieval")
        nb.add(self.tab_embed, text="Embedding")

        # 先建立 UI 骨架（會用到顯示模組路徑的 Label 變數）
        self.retrieval_path_var = tk.StringVar(value=str(RETRIEVAL_DEFAULT))
        self.chunking_path_var = tk.StringVar(value=str(CHUNKING_DEFAULT))

        self._build_tab_search()
        self._build_tab_embed()

        # 底部狀態列
        self.status_var = tk.StringVar(value="就緒")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", padx=6, pady=3)

        # 自動載入固定模組
        self._autoload_modules()

        self.after(150, self._poll_log_queue)

    # --------------- 自動載入Embedding & Retrieval模組 ---------------
    def _autoload_modules(self):
        errs = []
        # 載入檢索模組
        if RETRIEVAL_DEFAULT.exists():
            try:
                self.mod_retrieval = load_module_from_path(RETRIEVAL_DEFAULT, "user_RAG_Retrieval")
            except Exception as e:
                errs.append(f"載入 {RETRIEVAL_DEFAULT.name} 失敗：{e}")
        else:
            errs.append(f"找不到Retrieval Module：{RETRIEVAL_DEFAULT}")
        # 載入嵌入模組
        if CHUNKING_DEFAULT.exists():
            try:
                self.mod_chunking = load_module_from_path(CHUNKING_DEFAULT, "user_RAG_Chunking")
            except Exception as e:
                errs.append(f"載入 {CHUNKING_DEFAULT.name} 失敗：{e}")
        else:
            errs.append(f"找不Embedding Module：{CHUNKING_DEFAULT}")

        if errs:
            messagebox.showerror("Module載入錯誤", "".join(errs))
            self.status_var.set("Module載入失敗，請確認檔案存在於固定路徑。")
        else:
            self.status_var.set("已從固定路徑載入模組。")

    # --------------- Tab 1：Retrieval Page Setting---------------
    def _build_tab_search(self):
        frm = ttk.Frame(self.tab_search, padding=10)
        frm.pack(fill="both", expand=True)

        # 固定模組資訊（唯讀）
        box_mod = ttk.LabelFrame(frm, text="Retrieval Module（固定路徑）")
        box_mod.pack(fill="x", pady=6)
        ttk.Label(box_mod, textvariable=self.retrieval_path_var).pack(fill="x", padx=6, pady=6)

        # 索引與檢索設定（已移除：模型路徑、Snippet、權重）
        box_cfg = ttk.LabelFrame(frm, text="向量資料庫與搜尋筆數設定")
        box_cfg.pack(fill="x", pady=6)

        # 索引資料夾（內含 index.faiss / metadata.jsonl / config.json）
        ttk.Label(box_cfg, text="向量資料庫資料夾：").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.entry_index_dir = tk.Entry(box_cfg)
        self.entry_index_dir.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(box_cfg, text="選擇...", command=self._choose_index_dir).grid(row=0, column=2, padx=3)

        # Top-K
        ttk.Label(box_cfg, text="Top-K：").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        self.spin_topk = tk.Spinbox(box_cfg, from_=1, to=50, width=6)
        self.spin_topk.delete(0, "end"); self.spin_topk.insert(0, "5")
        self.spin_topk.grid(row=1, column=1, sticky="w", padx=6, pady=6)

        for c in range(3):
            box_cfg.grid_columnconfigure(c, weight=1)

        # 查詢與輸出
        box_q = ttk.LabelFrame(frm, text="查詢")
        box_q.pack(fill="x", pady=6)

        self.txt_query = tk.Text(box_q, height=4)
        self.txt_query.pack(fill="x", padx=6, pady=6)
        ttk.Button(box_q, text="檢索並生成 Prompt", command=self._on_search_prompt).pack(padx=6, pady=6, anchor="e")

        box_out = ttk.LabelFrame(frm, text="整合後的 Prompt（可直接複製）")
        box_out.pack(fill="both", expand=True, pady=6)
        self.txt_prompt = tk.Text(box_out, height=12, wrap="word")
        self.txt_prompt.pack(fill="both", expand=True, padx=6, pady=6)

        box_ref = ttk.LabelFrame(frm, text="參考來源/分數")
        box_ref.pack(fill="x", pady=6)
        self.txt_refs = tk.Text(box_ref, height=6)
        self.txt_refs.pack(fill="x", padx=6, pady=6)

        btns = ttk.Frame(frm)
        btns.pack(fill="x")
        ttk.Button(btns, text="複製 Prompt", command=self._copy_prompt).pack(side="left", padx=6, pady=6)
        ttk.Button(btns, text="儲存 Prompt...", command=self._save_prompt).pack(side="left", padx=3, pady=6)

    # --------------- Tab 2：Embedding Page Setting---------------
    def _build_tab_embed(self):
        frm = ttk.Frame(self.tab_embed, padding=10)
        frm.pack(fill="both", expand=True)

        # 固定模組資訊（唯讀）
        box_mod = ttk.LabelFrame(frm, text="Embbeding Module（固定路徑）")
        box_mod.pack(fill="x", pady=6)
        ttk.Label(box_mod, textvariable=self.chunking_path_var).pack(fill="x", padx=6, pady=6)

        # 基本設定
        box_cfg = ttk.LabelFrame(frm, text="輸出向量資料庫設定")
        box_cfg.pack(fill="x", pady=6)

        ttk.Label(box_cfg, text="向量資料庫資料夾（輸出到此 / 路徑請用英文）：").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.entry_out_index = tk.Entry(box_cfg)
        self.entry_out_index.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(box_cfg, text="選擇...", command=self._choose_out_index).grid(row=0, column=2, padx=3)

        for c in range(4):
            box_cfg.grid_columnconfigure(c, weight=1)

        # 目標檔案/資料夾
        box_targets = ttk.LabelFrame(frm, text="目標（可加入單一 PDF 或整個資料夾 / 內容或路徑名稱可中英文）")
        box_targets.pack(fill="both", expand=False, pady=6)

        btns = ttk.Frame(box_targets)
        btns.pack(fill="x")
        ttk.Button(btns, text="加入 PDF 檔...", command=self._add_pdf_files).pack(side="left", padx=3, pady=6)
        ttk.Button(btns, text="加入資料夾...", command=self._add_pdf_folder).pack(side="left", padx=3, pady=6)
        ttk.Button(btns, text="清空", command=self._clear_targets).pack(side="left", padx=3, pady=6)

        self.list_targets = tk.Listbox(box_targets, height=8)
        self.list_targets.pack(fill="both", expand=True, padx=6, pady=6)

        # 執行區
        box_run = ttk.LabelFrame(frm, text="執行與狀態")
        box_run.pack(fill="both", expand=True, pady=6)

        self.prog = ttk.Progressbar(box_run, mode="indeterminate")
        self.prog.pack(fill="x", padx=6, pady=6)

        self.txt_log = tk.Text(box_run, height=12)
        self.txt_log.pack(fill="both", expand=True, padx=6, pady=6)

        ttk.Button(frm, text="開始執行", command=self._start_embedding).pack(pady=6, anchor="e")

    # --------------- 動作：Retrieval & Prompting ---------------
    def _choose_index_dir(self):
        d = filedialog.askdirectory(title="選擇向量資料庫資料夾（含 index.faiss / metadata.jsonl）")
        if d:
            self.entry_index_dir.delete(0, "end")
            self.entry_index_dir.insert(0, d)
        
        if not d:  # 使用者取消
            return

        d = str(Path(d))  # 正規化一下字串路徑
        if has_non_ascii_or_cjk(d):
            messagebox.showwarning(
                "路徑提醒",
                "所選路徑含中文或全形字元，建議改用只含英文/數字的路徑，以避免讀寫失敗。"
            )
            return  # 只提醒，不加入清單

    def _guess_model_from_config(self, index_dir: Path):
        cfg = index_dir / "config.json"
        if not cfg.exists():
            return None
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 常見鍵名
            for k in [
                "model_name", "embedding_model", "embed_model",
                "sentence_transformers_model", "sentence_model",
                "encoder", "model", "name"
            ]:
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        except Exception:
            pass
        return None

    def _on_search_prompt(self):
        if self.mod_retrieval is None:
            messagebox.showwarning("提示", f"未載入Retrieval Module，請確認存在：{RETRIEVAL_DEFAULT}")
            return
        index_dir = Path(self.entry_index_dir.get().strip())
        if not index_dir.exists():
            messagebox.showwarning("提示", "向量資料庫資料夾不存在")
            return
        try:
            topk = int(self.spin_topk.get())
        except Exception:
            messagebox.showwarning("提示", "請確認 Top-K 數值無誤")
            return
        query = self.txt_query.get("1.0", "end").strip()
        if not query:
            messagebox.showwarning("提示", "請先輸入問題內容")
            return

        try:
            # 載入語料（metadata.jsonl）
            meta_path = index_dir / "metadata.jsonl"
            docs = self.mod_retrieval.load_corpus(meta_path)
            # BM25
            bm25 = self.mod_retrieval.BM25(docs)
            bm25_res = bm25.search(query, topk=topk)

            # FAISS & 向量檢索（若缺少或模型無法載入，則只用 BM25）
            vec_res = []
            index = self.mod_retrieval.try_load_faiss(index_dir / "index.faiss")
            embedder = None
            if index is not None:
                model_guess = self._guess_model_from_config(index_dir)
                if model_guess:
                    embedder = self.mod_retrieval.try_load_embedder(model_guess)
                if embedder is not None:
                    vec_res = self.mod_retrieval.faiss_search(query, index, embedder, topk=topk, query_prefix="query: ")

            # 融合排序（固定權重）
            if vec_res:
                ranked = self.mod_retrieval.hybrid_merge(bm25_res, vec_res, w_bm25=W_BM25, w_vec=W_VEC, topk=topk)
            else:
                ranked = bm25_res

            # 組裝 Prompt（固定 snippet 字元上限）
            aug = self.mod_retrieval.compose_augmented_prompt(DEFAULT_SNIPPET_CHARS, query, docs, ranked)
            refs = self.mod_retrieval.format_references(docs, ranked)

            self.txt_prompt.delete("1.0", "end")
            self.txt_prompt.insert("1.0", aug.strip())

            self.txt_refs.delete("1.0", "end")
            self.txt_refs.insert("1.0", refs.strip())
            self.status_var.set("檢索完成。")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("檢索失敗", f"{e}")

    def _copy_prompt(self):
        s = self.txt_prompt.get("1.0", "end").strip()
        if not s:
            return
        self.clipboard_clear()
        self.clipboard_append(s)
        self.status_var.set("已複製到剪貼簿。")

    def _save_prompt(self):
        s = self.txt_prompt.get("1.0", "end").strip()
        if not s:
            return
        path = filedialog.asksaveasfilename(title="儲存 Prompt", defaultextension=".txt", filetypes=[["Text", ".txt"],["All", ".*"]])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
            self.status_var.set(f"已儲存：{path}")

    # --------------- 動作：Embedding ---------------
    
    def _choose_out_index(self):
        d = filedialog.askdirectory(title="選擇索引輸出資料夾")
        if d:
            self.entry_out_index.delete(0, "end")
            self.entry_out_index.insert(0, d)

        if not d:  # 使用者取消
            return

        d = str(Path(d))  # 正規化一下字串路徑
        if has_non_ascii_or_cjk(d):
            messagebox.showwarning(
                "路徑提醒",
                "所選路徑含中文或全形字元，建議改用只含英文/數字的路徑，以避免讀寫失敗。"
            )
            return  # 只提醒，不加入清單

    def _choose_model_dir_embed(self):
        d = filedialog.askdirectory(title="選擇模型資料夾（離線）")
        if d:
            self.entry_embed_model.delete(0, "end")
            self.entry_embed_model.insert(0, d)

    def _add_pdf_files(self):
        files = filedialog.askopenfilenames(title="選擇 PDF", filetypes=[["PDF", ".pdf"], ["All", ".*"]])
        for p in files:
            self.list_targets.insert("end", p)

    def _add_pdf_folder(self):
        d = filedialog.askdirectory(title="選擇資料夾（將處理其中 *.pdf）")
        self.list_targets.insert("end", d)

    def _clear_targets(self):
        self.list_targets.delete(0, "end")

    def _start_embedding(self):
        if self.mod_chunking is None:
            messagebox.showwarning("提示", f"未載入Embedding Module，請確認存在：{CHUNKING_DEFAULT}")
            return
        out_dir = Path(self.entry_out_index.get().strip())
        if not out_dir:
            messagebox.showwarning("提示", "請先指定索引輸出資料夾")
            return
        #model_path = "intfloat/multilingual-e5-base"
        targets = [self.list_targets.get(i) for i in range(self.list_targets.size())]
        if not targets:
            messagebox.showwarning("提示", "請加入 PDF 檔或資料夾")
            return

        # 開執行緒跑嵌入
        self.prog.start(10)
        self.status_var.set("正在執行嵌入...")
        t = threading.Thread(target=self._run_embedding_worker, args=(targets, out_dir, model_path), daemon=True)
        t.start()

    def _run_embedding_worker(self, targets, out_dir: Path, model_path: str):
        def log(msg: str):
            self.log_queue.put(msg)
        try:
            added = 0
            out_dir.mkdir(parents=True, exist_ok=True)
            # 展開目標：若是資料夾 -> 取其中的 *.pdf
            pdfs = []
            for t in targets:
                p = Path(t)
                if p.is_dir():
                    pdfs.extend(sorted([q for q in p.glob("*.pdf") if q.is_file()]))
                elif p.is_file() and p.suffix.lower() == ".pdf":
                    pdfs.append(p)
            if not pdfs:
                raise FileNotFoundError("找不到任何 PDF")

            log(f"將處理 {len(pdfs)} 份 PDF：")
            for i, pdf_path in enumerate(pdfs, 1):
                log(f"[ {i}/{len(pdfs)} ] {pdf_path}")
                try:
                    chunk_tokens = 480
                    overlap = 80
                    self.mod_chunking.add_pdf_to_db(pdf_path=pdf_path,
                                                    index_dir=out_dir,
                                                    model_name=model_path,
                                                    chunk_tokens=chunk_tokens,
                                                    overlap=overlap)
                    added += 1
                except Exception as e:
                    log(f"[錯誤] {pdf_path.name}: {e}")
            log("完成。實際處理：%d / %d" % (added, len(pdfs)))
            self.status_var.set("嵌入完成")
            self.list_targets.delete(0,"end")
        except Exception as e:
            self.status_var.set("嵌入失敗")
            self.log_queue.put("[致命錯誤]" + traceback.format_exc())
            self.after(0, lambda: messagebox.showwarning("嵌入失敗", f"{pdf_path.name}\n{e}"))
        finally:
            self.after(0, self.prog.stop)

    # --------------- log queue 輪詢 ---------------
    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.txt_log.insert("end", msg + "\n")
                self.txt_log.see("end")
        except queue.Empty:
            pass
        self.after(150, self._poll_log_queue)


if __name__ == "__main__":
    app = RAGApp()
    app.mainloop()
