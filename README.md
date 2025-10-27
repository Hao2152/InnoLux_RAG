# InnoLux_RAG
## Retrieval-Augmented Generation (RAG) System for Offline Corporate Knowledge Management

## Introduction
InnoLux_RAG 是一個離線運行的文件檢索與擴增式生成系統
此專案整合了 PDF 文件嵌入 __（RAG_Chunking）__、多模混合檢索 __（RAG_Retrieval）__ 與 圖形介面操作 __（RAG_GUI）__，可在無網路環境下完成從文件向量化、索引管理到語義檢索與 Prompt 組裝的全流程

## 專案架構
```
InnoLux_RAG/
├── RAG_GUI.py           # 主 GUI：整合檢索與嵌入流程
├── RAG_Chunking.py      # 文件切片與向量化嵌入
├── RAG_Retrieval.py     # 檢索
├── models/
│   └── intfloat-multilingual-e5-base/  # 預設嵌入模型（SentenceTransformer）
├── Paths.py             # 共用路徑設定（BASE, MODELS_DIR, DATA_DIR 等）
├── logo.png             # GUI 標誌（Innolux）
└── output/              # 建立的向量資料庫與 JSONL metadata
```

## 系統功能說明
| 模組                | 功能                           |
| ----------------- | ---------------------------     |
| **RAG_Chunking**  | 將 PDF 轉為文本片段並建立向量資料庫  |
| **RAG_Retrieval** | 對現有資料庫進行檢索與 Prompt 組裝  |
| **RAG_GUI**       | 使用者圖形介面                    | 

## RAG_Chunking
### 目標
將多份 PDF 逐頁讀取、以 token 上限與重疊策略切塊，再用 E5 多語模型轉成向量，寫入 index.faiss + metadata.jsonl + config.json，並以檔案 SHA256 判斷是否重複寫入
### 流程
```
Input : [PDF 檔案 / 資料夾]

[Step 1] 讀取與清理文字（PyMuPDF）
[Step 2] 雙層切塊（粗切+細切）
[Step 3] Embedding 向量化（Sentence Transformer）
[Step 4] 建立或更新Index
[Step 5] 保存 index.faiss / metadata.jsonl / config.json
[可供 RAG_Retrieval 使用的 Vector DB]
```
- [1] 讀取與清理文字（PyMuPDF）
    - 使用 fitz.open() 開啟 PDF
    - 逐頁抓取純文字
    - 呼叫 normalize_text()處理多餘空白、換行、頁碼
- [2] 雙層切塊（粗切 + 細切）
    - [2-1]Chunking(粗切)
        - 先依段落 (\n\n) 切分
        - 使用 Tokenizer 估算長度，超過 chunk_tokens（預設 480）時進行切分
        - 在切割處插入 overlap token（預設 80），確保語意連貫
    - [2-2]安全切分(細切)
        - 針對每一段再檢查是否超過模型上限（E5 Model 約 512 tokens） 
        - 若超過，依比例重疊切開
- [3] Embedding
    - 自動偵測 GPU / CPU
    - 將拆下來的
- [4] 建立或更新Index
    - 載入 index.faiss（若無則建立新的Index）
    - 若模型名與維度不符則 __報錯__（防止資料不一致）。
        - 也就是說一個資料庫的維度、Embedding模型必須一樣
    - 對每個新向量給予連續的 ID
    - 同步更新 metadata，每筆記錄都附上 id
- [5] 儲存結果
 
| 檔案               | 內容                | 用途                 |
| ---------------- | ----------------- | ------------------ |
| `index.faiss`    | 向量index              | RAG_Retrieval 載入檢索 |
| `metadata.jsonl` | 每行一段的文字與 metadata | 查詢時顯示片段內容          |
| `config.json`    | 模型名稱、維度、建立工具資訊    | 驗證寫入向量的一致性              |
### Index.faiss
存放每一個chunk的id與對應的向量，其中chunk_id也對應metadata中的id
|欄位名稱|說明|
|-------------|---------|
|chunk_id|每個chunk的連續編碼|
|vector|存放每個chunk編碼後的向量|
### metadata.jsonl
| 欄位名稱                                              | Data Type  | 說明  |
| ------------------------------------------------- | --- | ---------------------- |
| `id`                                              | int | 全域唯一 ID（對應 FAISS 向量索引） |
| `text`                                            | str | 經過正規化與安全切分後的文本內容       |
| `doc_id`                                          | str | 原始 PDF 檔名（不含路徑）        |
| `checksum`                                        | str | SHA-256 雜湊，用於避免重複檔案    |
| `source_uri`                                      | str | PDF 絕對路徑               |
| `page_no`                                         | int | 來源頁碼（從 1 開始）           |
| `chunk_no`                                        | int | 此頁的粗分段編號               |
| `piece_no`                                        | int | 安全切分後的細分段編號            |
| `local_seq`                                       | int | 在整份文件中的局部遞增序號          |

### config.json
存放資料庫所用的模型資訊（模型名稱、維度）

## Funcitons
- [ ] normalize_text_no_newline(s) -> str
    - [ ] 把字串中的實體換行與字面上的 \n全部轉成空白，並壓縮所有連續空白，最後 strip()
    - [ ] 對Chunk做處理
- [ ] split_if_overlong(text, tokenizer, model_name, margin, prefix, overlap_ratio) -> list[str]
    - [ ] 若 text 經 tokenizer 編碼後超過模型允許的 token 長度(512)就切段並保留重疊
- [ ] sha256_file(path) -> str
    - [ ] 計算整個檔案內容的 SHA-256 編碼
- [ ] normalize_text(t) -> str
    - [ ] 頁面文字正規化：移除 \r、修正英文連字斷行（-\n）、壓縮空白與空行、修行尾空白、刪除純頁碼行
    - [ ] 對頁面做處理
- [ ] pdf_pages_text(pdf_path) -> List[str]
    - [ ] 用 PyMuPDF 開啟 PDF，每頁取 __純文字__，每頁套用 normalize_text()
    - [ ] 關檔後回傳每頁一段的文字陣列。
- [ ] chunk_by_token(tokenizer, page_text, chunk_tokens, overlap) -> List[str]
    - [ ] 先找出段落，每段累加其 token 數
    - [ ] 若該段落將超過 chunk_tokens，就把目前先送出為一個 chunk，並將下一個 chunk在開頭回貼 overlap 個 token 形成重疊，再開始累積下一個 chunk
- [ ] load_existing(index_dir) -> Optional[dict]
    - [ ] 檢查 index.faiss、metadata.jsonl、config.json 是否同時存在
    - [ ] 若存在，讀取 index、metadata、config，以 dict 形式回傳
- [ ] save_index(index_dir, index, metas, cfg) -> None
    - [ ] 將chunk後的結果寫入 index.faiss、metadata.jsonl、與 config.json
- [ ] add_pdf_to_db(pdf_path, index_dir, model_name, chunk_tokens, overlap) -> None
    - [ ] 將讀取到的pdf file做處理，並寫入資料庫

## RAG_Retrieval
### 目標
從現有的向量資料庫（index.faiss + metadata.jsonl）中，根據使用者的查詢（query）搜尋出最相關的文本片段，並自動組合成 LLM 可使用的 Retrieval-Augmented Prompt
```
Input : 使用者輸入查詢（query）

[Step 1] 載入資料庫 (index.faiss + metadata.jsonl)       
[Step 2] metadata載入與清理       
[Step 3] 建立檢索器（BM25 + 向量檢索） 並檢索         
[Step 4] 結合排序       
[Step 5] 句子級擷取 Snippet + 文獻過濾      
[Step 6] 組合 Augmented Prompt 輸出
```

- [1] 載入資料庫
    - 建立 id → text 與 id → metadata 對應表，這樣後面 FAISS 回傳的 ID 才能找回文字內容
- [2] metadata載入與處理
    - 針對每一筆紀錄做 flatten，確保每個 key 都能直接取用
    - 程式載入資料庫完成後，會建立以下結構
    - 查詢時 FAISS 回傳的 id 會透過這個對應表回找文字與來源
        - corpus_texts：純文字陣列
        - corpus_lookup：id → 資料的lookup table
- [3] 建立檢索器並檢索
    - BM25 模型(評估文件與查詢字串之間的相關性)
    - 向量檢索（FAISS）
        - 用同樣的 SentenceTransformer 模型把 user query 轉成 embedding
        - 搜尋最相似的向量（以內積 / cosine 相似度計算）
- [4] 結合排序
    - 結合BM25與向量相似度計算的結果來評估檢索結果的正確性
- [5] 句子級擷取 Snippet + 文獻過濾
    - 對於每個命中的 chunk 依中英標點切成句子，比較 query 與各句子的語意相似度，並且移除那些看起來像參考文獻的內容
- [6] 組合 Augmented Prompt 輸出
    - 將檢索結果組合成一份可直接丟給 LLM 的提示
    - 例如 : 
        - \# 角色
            你是專業的技術研究助理。請根據「已檢索到的內容」回答以及你「另外的理解」回答，分為兩部分；若內容不足300字，請明確說明缺少資訊並提出可追問的關鍵點。避免編造。

        - \#已檢索到的內容
            \內容\

        - \# 使用者問題
            \使用者輸入的問題\

        - \# 回答要求
             優先使用檢索內容中的觀點與數據，必要時可整合多段內容。
             若不同段落有衝突，請指出衝突並給出最合理的解釋。
## Functions
- [ ] looks_like_bib_by_snippet(text, user_query, max_chars) -> bool
    - [ ] 判斷一段文字（例如 PDF 頁面內容）是否像參考文獻(References)，會先用_pick_best_sentence() 找出與查詢最相關的句子片段，再用 looks_like_bibliography() 分析。若判定像文獻，會在檢索時過濾掉
- [ ] _split_sentences(text) -> List[str]
    - [ ] 將段落依據中英標點符號（例如 。！？!?;:）切分成句子。若整段沒標點但太長（>140字），則每100字切一次。輸出句子列表
- [ ] _tokenize(s) -> str
    - [ ] 將句子轉為小寫，比對用
- [ ] _grow_span_around(sents, center, min_chars, max_chars) -> str
    - [ ] 若某句太短，從該句為中心向左右擴張，直到達到最小字數（不超過最大字數），用於擴展 snippet
    - [ ] 當 _pick_best_sentence() 找到最相關的句子太短時，用來“往上下擴張”句子範圍，使片段更完整、語意更連貫
- [ ] _simple_tokens(t) -> List[str]
    - [ ] 把中英文與數字以非字母符號分開，用於重疊率評估或關鍵詞比對
- [ ] make_snippet(text, maxlen) -> str
    - [ ] 壓縮空白、多餘換行，並將過長文字截斷加上「…」，用於製作簡要摘要
- [ ] _extract_query_terms(q) -> List[str]
    - [ ] 拆分使用者輸入
    - [ ] 例如 : "我是工程師" → "我"、"是"、"工程師"
- [ ] mask_snippet(snippet, query, left='【' , right='】') -> str
    - [ ] 對 snippet 中命中查詢詞的部分加上標記
    - [ ] 例如 : "我是工程師" → "【我】"、"是"、"【工程師】"
    - [ ] __GUI執行時未用到，未來可擴充__
- [ ] _kw_bonus(sent) -> float
    - [ ] 若句子中包含與標記相關的關鍵詞，給予加分（+0.05），幫助 _pick_best_sentence() 優先選擇關鍵描述
- [ ] _pick_best_sentence(user_query, text, embedder, min_chars, max_chars) -> str
    - [ ] 計算 query 與各句 cosine 相似度選最高者
    - [ ] 若句子太短則呼叫 _grow_span_around() 擴張上下文。輸出一段完整、語意相關的句子摘要
- [ ] load_corpus(jsonl_path) -> List[Dict]
    - [ ] 讀取metadata.jsonl，將每一行攤平成平面結構
- [ ] BM25.search(query, topk, k1, b)
    - [ ] 根據查詢 query 在一組文件中找出最相關的前 topk 筆 Chunks
    - [ ] k1, b為BM25模型的參數
- [ ] faiss_search(query, index, embedder, topk, query_prefix) -> List[tuple]
    - [ ] 以向量檢索方式搜尋 query，將 query encode 為 embedding，搜尋 index
- [ ] normalize(scores) -> Dict[int, float]
    - [ ] 正規化到 0~1 區間，用於混合分數融合
- [ ] hybrid_merge(bm25_res, vec_res, w_bm25, w_vec, topk)
    - [ ] 將 BM25 與 FAISS 結果按比例融合，並重新排序取前 k 筆。
- [ ] looks_like_bibliography(t) -> bool
    - [ ] 用多種規則判斷文字是否像參考文獻區（包含 “references”, “doi:”, “[1]”, “Proc.” 等樣式）
- [ ] compose_augmented_prompt(snippet_chars, user_query, docs, ranked, id2row, highlight) -> str
    - [ ] 依照檢索結果組合成最終Augmented Prompt
- [ ] format_references(docs, ranked, id2row) -> str
    - [ ] 將檢索結果整理為 [參考資料] 清單，列出每段來源檔名、頁碼與分數

---
 __GUI執行時未用到，未來可擴充__
- [ ] expand_queries(user_query) -> List[str]
    - [ ] 根據原始查詢產生多個擴充查詢，提升檢索覆蓋率
- [ ] page_key(d) -> Tuple[str, Optional[int]]
    - [ ] 從一個文件資料 d 中提取出 __檔案名稱__ 和 __頁碼__，並組成一個 tuple 作為 key


# 環境建置
1. 先在VSCode建立虛擬環境
2. 下載requirement.txt的modules "python -m pip install -r requirement.txt"
3. 更新pip "python -m pip install -U pip"
4. 下載transformer model(multilingual-e5-base) 
    可能需要申請開通huggingface的port 
    "pip install -U huggingface_hub" 
    "huggingface-cli download intfloat/multilingual-e5-base --local-dir models/intfloat-multilingual-e5-base --local-dir-use-symlinks False"
5. 打包成exe 
    "python -m pip install -U pyinstaller" 
    "pyinstaller --name RAG --windowed --onedir --add-data "logo.png;." --add-data "models;models" --collect-all sentence_transformers --hidden-import torch._C --hidden-import PIL._imagingtk --hidden-import RAG_Chunking --hidden-import RAG_Retrieval RAG_GUI.py"
6. 輸出完的exe檔會在dist資料夾(名稱為RAG.exe)，再來將RAG_Retrieval.py以及RAG_Chunking.py放入dist/_internal即可