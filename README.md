1. 先在VSCode建立虛擬環境
2. 下載requirement.txt的modules "python -m pip install -r requirement.txt"
3. 更新pip "python -m pip install -U pip"
4. 下載transformer model(multilingual-e5-base) 可能需要申請開通huggingface的port
    "pip install -U huggingface_hub"  "huggingface-cli download intfloat/multilingual-e5-base --local-dir models/intfloat-multilingual-e5-base --local-dir-use-symlinks False"
5. 打包成exe
    "python -m pip install -U pyinstaller" 
    "pyinstaller --name RAG --windowed --onedir --add-data "logo.png;." --add-data "models;models" --collect-all sentence_transformers --hidden-import torch._C --hidden-import PIL._imagingtk --hidden-import RAG_Chunking --hidden-import RAG_Retrieval RAG_GUI.py"
6. 輸出完的exe檔會在dist資料夾(名稱為RAG.exe)，再來將RAG_Retrieval.py以及RAG_Chunking.py放入dist/_internal即可
