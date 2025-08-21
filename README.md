Pack to exe
"pyinstaller --name RAG --windowed --onedir --add-data "logo.png;." --add-data "models;models" --collect-all sentence_transformers --hidden-import torch._C --hidden-import PIL._imagingtk --hidden-import RAG_Chunking --hidden-import RAG_Retrieval RAG_GUI.py"
