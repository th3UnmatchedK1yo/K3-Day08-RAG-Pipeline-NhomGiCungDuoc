# Finish indexing after BAAI/bge-m3 download

The embedding weights (`pytorch_model.bin`, ~2.2GB) download slowly on some networks.

Target file:

```text
.cache\huggingface\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181\pytorch_model.bin
```

Expected size: about **2,270,000,000+ bytes**.

When complete, run:

```bat
.venv\Scripts\python.exe -m src.task4_chunking_indexing
.venv\Scripts\python.exe -m src.task5_semantic_search
.venv\Scripts\python.exe -m src.task6_lexical_search
.venv\Scripts\python.exe -m src.task9_retrieval_pipeline
.venv\Scripts\python.exe -m group_project.evaluation.calibrate_threshold
.venv\Scripts\python.exe -m group_project.evaluation.eval_pipeline
.venv\Scripts\python.exe -m pytest tests -v
streamlit run app.py
```

Resume download if interrupted:

```bat
curl.exe -L -C - -o .cache\huggingface\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181\pytorch_model.bin https://huggingface.co/BAAI/bge-m3/resolve/main/pytorch_model.bin
```
