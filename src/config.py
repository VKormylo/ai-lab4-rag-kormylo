# src/config.py
VARIANT = 5
DOC_PATH = "data/document.pdf"
EMB_MODEL = "intfloat/multilingual-e5-small"
QUERY_PREFIX, PASSAGE_PREFIX = "query: ", "passage: "  # модель e5 потребує префіксів
CHUNKS = {"A": (256, 32), "B": (512, 64)}  # (chunk_size, chunk_overlap) у токенах
LLM_MODEL = "qwen2.5:3b"  # модель Ollama
TOP_K = 4  # кількість чанків у контексті; уточнюється на кроці 6
THRESHOLD = 0.0  # поріг відмови; уточнюється на кроці 6
SEED = VARIANT
