# src/config.py
VARIANT = 5
DOC_PATH = "data/document.pdf"
EMB_MODEL = "intfloat/multilingual-e5-small"
QUERY_PREFIX, PASSAGE_PREFIX = "query: ", "passage: "  # модель e5 потребує префіксів
CHUNKS = {"A": (256, 32), "B": (512, 64)}  # (chunk_size, chunk_overlap) у токенах
LLM_MODEL = "qwen2.5:3b"  # модель Ollama
CONFIG = "B"  # краща конфігурація за кроком 6 (B-dense: Hit@5 = 1,00)
TOP_K = 5  # початок плато кривої Hit@k для B-dense (крок 6)
THRESHOLD = 0.847  # поріг відмови за розподілом top-1 конфігурації B (крок 6)
SEED = VARIANT
