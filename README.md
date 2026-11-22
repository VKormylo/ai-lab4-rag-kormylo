# ai-lab4-rag-kormylo

Лабораторна робота №4 «Побудова рішення пошуково-доповненої генерації (RAG) на базі LLM»
з дисципліни «Технології ШІ в інженерії даних».

- **Виконав:** Кормило Віталій, група ПЗ-47
- **Варіант:** 5
- **Документ:** Biology 2e (OpenStax), https://openstax.org/details/books/biology-2e
  ([PDF](https://assets.openstax.org/oscms-prodcms/media/documents/Biology-2e_-_WEB.pdf), 1475 сторінок).
  Документ має понад 400 сторінок, тому використано неперервну частину: **глави 1–5, сторінки 29–180 повного PDF (152 сторінки)**.
  Частину вирізає скрипт `src/extract_part.py` у файл `data/document.pdf`. Повний PDF (400 МБ) у репозиторій не додається.
  Усюди в роботі використовуються номери сторінок файлу `data/document.pdf` (1–152).
- **Модель ембедингів:** `intfloat/multilingual-e5-small` (d = 384, max_seq_length = 512, префікси `query: ` / `passage: `)
- **Чанки A; B:** 256/32; 512/64 токенів
- **LLM:** `qwen2.5:3b` (Ollama), temperature = 0, seed = 5
- **Додатковий експеримент:** Е10, тобто індекс і запити з префіксами e5 (`query:`/`passage:`) і без них

## Запуск

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
ollama pull qwen2.5:3b

python src/ingest.py              # крок 2
python src/chunking.py            # крок 3
python src/index.py               # крок 4
python src/eval_retrieval.py      # крок 6
python src/rag.py                 # крок 7
python src/eval_answers.py        # крок 8
python src/experiment.py          # крок 8, експеримент Е10
python src/git_timeline.py        # крок 10
```

Усі скрипти запускаються з кореневого каталогу репозиторію, графіки зберігаються в `reports/figures/`.
