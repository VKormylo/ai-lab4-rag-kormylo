# src/eval_retrieval.py
import csv, re
import numpy as np
import matplotlib.pyplot as plt
from rank_bm25 import BM25Okapi
from index import get_embeddings, load_chunks, load_store
from config import CHUNKS, CONFIG, THRESHOLD

K_MAX = 10


def load_questions(path="eval/questions.csv"):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["gold"] = {int(p) for p in r["pages"].split(";") if p.strip()}
    return rows


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def bm25_ranking(bm25, ids, q, k=K_MAX):
    scores = bm25.get_scores(tokenize(q))
    return [ids[i] for i in np.argsort(scores)[::-1][:k]]


def rrf(*rankings, k=60):
    score = {}
    for ranking in rankings:
        for r, cid in enumerate(ranking, start=1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + r)
    return sorted(score, key=score.get, reverse=True)


def hit_at_k(pages, gold, k):
    return float(any(p in gold for p in pages[:k]))


def reciprocal_rank(pages, gold):
    return next((1.0 / r for r, p in enumerate(pages, 1) if p in gold), 0.0)


def first_rank(pages, gold):
    """Позиція першого релевантного чанка (None, якщо немає в top-K_MAX)."""
    return next((r for r, p in enumerate(pages, 1) if p in gold), None)


if __name__ == "__main__":
    qs = load_questions()
    answerable = [q for q in qs if q["gold"]]
    emb = get_embeddings()
    rankings = {}
    top1 = {name: {"answerable": [], "unanswerable": []} for name in CHUNKS}
    top1_by_q = {}
    for name in CHUNKS:
        chunks = load_chunks(name)
        page_of = {c["chunk_id"]: c["page"] for c in chunks}
        ids = [c["chunk_id"] for c in chunks]
        bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        store = load_store(name, emb)
        for q in qs:
            hits = store.similarity_search_with_score(q["question"], k=K_MAX)
            dense = [d.metadata["chunk_id"] for d, _ in hits]
            lex = bm25_ranking(bm25, ids, q["question"])
            group = "answerable" if q["gold"] else "unanswerable"
            top1[name][group].append(hits[0][1])
            top1_by_q[(name, q["qid"])] = hits[0][1]
            for method, ranking in [("dense", dense), ("bm25", lex),
                                    ("hybrid", rrf(dense, lex)[:K_MAX])]:
                pages = [page_of[c] for c in ranking]
                rankings[(f"{name}-{method}", q["qid"])] = pages

    systems = sorted({s for s, _ in rankings})
    ks = range(1, K_MAX + 1)
    with open("eval/retrieval_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "Hit@1", "Hit@3", "Hit@5", "MRR"])
        for s in systems:
            curve = [np.mean([hit_at_k(rankings[(s, q["qid"])], q["gold"], k)
                              for q in answerable]) for k in ks]
            mrr = np.mean([reciprocal_rank(rankings[(s, q["qid"])], q["gold"])
                           for q in answerable])
            values = (curve[0], curve[2], curve[4], mrr)
            w.writerow([s] + [f"{v:.2f}" for v in values])
            plt.plot(ks, curve, marker="o", label=s)
            print(f"{s:10s} Hit@k: " + " ".join(f"{v:.2f}" for v in curve))
    plt.xlabel("k"); plt.ylabel("Hit@k"); plt.ylim(0, 1.05)
    plt.grid(alpha=0.3); plt.legend()
    plt.savefig("reports/figures/03_hit_at_k.png", dpi=150); plt.close()

    # розподіл схожості top-1 (dense) для обох конфігурацій; спільні bins
    fig, axes = plt.subplots(1, len(CHUNKS), figsize=(10, 4), sharey=True)
    bins = np.linspace(0.82, 0.90, 17)
    for ax, name in zip(axes, CHUNKS):
        ax.hist(top1[name]["answerable"], bins=bins, alpha=0.6, label="є відповідь")
        ax.hist(top1[name]["unanswerable"], bins=bins, alpha=0.6, label="немає відповіді")
        if name == CONFIG:
            ax.axvline(THRESHOLD, color="red", linestyle="--", label=f"τ = {THRESHOLD}")
        ax.set_title(f"Конфігурація {name}"); ax.set_xlabel("Косинусна схожість top-1 чанка")
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.legend()
    axes[0].set_ylabel("Кількість запитань")
    plt.tight_layout(); plt.savefig("reports/figures/04_top1_scores.png", dpi=150)

    print()
    print(open("eval/retrieval_metrics.csv", encoding="utf-8").read())
    print("Позиція першого релевантного чанка (- означає, що немає в top-10):")
    print("qid  " + " ".join(f"{s:>9s}" for s in systems))
    for q in answerable:
        ranks = [first_rank(rankings[(s, q["qid"])], q["gold"]) for s in systems]
        print(f"{q['qid']}  " + " ".join(f"{(r or '-'):>9}" for r in ranks))
    print("\nСхожість top-1 (dense) по запитаннях:")
    print("qid   " + "  ".join(f"{n:>6s}" for n in CHUNKS) + "  тип")
    for q in qs:
        print(f"{q['qid']}  " + "  ".join(f"{top1_by_q[(n, q['qid'])]:6.3f}" for n in CHUNKS)
              + f"  {q['type']}")
