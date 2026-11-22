# src/experiment.py
# Е10: префікси моделі e5. Індекс і запити з префіксами "query: "/"passage: " і без них.
# Порівнюються Hit@k, MRR (dense-пошук), розподіли схожості top-1 і час індексації.
import csv, time
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from index import load_chunks, DEVICE
from eval_retrieval import load_questions, hit_at_k, reciprocal_rank, K_MAX
from config import EMB_MODEL, CHUNKS, CONFIG, THRESHOLD

# назва варіанта -> (префікс запиту, префікс чанка)
VARIANTS = {
    "query+passage": ("query: ", "passage: "),  # основний експеримент
    "без префіксів": ("", ""),
    "лише query": ("query: ", ""),
    "лише passage": ("", "passage: "),
}


def encode(model, texts, prefix):
    # prompt=... додає префікс так само, як HuggingFaceEmbeddings у src/index.py
    return model.encode(texts, prompt=prefix or None, normalize_embeddings=True,
                        batch_size=32, convert_to_numpy=True)


if __name__ == "__main__":
    model = SentenceTransformer(EMB_MODEL, device=DEVICE)
    qs = load_questions()
    answerable = [q for q in qs if q["gold"]]
    ks = range(1, K_MAX + 1)
    results, curves, top1 = [], {}, {}
    for name in CHUNKS:
        chunks = load_chunks(name)
        texts = [c["text"] for c in chunks]
        pages = np.array([c["page"] for c in chunks])
        for var, (qp, pp) in VARIANTS.items():
            t0 = time.perf_counter()
            D = encode(model, texts, pp)  # «індекс»: матриця векторів чанків
            t_index = time.perf_counter() - t0
            Q = encode(model, [q["question"] for q in qs], qp)
            S = Q @ D.T  # косинусна схожість (вектори нормовані)
            ranked = {q["qid"]: list(pages[np.argsort(-S[i])[:K_MAX]])
                      for i, q in enumerate(qs)}
            best = {q["qid"]: float(S[i].max()) for i, q in enumerate(qs)}
            curve = [np.mean([hit_at_k(ranked[q["qid"]], q["gold"], k)
                              for q in answerable]) for k in ks]
            mrr = np.mean([reciprocal_rank(ranked[q["qid"]], q["gold"])
                           for q in answerable])
            ans = [best[q["qid"]] for q in answerable]
            una = [best[q["qid"]] for q in qs if not q["gold"]]
            curves[(name, var)] = curve
            top1[(name, var)] = (ans, una)
            results.append({
                "config": name, "variant": var,
                "Hit@1": round(float(curve[0]), 2), "Hit@3": round(float(curve[2]), 2),
                "Hit@5": round(float(curve[4]), 2), "Hit@10": round(float(curve[9]), 2),
                "MRR": round(float(mrr), 2),
                # як спрацює поріг τ, обраний на кроці 6 (з префіксами)
                "false_refusals_at_tau": sum(s < THRESHOLD for s in ans),
                "blocked_unanswerable_at_tau": sum(s < THRESHOLD for s in una),
                "top1_answerable_mean": round(float(np.mean(ans)), 3),
                "top1_answerable_min": round(min(ans), 3),
                "top1_unanswerable_max": round(max(una), 3),
                # запас розділення: > 0 означає, що існує поріг без помилок
                "margin": round(min(ans) - max(una), 3),
                "index_time_s": round(t_index, 2)})

    with open("eval/experiment_e10.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader(); w.writerows(results)
    for r in results:
        print(r)

    # графік 1: криві Hit@k для обох конфігурацій
    fig, axes = plt.subplots(1, len(CHUNKS), figsize=(11, 4), sharey=True)
    for ax, name in zip(axes, CHUNKS):
        for var in VARIANTS:
            ax.plot(ks, curves[(name, var)], marker="o", label=var)
        ax.set_title(f"Конфігурація {name}, dense"); ax.set_xlabel("k")
        ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Hit@k"); axes[0].legend()
    plt.tight_layout(); plt.savefig("reports/figures/08_e10_hit_at_k.png", dpi=150)
    plt.close()

    # графік 2: розподіл top-1 для обраної конфігурації, з префіксами і без
    fig, ax = plt.subplots(figsize=(8, 4))
    data, labels = [], []
    for var in ["query+passage", "без префіксів"]:
        ans, una = top1[(CONFIG, var)]
        data += [ans, una]
        labels += [f"{var}\nє відповідь", f"{var}\nнемає відповіді"]
    ax.boxplot(data, tick_labels=labels)
    for i, d in enumerate(data, start=1):  # окремі точки поверх boxplot
        ax.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-0.08, 0.08, len(d)),
                   d, s=14, alpha=0.7, color="black", zorder=3)
    ax.axhline(THRESHOLD, color="red", linestyle="--", label=f"τ = {THRESHOLD} (з префіксами)")
    ax.set_ylabel("Косинусна схожість top-1"); ax.set_title(f"Конфігурація {CONFIG}")
    ax.legend(fontsize=8); ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout(); plt.savefig("reports/figures/09_e10_top1_scores.png", dpi=150)
