# src/eval_answers.py
import pandas as pd
import matplotlib.pyplot as plt

LABELS = ["correct", "partial", "wrong", "hallucination",
          "refusal_ok", "refusal_wrong"]

df = pd.read_csv("eval/answers_labeled.csv")
q = pd.read_csv("eval/questions.csv", dtype=str).fillna("")
df = df.merge(q[["qid", "pages"]].rename(columns={"pages": "gold"}), on="qid")
assert df["label"].isin(LABELS).all(), "Розмітьте вручну всі відповіді"

rows = []
for mode, g in df.groupby("mode"):
    ans, unans = g[g["gold"] != ""], g[g["gold"] == ""]
    correct = ((ans["label"] == "correct")
               + 0.5 * (ans["label"] == "partial")).mean()
    halluc = (g["label"] == "hallucination").mean()
    refusal = (unans["label"] == "refusal_ok").mean()
    false_ref = (ans["label"] == "refusal_wrong").sum()
    print(f"{mode}: правильність={correct:.2f}, галюцинації={halluc:.2f}, "
          f"коректні відмови={refusal:.2f}, хибні відмови={false_ref}, "
          f"сер. час генерації={g['t_gen'].mean():.2f} с")
    rows.append({"mode": mode, "correctness": round(correct, 2),
                 "hallucination_rate": round(halluc, 2),
                 "correct_refusals": round(refusal, 2),
                 "false_refusals": int(false_ref),
                 "mean_t_gen": round(g["t_gen"].mean(), 2)})
pd.DataFrame(rows).to_csv("eval/answer_metrics.csv", index=False)

rag = df[(df["mode"] == "rag") & df["citation_ok"].notna()]
print(f"Точність посилань на сторінки (rag): {rag['citation_ok'].mean():.2f} "
      f"({int(rag['citation_ok'].sum())} з {len(rag)})")

tab = pd.crosstab(df["mode"], df["label"], normalize="index")
tab = tab.reindex(columns=LABELS, fill_value=0)
tab.plot.barh(stacked=True, figsize=(8, 3))
plt.xlabel("Частка відповідей"); plt.legend(ncol=3, fontsize=8, loc="upper center",
                                            bbox_to_anchor=(0.5, -0.25))
plt.tight_layout(); plt.savefig("reports/figures/05_answer_labels.png", dpi=150)
plt.close()

r = df[(df["mode"] == "rag") & (df["t_gen"] > 0)]  # без відмов за порогом
share = r["t_gen"].sum() / (r["t_gen"].sum() + r["t_retr"].sum())
print(f"Пошук: медіана {r['t_retr'].median():.3f} с; генерація: медіана "
      f"{r['t_gen'].median():.2f} с; частка генерації в часі відповіді {share:.1%}")
plt.boxplot([r["t_retr"], r["t_gen"]], tick_labels=["пошук", "генерація"])
plt.ylabel("Час, с"); plt.yscale("log")
plt.savefig("reports/figures/06_latency.png", dpi=150)
