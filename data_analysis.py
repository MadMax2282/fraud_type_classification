import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import TARGET


def analyze_dataset(df, fraud, fig_dir):
    print("=" * 60)
    print("1. Аналіз набору даних")
    print("=" * 60)

    print(f"Загальна кількість транзакцій: {len(df)}")
    print(f"Кількість ознак: {df.shape[1]}")
    print(f"Кількість шахрайських транзакцій: {len(fraud)}")
    print(f"Частка шахрайства у наборі: {len(fraud) / len(df) * 100:.2f}%")
    print()

    print("Розподіл за типом шахрайства:")
    counts = fraud[TARGET].value_counts()
    print(counts)
    print()

    print("Статистичний аналіз числових ознак:")
    numeric = fraud.select_dtypes(include="number").drop(columns=["is_fraud"])
    stats = numeric.describe().T
    print(stats[["mean", "std", "min", "50%", "max"]])
    print()

    plot_target_distribution(counts, fig_dir)
    return counts, stats


def plot_target_distribution(counts, fig_dir):
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                     palette="viridis", legend=False)
    ax.set_title("Розподіл транзакцій за типом шахрайства")
    ax.set_xlabel("Тип шахрайства")
    ax.set_ylabel("Кількість транзакцій")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha="center", fontsize=9)
    plt.xticks(rotation=20)
    plt.tight_layout()
    path = os.path.join(fig_dir, "target_distribution.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Збережено графік: {path}")
