import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def results_table(results, fig_dir, out_dir):
    df = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)
    df["accuracy"] = df["accuracy"].round(4)
    df["train_time"] = df["train_time"].round(3)

    print("=" * 60)
    print("4.1. Зведена таблиця результатів")
    print("=" * 60)
    print(df.to_string(index=False))

    csv_path = os.path.join(out_dir, "results_summary.csv")
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(df) + 1))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=["Модель", "Точність", "Час навчання, с"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for j in range(3):
        table[0, j].set_facecolor("#2E75B6")
        table[0, j].set_text_props(color="white", weight="bold")
    plt.title("Точність та час навчання моделей", pad=10)
    plt.tight_layout()
    path = os.path.join(fig_dir, "results_table.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Збережено таблицю: {path}")
    return df


def accuracy_bar(results, fig_dir):
    df = pd.DataFrame(results).sort_values("accuracy", ascending=True)
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(x="accuracy", y="model", data=df, hue="model",
                     palette="mako", legend=False)
    ax.axvline(1 / 6, color="red", linestyle="--", linewidth=1)
    ax.text(1 / 6 + 0.002, 0.2, "базовий рівень (1/6)", color="red", fontsize=9)
    ax.set_title("Порівняння точності моделей")
    ax.set_xlabel("Точність")
    ax.set_ylabel("Модель")
    plt.tight_layout()
    path = os.path.join(fig_dir, "accuracy_comparison.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Збережено графік: {path}")


def tsne_plots(x_test, y_test, predictions, class_names, fig_dir, random_state=42):
    print("=" * 60)
    print("4.2. Візуалізація t-SNE")
    print("=" * 60)
    print("Обчислення проєкції t-SNE...")

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, init="pca")
    emb = tsne.fit_transform(x_test)

    palette = sns.color_palette("tab10", len(class_names))

    plt.figure(figsize=(9, 7))
    for idx, name in enumerate(class_names):
        mask = y_test == idx
        plt.scatter(emb[mask, 0], emb[mask, 1], s=8, color=palette[idx], label=name, alpha=0.6)
    plt.title("t-SNE: очікувані класи (справжні мітки)")
    plt.xlabel("Компонента 1")
    plt.ylabel("Компонента 2")
    plt.legend(markerscale=2, fontsize=8, loc="best")
    plt.tight_layout()
    true_path = os.path.join(fig_dir, "tsne_true.png")
    plt.savefig(true_path, dpi=130)
    plt.close()
    print(f"Збережено графік: {true_path}")

    n = len(predictions)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows))
    axes = np.array(axes).reshape(-1)

    for ax_idx, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[ax_idx]
        for idx in range(len(class_names)):
            mask = y_pred == idx
            ax.scatter(emb[mask, 0], emb[mask, 1], s=6, color=palette[idx], alpha=0.6)
        ax.set_title(name, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    for k in range(n, len(axes)):
        axes[k].axis("off")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[i], markersize=8)
        for i in range(len(class_names))
    ]
    fig.legend(handles, class_names, loc="lower center", ncol=6, fontsize=9)
    fig.suptitle("t-SNE: отримані класи за кожною моделлю", fontsize=14)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    pred_path = os.path.join(fig_dir, "tsne_predicted.png")
    plt.savefig(pred_path, dpi=120)
    plt.close()
    print(f"Збережено графік: {pred_path}")
    return emb
