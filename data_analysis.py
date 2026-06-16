import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocessing import TARGET


def analyze_dataset(df, fig_dir):
    print("=" * 60)
    print("1. Аналіз набору даних (EDA)")
    print("=" * 60)

    os.makedirs(fig_dir, exist_ok=True)

    print(f"Загальна кількість транзакцій: {len(df)}")
    print(f"Кількість ознак: {df.shape[1]}")
    print()

    print("Розподіл за типом транзакцій:")
    counts = df[TARGET].value_counts()
    print(counts)
    print()

    numeric = df.select_dtypes(include="number").drop(columns=["is_fraud", "transaction_id", "customer_id"], errors='ignore')
    
    print("Статистичний аналіз числових ознак:")
    stats = numeric.describe().T
    print(stats[["mean", "std", "min", "50%", "max"]])
    print()

    plot_target_distribution(counts, fig_dir)
    plot_histograms(numeric, fig_dir)
    plot_boxplots(numeric, fig_dir)
    plot_correlation(numeric, fig_dir)
    plot_scaler_comparison(numeric, fig_dir)
    
    return counts, stats


def plot_target_distribution(counts, fig_dir):
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="viridis", legend=False)
    ax.set_title("Розподіл транзакцій за типом")
    ax.set_xlabel("Тип")
    ax.set_ylabel("Кількість транзакцій")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha="center", fontsize=9)
    plt.xticks(rotation=25)
    plt.tight_layout()
    path = os.path.join(fig_dir, "target_distribution.png")
    plt.savefig(path, dpi=130)
    plt.close()


def plot_histograms(numeric, fig_dir):
    numeric.hist(figsize=(12, 10), bins=30, color="skyblue", edgecolor="black")
    plt.suptitle("Гістограми розподілу числових ознак", fontsize=14)
    plt.tight_layout()
    path = os.path.join(fig_dir, "histograms.png")
    plt.savefig(path, dpi=130)
    plt.close()


def plot_boxplots(numeric, fig_dir):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=numeric, orient="h", palette="Set2")
    plt.title("Boxplot числових змінних")
    plt.tight_layout()
    path = os.path.join(fig_dir, "boxplots.png")
    plt.savefig(path, dpi=130)
    plt.close()


def plot_correlation(numeric, fig_dir):
    plt.figure(figsize=(10, 8))
    corr = numeric.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
    plt.title("Кореляційна матриця")
    plt.tight_layout()
    path = os.path.join(fig_dir, "correlation_heatmap.png")
    plt.savefig(path, dpi=130)
    plt.close()


def plot_scaler_comparison(numeric, fig_dir):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric)
    scaled_df = pd.DataFrame(scaled_data, columns=numeric.columns)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.kdeplot(data=numeric, ax=axes[0], legend=False)
    axes[0].set_title("Розподіл до StandardScaler")
    
    sns.kdeplot(data=scaled_df, ax=axes[1], legend=False)
    axes[1].set_title("Розподіл після StandardScaler")
    
    plt.tight_layout()
    path = os.path.join(fig_dir, "scaler_comparison.png")
    plt.savefig(path, dpi=130)
    plt.close()
