import os
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from preprocessing import load_data, get_fraud_subset, balanced_sample, build_features, TARGET
from data_analysis import analyze_dataset
from models import train_and_evaluate
from visualization import results_table, accuracy_bar, tsne_plots


DATA_PATH = os.path.join("data", "bank_fraud.csv")
RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
SAMPLE_PER_CLASS = 2000
TEST_SIZE = 0.3
RANDOM_STATE = 42


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    fraud = get_fraud_subset(df)

    analyze_dataset(df, fraud, FIG_DIR)

    sample = balanced_sample(fraud, SAMPLE_PER_CLASS, RANDOM_STATE)
    x, y_text, feature_names, numeric_cols = build_features(sample)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    class_names = list(encoder.classes_)

    print()
    print("2. Підготовка даних")
    print(f"Розмір вибірки для моделювання: {x.shape[0]} об'єктів, {x.shape[1]} ознак")
    print(f"Видалено неінформативні стовпці: ідентифікатори, дата, час, is_fraud")
    print(f"Кількість класів: {len(class_names)}")
    print()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Навчальна вибірка: {x_train.shape[0]}, тестова вибірка: {x_test.shape[0]}")
    print()

    results, predictions = train_and_evaluate(
        x_train, x_test, y_train, y_test, class_names, FIG_DIR
    )

    summary = results_table(results, FIG_DIR, RESULTS_DIR)
    accuracy_bar(results, FIG_DIR)
    tsne_plots(x_test, y_test, predictions, class_names, FIG_DIR, RANDOM_STATE)

    print()
    print("Роботу завершено. Усі результати збережено у results.")


if __name__ == "__main__":
    main()
