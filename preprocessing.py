import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


DROP_COLS = [
    "transaction_id",
    "customer_id",
    "transaction_date",
    "transaction_time",
    "is_fraud",
]

CATEGORICAL_COLS = [
    "country",
    "city",
    "merchant_category",
    "payment_method",
    "device_type",
]

TARGET = "fraud_type"


def load_data(path):
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    return df


def prepare_balanced_dataset(df, random_state=42):
    df_copy = df.copy()
    mask_legit = df_copy["is_fraud"] == 0
    df_copy.loc[mask_legit, TARGET] = "Legitimate"
    
    fraud_df = df_copy[df_copy["is_fraud"] == 1]
    fraud_counts = fraud_df[TARGET].value_counts()
    min_count = fraud_counts.min()
    
    parts = []
    for label, group in df_copy.groupby(TARGET):
        n = min(min_count, len(group))
        parts.append(group.sample(n=n, random_state=random_state))
        
    sample = pd.concat(parts).sample(frac=1, random_state=random_state)
    return sample.reset_index(drop=True)


def build_features(df):
    y = df[TARGET].values
    x = df.drop(columns=DROP_COLS + [TARGET])

    numeric_cols = [c for c in x.columns if c not in CATEGORICAL_COLS]

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    x_prepared = transformer.fit_transform(x)
    if hasattr(x_prepared, "toarray"):
        x_prepared = x_prepared.toarray()

    feature_names = numeric_cols + list(
        transformer.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS)
    )
    return x_prepared, y, feature_names, numeric_cols
