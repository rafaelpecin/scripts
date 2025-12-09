import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def main():
    # Load cleaned dataset
    df = pd.read_csv("ecommerce_data_processed.csv")

    # Define target and features
    target = "conversion_status_numeric"
    categorical_cols = ["device_type", "country", "coupon_used", "product_category_viewed"]
    
    # Select feature columns (exclude session_id)
    feature_cols = [col for col in df.columns if col not in ["session_id", target, "conversion_status"]]

    X = df[feature_cols]
    y = df[target]

    # Preprocessing: One-hot encode categorical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"  # leave numeric columns as-is
    )

    # Build pipeline with preprocessing + RandomForest
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit model
    model.fit(X_train, y_train)

    # Extract feature names after one-hot encoding
    encoded_feature_names = (
        list(model.named_steps["preprocessor"]
             .named_transformers_["cat"]
             .get_feature_names_out(categorical_cols))
    )

    # Include numeric columns
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    all_feature_names = encoded_feature_names + numeric_cols

    # Extract feature importances
    importances = model.named_steps["rf"].feature_importances_

    # Combine into a sorted list of (feature, importance)
    importance_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    # Print top 3 features
    print("=== Top 3 Most Important Features ===")
    print(importance_df.head(3))

if __name__ == "__main__":
    main()

