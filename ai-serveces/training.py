import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from xgboost import XGBClassifier
from preprocess import preprocess_data
# Load Dataset
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "train_featured.csv"
)

df = pd.read_csv(DATA_PATH)
# Preprocessing
X, y = preprocess_data(df)
print("Features:", X.shape)
print("Target:", y.shape)
# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)
print(
    "Training:",
    X_train.shape
)
print(
    "Testing:",
    X_test.shape
)
# XGBoost Model
model = XGBClassifier(

    n_estimators=500,

    learning_rate=0.05,

    max_depth=8,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=3,

    gamma=0.1,

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1

)
# Train
model.fit(
    X_train,
    y_train
)
# Evaluation
predictions = model.predict(
    X_test
)
probabilities = model.predict_proba(
    X_test
)[:,1]
accuracy = accuracy_score(
    y_test,
    predictions
)
auc = roc_auc_score(
    y_test,
    probabilities
)
print(
    f"\nAccuracy: {accuracy:.4f}"
)
print(
    f"ROC-AUC: {auc:.4f}\n"
)
print(
    classification_report(
        y_test,
        predictions
    )
)
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# Feature Importance
importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
        model.feature_importances_

})


importance = importance.sort_values(

    by="Importance",

    ascending=False

)
print(
    "\nTop Features\n"
)
print(
    importance.head(20)
)

# Save Model
MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "credit_model.pkl"
)

joblib.dump(
    model,
    MODEL_PATH
)

print(
    "\nModel saved:"
)

print(
    MODEL_PATH
)