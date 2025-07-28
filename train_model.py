import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

SEED = 42
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"

def main(csv_path: str = "diabetes.csv", model_path: str = "diabetes_model.pkl"):
    df = pd.read_csv(csv_path)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=SEED))
        ]
    )

    cv = cross_validate(
        pipe, X_train, y_train,
        cv=5,
        scoring=["accuracy", "roc_auc"],
        n_jobs=-1,
        return_train_score=False
    )
    print("CV accuracy  : ", cv["test_accuracy"].mean(), "±", cv["test_accuracy"].std())
    print("CV ROC-AUC   : ", cv["test_roc_auc"].mean(), "±", cv["test_roc_auc"].std())

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Hold-out Results ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(pipe, model_path)
    print(f"\nSaved model to: {model_path}")

if __name__ == "__main__":
    main()
