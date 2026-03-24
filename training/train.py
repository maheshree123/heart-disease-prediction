import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("../data/heart_disease_dataset.csv")

# -------------------------------
# FEATURES & TARGET
# -------------------------------
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# -------------------------------
# TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# DEFINE MODELS
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42)
}

best_model = None
best_recall = 0

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

# -------------------------------
# TRAIN & COMPARE WITH FULL METRICS
# -------------------------------
print("\n----- MODEL PERFORMANCE -----")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    prec = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"\n{name}")
    print("Accuracy :", acc)
    print("Recall   :", rec)
    print("Precision:", prec)
    print("F1 Score :", f1)

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Select best model based on recall
    if rec > best_recall:
        best_recall = rec
        best_model = model

# -------------------------------
# SAVE BEST MODEL
# -------------------------------
os.makedirs("../model", exist_ok=True)

joblib.dump(best_model, "../model/model.pkl")

print("\n✅ Best model saved at: ../model/model.pkl")