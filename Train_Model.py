import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

CSV_PATH = r"C:\Users\Asus\Downloads\rfm_modeling_snapshot.csv"
MODEL_PATH = "churn_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"
FEATURES_PATH = "model_features.pkl"

print("=" * 60)
print("D2C CUSTOMER CHURN PREDICTION MODEL TRAINING")
print("=" * 60)

print("\n[1] Loading data...")
df = pd.read_csv(CSV_PATH)
print(f"    Dataset shape: {df.shape}")
print(f"    Churn rate (overall): {df['churn_next_60d'].mean():.4f}")

train_df = df[df['split'] == 'train'].copy()
val_df = df[df['split'] == 'validation'].copy()
test_df = df[df['split'] == 'test'].copy()

print(f"\n[2] Data splits:")
print(f"    Train:      {len(train_df)} samples ({train_df['churn_next_60d'].mean():.4f} churn rate)")
print(f"    Validation: {len(val_df)} samples ({val_df['churn_next_60d'].mean():.4f} churn rate)")
print(f"    Test:       {len(test_df)} samples ({test_df['churn_next_60d'].mean():.4f} churn rate)")

print("\n[3] Feature engineering...")

drop_cols = ['customer_id', 'snapshot_date', 'split', 'churn_next_60d']

cat_cols = ['city_tier', 'age_group', 'acquisition_channel', 'loyalty_tier',
            'preferred_category', 'marketing_consent']

num_cols = [col for col in train_df.columns if col not in drop_cols and col not in cat_cols]

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fit on all data, transform each split
    all_values = pd.concat([train_df[col], val_df[col], test_df[col]]).astype(str).unique()
    le.fit(all_values)
    train_df[col] = le.transform(train_df[col].astype(str))
    val_df[col] = le.transform(val_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le
    print(f"    Encoded '{col}' ({len(le.classes_)} classes)")

feature_cols = cat_cols + num_cols
print(f"    Total features: {len(feature_cols)}")

X_train = train_df[feature_cols].values
y_train = train_df['churn_next_60d'].values
X_val = val_df[feature_cols].values
y_val = val_df['churn_next_60d'].values
X_test = test_df[feature_cols].values
y_test = test_df['churn_next_60d'].values

print("\n[4] Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
model.fit(X_train, y_train)
print("    Model trained successfully!")

print("\n[5] Validation Set Evaluation:")
val_pred = model.predict(X_val)
val_proba = model.predict_proba(X_val)[:, 1]

print(f"    Accuracy:  {accuracy_score(y_val, val_pred):.4f}")
print(f"    Precision: {precision_score(y_val, val_pred):.4f}")
print(f"    Recall:    {recall_score(y_val, val_pred):.4f}")
print(f"    F1 Score:  {f1_score(y_val, val_pred):.4f}")
print(f"    AUC-ROC:   {roc_auc_score(y_val, val_proba):.4f}")

print("\n[6] Test Set Evaluation:")
test_pred = model.predict(X_test)
test_proba = model.predict_proba(X_test)[:, 1]

print(f"    Accuracy:  {accuracy_score(y_test, test_pred):.4f}")
print(f"    Precision: {precision_score(y_test, test_pred):.4f}")
print(f"    Recall:    {recall_score(y_test, test_pred):.4f}")
print(f"    F1 Score:  {f1_score(y_test, test_pred):.4f}")
print(f"    AUC-ROC:   {roc_auc_score(y_test, test_proba):.4f}")

print("\n" + "-" * 60)
print("Classification Report (Test Set):")
print(classification_report(y_test, test_pred, target_names=['No Churn', 'Churn']))

print("\n[7] Top 10 Feature Importances:")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
for i, idx in enumerate(indices):
    print(f"    {i+1}. {feature_cols[idx]:30s} {importances[idx]:.4f}")

print("\n[8] Saving artifacts...")
joblib.dump(model, MODEL_PATH)
joblib.dump(label_encoders, ENCODERS_PATH)
joblib.dump({'feature_cols': feature_cols, 'cat_cols': cat_cols}, FEATURES_PATH)
print(f"    Model saved to:      {MODEL_PATH}")
print(f"    Encoders saved to:   {ENCODERS_PATH}")
print(f"    Features saved to:   {FEATURES_PATH}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)