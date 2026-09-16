import os
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'srivilliputhur.db')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM ml_dataset", conn)
    conn.close()
    return df

def train_and_evaluate():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} synthetic rows.")
    print("Target distribution:\n", df['overflow_risk_label'].value_counts())

    # Target variable: map LOW -> 0, MEDIUM -> 1
    # Note: HIGH is 0 count in synthetic dataset
    target_mapping = {'LOW': 0, 'MEDIUM': 1}
    inv_target_mapping = {0: 'LOW', 1: 'MEDIUM'}
    y = df['overflow_risk_label'].map(target_mapping)

    # Define feature sets
    # 1. Causal Pre-blockage features (Recommended: Leakage-Free)
    causal_num_cols = [
        'rainfall_mm_7day',
        'rainfall_mm_30day',
        'plastic_accumulation_score',
        'drain_width_m',
        'drain_depth_m',
        'days_since_cleaning',
        'previous_overflow_count_1yr',
        'water_level_cm'
    ]
    cat_cols = ['drain_type']

    # 2. Features including Blockage_Percent (Diagnostic for target leakage)
    leakage_num_cols = causal_num_cols + ['blockage_percent']

    # Leakage Analysis
    df['target_binary'] = y
    corr_blockage = float(df['blockage_percent'].corr(df['target_binary']))
    corr_plastic = float(df['plastic_accumulation_score'].corr(df['target_binary']))
    corr_rain7 = float(df['rainfall_mm_7day'].corr(df['target_binary']))
    corr_rain30 = float(df['rainfall_mm_30day'].corr(df['target_binary']))

    blockage_by_class = {
        'LOW': {
            'min': float(df[df['overflow_risk_label'] == 'LOW']['blockage_percent'].min()),
            'max': float(df[df['overflow_risk_label'] == 'LOW']['blockage_percent'].max()),
            'mean': float(df[df['overflow_risk_label'] == 'LOW']['blockage_percent'].mean()),
            'std': float(df[df['overflow_risk_label'] == 'LOW']['blockage_percent'].std())
        },
        'MEDIUM': {
            'min': float(df[df['overflow_risk_label'] == 'MEDIUM']['blockage_percent'].min()),
            'max': float(df[df['overflow_risk_label'] == 'MEDIUM']['blockage_percent'].max()),
            'mean': float(df[df['overflow_risk_label'] == 'MEDIUM']['blockage_percent'].mean()),
            'std': float(df[df['overflow_risk_label'] == 'MEDIUM']['blockage_percent'].std())
        }
    }

    print(f"Correlation of Blockage_Percent with Target: {corr_blockage:.4f}")

    # Build Preprocessors
    preprocessor_causal = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), causal_num_cols),
            ('cat', OneHotEncoder(categories=[['OPEN', 'COVERED', 'PARTIALLY_COVERED']], handle_unknown='ignore'), cat_cols)
        ]
    )

    preprocessor_leakage = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), leakage_num_cols),
            ('cat', OneHotEncoder(categories=[['OPEN', 'COVERED', 'PARTIALLY_COVERED']], handle_unknown='ignore'), cat_cols)
        ]
    )

    # 80/20 Stratified Split
    X_train_c, X_test_c, y_train, y_test = train_test_split(
        df[causal_num_cols + cat_cols], y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_l, X_test_l, _, _ = train_test_split(
        df[leakage_num_cols + cat_cols], y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ==========================================
    # 1. Pipeline A: Causal (Leakage-Free) Models
    # ==========================================
    pipe_lr_causal = Pipeline([
        ('pre', preprocessor_causal),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])

    pipe_rf_causal = Pipeline([
        ('pre', preprocessor_causal),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42))
    ])

    # Cross-validation
    cv_lr_causal = cross_validate(pipe_lr_causal, df[causal_num_cols + cat_cols], y, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1'])
    cv_rf_causal = cross_validate(pipe_rf_causal, df[causal_num_cols + cat_cols], y, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1'])

    pipe_lr_causal.fit(X_train_c, y_train)
    pipe_rf_causal.fit(X_train_c, y_train)

    y_pred_lr_c = pipe_lr_causal.predict(X_test_c)
    y_pred_rf_c = pipe_rf_causal.predict(X_test_c)

    # Feature Importance for Random Forest Causal
    rf_feat_names = causal_num_cols + ['drain_type_OPEN', 'drain_type_COVERED', 'drain_type_PARTIALLY_COVERED']
    rf_causal_importances = pipe_rf_causal.named_steps['clf'].feature_importances_.tolist()
    feat_importance_causal = sorted(
        [{'feature': f, 'importance': round(imp, 4)} for f, imp in zip(rf_feat_names, rf_causal_importances)],
        key=lambda x: x['importance'], reverse=True
    )

    # ==========================================
    # 2. Pipeline B: Diagnostic with Leakage (Blockage_Percent)
    # ==========================================
    pipe_lr_leak = Pipeline([
        ('pre', preprocessor_leakage),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])

    pipe_rf_leak = Pipeline([
        ('pre', preprocessor_leakage),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42))
    ])

    cv_lr_leak = cross_validate(pipe_lr_leak, df[leakage_num_cols + cat_cols], y, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1'])
    cv_rf_leak = cross_validate(pipe_rf_leak, df[leakage_num_cols + cat_cols], y, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1'])

    pipe_lr_leak.fit(X_train_l, y_train)
    pipe_rf_leak.fit(X_train_l, y_train)

    y_pred_lr_l = pipe_lr_leak.predict(X_test_l)
    y_pred_rf_l = pipe_rf_leak.predict(X_test_l)

    rf_leak_feat_names = leakage_num_cols + ['drain_type_OPEN', 'drain_type_COVERED', 'drain_type_PARTIALLY_COVERED']
    rf_leak_importances = pipe_rf_leak.named_steps['clf'].feature_importances_.tolist()
    feat_importance_leak = sorted(
        [{'feature': f, 'importance': round(imp, 4)} for f, imp in zip(rf_leak_feat_names, rf_leak_importances)],
        key=lambda x: x['importance'], reverse=True
    )

    # Assemble metrics object
    metrics = {
        'dataset_summary': {
            'total_records': len(df),
            'class_distribution': {
                'LOW': int((df['overflow_risk_label'] == 'LOW').sum()),
                'MEDIUM': int((df['overflow_risk_label'] == 'MEDIUM').sum()),
                'HIGH': int((df['overflow_risk_label'] == 'HIGH').sum())
            },
            'disclaimer': 'Synthetic Prototype Dataset (50 rows). Contains LOW and MEDIUM classes only; zero HIGH classes.'
        },
        'leakage_analysis': {
            'blockage_correlation': round(corr_blockage, 4),
            'plastic_correlation': round(corr_plastic, 4),
            'rainfall_7day_correlation': round(corr_rain7, 4),
            'rainfall_30day_correlation': round(corr_rain30, 4),
            'blockage_by_class': blockage_by_class,
            'leakage_explanation': (
                'In the synthetic dataset, Blockage_Percent has an overwhelming 0.8508 Pearson correlation with Overflow_Risk_Label. '
                'LOW blockage ranges from 5.2% to 49.5% (mean: 24.89%), while MEDIUM ranges from 46.4% to 81.6% (mean: 65.20%). '
                'Including Blockage_Percent causes target leakage because the synthetic label was generated directly using blockage thresholds. '
                'The Causal Pipeline appropriately excludes Blockage_Percent to model upstream hydrological and waste drivers.'
            )
        },
        'models': {
            'causal_pipeline': {
                'description': 'Leakage-Free Causal Pre-Blockage Pipeline (Excludes Blockage_Percent)',
                'features_used': causal_num_cols + cat_cols,
                'excluded_features': ['blockage_percent (target leakage)', 'ward_no (spatial proxy)', 'latitude', 'longitude'],
                'logistic_regression': {
                    'cv_5fold_accuracy': round(float(np.mean(cv_lr_causal['test_accuracy'])), 4),
                    'cv_5fold_f1': round(float(np.mean(cv_lr_causal['test_f1'])), 4),
                    'test_accuracy': round(float(accuracy_score(y_test, y_pred_lr_c)), 4),
                    'test_precision': round(float(precision_score(y_test, y_pred_lr_c, zero_division=0)), 4),
                    'test_recall': round(float(recall_score(y_test, y_pred_lr_c, zero_division=0)), 4),
                    'test_f1': round(float(f1_score(y_test, y_pred_lr_c, zero_division=0)), 4),
                    'confusion_matrix': confusion_matrix(y_test, y_pred_lr_c).tolist()
                },
                'random_forest': {
                    'cv_5fold_accuracy': round(float(np.mean(cv_rf_causal['test_accuracy'])), 4),
                    'cv_5fold_f1': round(float(np.mean(cv_rf_causal['test_f1'])), 4),
                    'test_accuracy': round(float(accuracy_score(y_test, y_pred_rf_c)), 4),
                    'test_precision': round(float(precision_score(y_test, y_pred_rf_c, zero_division=0)), 4),
                    'test_recall': round(float(recall_score(y_test, y_pred_rf_c, zero_division=0)), 4),
                    'test_f1': round(float(f1_score(y_test, y_pred_rf_c, zero_division=0)), 4),
                    'confusion_matrix': confusion_matrix(y_test, y_pred_rf_c).tolist(),
                    'feature_importance': feat_importance_causal
                }
            },
            'diagnostic_leakage_pipeline': {
                'description': 'Diagnostic Baseline Pipeline (Includes Blockage_Percent)',
                'features_used': leakage_num_cols + cat_cols,
                'logistic_regression': {
                    'cv_5fold_accuracy': round(float(np.mean(cv_lr_leak['test_accuracy'])), 4),
                    'cv_5fold_f1': round(float(np.mean(cv_lr_leak['test_f1'])), 4),
                    'test_accuracy': round(float(accuracy_score(y_test, y_pred_lr_l)), 4),
                    'confusion_matrix': confusion_matrix(y_test, y_pred_lr_l).tolist()
                },
                'random_forest': {
                    'cv_5fold_accuracy': round(float(np.mean(cv_rf_leak['test_accuracy'])), 4),
                    'cv_5fold_f1': round(float(np.mean(cv_rf_leak['test_f1'])), 4),
                    'test_accuracy': round(float(accuracy_score(y_test, y_pred_rf_l)), 4),
                    'confusion_matrix': confusion_matrix(y_test, y_pred_rf_l).tolist(),
                    'feature_importance': feat_importance_leak
                }
            }
        }
    }

    # Save models
    joblib.dump(pipe_lr_causal, os.path.join(MODELS_DIR, 'logistic_causal.joblib'))
    joblib.dump(pipe_rf_causal, os.path.join(MODELS_DIR, 'rf_causal.joblib'))
    joblib.dump(pipe_lr_leak, os.path.join(MODELS_DIR, 'logistic_leakage.joblib'))
    joblib.dump(pipe_rf_leak, os.path.join(MODELS_DIR, 'rf_leakage.joblib'))

    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("ML Pipeline training and evaluation complete.")
    print("Metrics saved to models/metrics.json.")

if __name__ == '__main__':
    train_and_evaluate()
