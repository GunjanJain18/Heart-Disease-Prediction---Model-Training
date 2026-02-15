import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Heart Disease Prediction", page_icon="<3", layout="wide")

MODEL_DIR = "model"

@st.cache_resource
def load_models():
    files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'KNN': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest (Ensemble)': 'random_forest.pkl',
        'XGBoost (Ensemble)': 'xgboost_model.pkl'
    }
    loaded = {}
    for name, fname in files.items():
        p = os.path.join(MODEL_DIR, fname)
        if os.path.exists(p):
            loaded[name] = joblib.load(p)
    return loaded

@st.cache_resource
def load_scaler():
    p = os.path.join(MODEL_DIR, 'scaler.pkl')
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_data
def load_metrics():
    p = os.path.join(MODEL_DIR, 'metrics.json')
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None

@st.cache_data
def load_sample():
    p = os.path.join(MODEL_DIR, 'test_sample.csv')
    return pd.read_csv(p) if os.path.exists(p) else None

models = load_models()
scaler = load_scaler()
metrics_data = load_metrics()
sample_data = load_sample()

# sidebar
st.sidebar.title("Heart Disease ML")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Home", "Model Comparison", "Predictions"])
st.sidebar.markdown("---")
st.sidebar.info("ML Assignment 2\n\n**Dataset:** Heart Disease (Kaggle)\n\n**Models:** 6 classifiers\n\n**Name:** Gunjan Jain\n\n**Roll No:** 2025ab05216\n\n**Email:** 2025ab05216@wilp.bits-pilani.ac.in\n\n**BITS Pilani**")

# ---- HOME PAGE ----
if page == "Home":
    st.title("Heart Disease Prediction using ML Models")
    st.caption("ML Classification - Assignment 2")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Features", "13")
    c2.metric("Instances", "1,025")
    c3.metric("Models", "6")
    c4.metric("Type", "Binary")

    st.markdown("---")

    st.header("Problem Statement")
    st.write(
        "Heart disease is one of the leading causes of death globally. Early prediction "
        "can help in timely medical intervention. This project uses six ML classification "
        "models to predict heart disease from 13 clinical features."
    )

    st.header("Dataset")
    st.write(
        "- **Source:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)\n"
        "- **Features:** 13 clinical attributes (age, sex, chest pain type, blood pressure, cholesterol, etc.)\n"
        "- **Target:** 0 = No Heart Disease, 1 = Heart Disease\n"
        "- **Size:** 1,025 records"
    )

    with st.expander("Feature Details"):
        st.dataframe(pd.DataFrame({
            'Feature': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
            'Description': [
                'Age (years)', 'Sex (0=F, 1=M)', 'Chest pain type (0-3)',
                'Resting BP (mm Hg)', 'Cholesterol (mg/dl)', 'Fasting blood sugar > 120 (0/1)',
                'Resting ECG (0-2)', 'Max heart rate', 'Exercise angina (0/1)',
                'ST depression', 'ST slope (0-2)', 'Major vessels (0-4)', 'Thalassemia (0-3)'
            ],
            'Type': ['Num', 'Cat', 'Cat', 'Num', 'Num', 'Bin', 'Cat', 'Num', 'Bin', 'Num', 'Cat', 'Num', 'Cat']
        }), use_container_width=True, hide_index=True)

    st.header("Models Used")
    c1, c2 = st.columns(2)
    c1.markdown("1. **Logistic Regression**\n2. **Decision Tree**\n3. **KNN**")
    c2.markdown("4. **Gaussian Naive Bayes**\n5. **Random Forest** (Ensemble)\n6. **XGBoost** (Ensemble)")

# ---- MODEL COMPARISON PAGE ----
elif page == "Model Comparison":
    st.title("Model Performance Comparison")
    st.markdown("---")

    if not metrics_data:
        st.warning("No metrics found. Run model/train_models.py first.")
    else:
        res = metrics_data['results']
        df = pd.DataFrame(res).T.reset_index()
        df.columns = ['Model'] + list(df.columns[1:])

        st.subheader("Metrics Table")
        st.dataframe(
            df.style.highlight_max(subset=['Accuracy','AUC','Precision','Recall','F1','MCC'], color='#90EE90')
                     .highlight_min(subset=['Accuracy','AUC','Precision','Recall','F1','MCC'], color='#FFB6C1')
                     .format({'Accuracy':'{:.4f}','AUC':'{:.4f}','Precision':'{:.4f}',
                              'Recall':'{:.4f}','F1':'{:.4f}','MCC':'{:.4f}'}),
            use_container_width=True, hide_index=True
        )

        best_name = df.loc[df['Accuracy'].idxmax(), 'Model']
        st.success(f"**Best (Accuracy):** {best_name} - {df['Accuracy'].max():.4f}")
        st.markdown("---")

        # chart
        st.subheader("Visual Comparison")
        pick = st.selectbox("Metric:", ['All Metrics','Accuracy','AUC','Precision','Recall','F1','MCC'])

        fig, ax = plt.subplots(figsize=(12, 6))
        mets = ['Accuracy','AUC','Precision','Recall','F1','MCC']
        cols = ['#3498db','#2ecc71','#e74c3c','#f39c12','#9b59b6','#1abc9c']

        if pick == 'All Metrics':
            x = np.arange(len(df))
            w = 0.12
            for i, (m, c) in enumerate(zip(mets, cols)):
                ax.bar(x + i*w, df[m], w, label=m, color=c, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x + w*2.5)
            ax.set_xticklabels(df['Model'], rotation=25, ha='right')
            ax.legend(loc='lower right')
        else:
            bars = ax.bar(df['Model'], df[pick], color=cols, edgecolor='black', linewidth=0.5)
            ax.set_xticklabels(df['Model'], rotation=25, ha='right')
            for b, v in zip(bars, df[pick]):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.4f}',
                        ha='center', fontweight='bold', fontsize=10)

        ax.set_ylim(0, 1.15)
        ax.set_title(f'Model Comparison - {pick}', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        # radar for top 3
        st.subheader("Radar Chart - Top 3")
        top3 = df.nlargest(3, 'Accuracy')
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2*np.pi, len(mets), endpoint=False).tolist()
        angles += angles[:1]
        radar_cols = ['#3498db','#e74c3c','#2ecc71']
        for i, (_, row) in enumerate(top3.iterrows()):
            vals = [row[m] for m in mets] + [row[mets[0]]]
            ax.plot(angles, vals, 'o-', linewidth=2, label=row['Model'], color=radar_cols[i])
            ax.fill(angles, vals, alpha=0.1, color=radar_cols[i])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(mets)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        st.pyplot(fig)

# ---- PREDICTIONS PAGE ----
elif page == "Predictions":
    st.title("Make Predictions")
    st.markdown("---")

    if not models:
        st.error("No models loaded. Run model/train_models.py first.")
        st.stop()

    st.subheader("1. Pick a model")
    chosen = st.selectbox("Model:", list(models.keys()), index=len(models)-1)

    st.markdown("---")
    st.subheader("2. Upload test data (CSV)")
    st.info("Upload CSV with the same features. If 'target' column exists, metrics will be shown.")

    c1, c2 = st.columns([3, 1])
    uploaded = c1.file_uploader("CSV file", type="csv")
    use_sample = c2.button("Use Sample Data")

    test_df = None
    if uploaded:
        test_df = pd.read_csv(uploaded)
        st.success(f"Loaded {uploaded.name} ({test_df.shape[0]} rows)")
    elif use_sample and sample_data is not None:
        test_df = sample_data.copy()
        st.success(f"Sample data loaded ({test_df.shape[0]} rows)")

    if test_df is not None:
        with st.expander("Preview data"):
            st.dataframe(test_df.head(20), use_container_width=True)

        st.markdown("---")
        has_target = 'target' in test_df.columns
        feat_names = metrics_data['feature_names'] if metrics_data else [c for c in test_df.columns if c != 'target']

        missing = [c for c in feat_names if c not in test_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        X_test = test_df[feat_names].copy()
        if X_test.isnull().sum().sum() > 0:
            X_test.fillna(X_test.median(), inplace=True)

        X_scaled = scaler.transform(X_test) if scaler else X_test.values

        mdl = models[chosen]
        y_pred = mdl.predict(X_scaled)
        y_prob = mdl.predict_proba(X_scaled)

        st.subheader("3. Results")
        result = test_df.copy()
        result['Prediction'] = y_pred
        result['Label'] = pd.Series(y_pred).map({0: 'No Disease', 1: 'Disease'}).values
        result['Confidence'] = np.max(y_prob, axis=1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Samples", len(y_pred))
        disease_n = int((y_pred == 1).sum())
        c2.metric("Predicted Disease", f"{disease_n} ({disease_n/len(y_pred)*100:.1f}%)")
        c3.metric("Avg Confidence", f"{np.mean(np.max(y_prob, axis=1))*100:.1f}%")

        with st.expander("All predictions", expanded=True):
            st.dataframe(result[['Label','Confidence'] + feat_names].style.format({'Confidence':'{:.4f}'}),
                         use_container_width=True)

        if has_target:
            y_true = test_df['target']
            st.markdown("---")
            st.subheader("4. Evaluation")

            acc = accuracy_score(y_true, y_pred)
            try: auc = roc_auc_score(y_true, y_prob[:, 1])
            except: auc = 0.0
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("AUC", f"{auc:.4f}")
            c3.metric("Precision", f"{prec:.4f}")
            c4.metric("Recall", f"{rec:.4f}")
            c5.metric("F1", f"{f1:.4f}")
            c6.metric("MCC", f"{mcc:.4f}")

            st.markdown("---")
            left, right = st.columns(2)

            with left:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['No Disease','Disease'], yticklabels=['No Disease','Disease'],
                            linewidths=1, linecolor='black', annot_kws={'size': 16, 'fontweight': 'bold'})
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title(f'{chosen}', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)

            with right:
                st.subheader("Classification Report")
                rpt = classification_report(y_true, y_pred, target_names=['No Disease','Disease'], output_dict=True)
                st.dataframe(pd.DataFrame(rpt).T.style.format('{:.4f}'), use_container_width=True)
    else:
        st.info("Upload a CSV or click 'Use Sample Data' to start.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'>Heart Disease Prediction | ML Assignment 2 | BITS Pilani</div>",
            unsafe_allow_html=True)
