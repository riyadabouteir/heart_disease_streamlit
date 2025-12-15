import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Modèle de Prédiction des Maladies Cardiaques",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SESSION STATE (CORRIGÉ)
# =========================================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.model_trained = False
    st.session_state.df = None
    st.session_state.target_col = 'DEATH_EVENT'
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.metrics = None
    st.session_state.model_feature_names = None
    st.session_state.model_name = None

# =========================================================
# PREPROCESSING
# =========================================================
def preprocess_data(df_raw):
    df = df_raw.copy()

    for col in ['creatinine_phosphokinase', 'serum_creatinine', 'platelets']:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        df[col] = np.where(df[col] > upper, upper, df[col])

    df['Kidney_Heart_Risk'] = df['age'] * df['serum_creatinine']

    num_cols = [
        'age', 'creatinine_phosphokinase', 'ejection_fraction',
        'platelets', 'serum_creatinine', 'serum_sodium',
        'time', 'Kidney_Heart_Risk'
    ]

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, scaler

# =========================================================
# TRAIN MODEL
# =========================================================
def train_model(X, y, model_name, params):
    if model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )
    elif model_name == "Logistic Regression":
        model = LogisticRegression(C=params["C"], max_iter=1000)
    else:
        model = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"]
        )

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(yte, yp),
        "Precision": precision_score(yte, yp),
        "Recall": recall_score(yte, yp),
        "F1": f1_score(yte, yp),
        "ROC-AUC": roc_auc_score(yte, proba),
        "CM": confusion_matrix(yte, yp)
    }

    return model, metrics, X.columns.tolist()

# =========================================================
# PREDICTION
# =========================================================
def predict_single(input_dict):
    df = pd.DataFrame([input_dict])
    df['Kidney_Heart_Risk'] = df['age'] * df['serum_creatinine']

    num_cols = [
        'age', 'creatinine_phosphokinase', 'ejection_fraction',
        'platelets', 'serum_creatinine', 'serum_sodium',
        'time', 'Kidney_Heart_Risk'
    ]

    df[num_cols] = st.session_state.scaler.transform(df[num_cols])
    X = df[st.session_state.model_feature_names]

    pred = st.session_state.model.predict(X)[0]
    proba = st.session_state.model.predict_proba(X)[0][1]
    return pred, proba

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("1. Chargement des Données")
    uploaded = st.file_uploader("Uploader le CSV", type="csv")

    if uploaded:
        st.session_state.df = pd.read_csv(uploaded)
        st.session_state.data_loaded = True
        st.session_state.model_trained = False
        st.success("✅ CSV chargé avec succès")

    st.header("2. Modèle")
    model_choice = st.selectbox(
        "Choisir le modèle",
        ["Random Forest", "Logistic Regression", "Gradient Boosting"]
    )

    st.subheader("Hyperparamètres")
    params = {}
    if model_choice == "Random Forest":
        params["n_estimators"] = st.slider("Nombre d'Estimateurs (n_estimators)", 50, 300, 100)
        params["max_depth"] = st.slider("Profondeur Max (max_depth)", 3, 20, 10)
    elif model_choice == "Logistic Regression":
        params["C"] = st.slider("C", 0.01, 10.0, 1.0)
    else:
        params["n_estimators"] = st.slider("Nombre d'Estimateurs (n_estimators)", 50, 300, 100)
        params["learning_rate"] = st.slider("Taux d'apprentissage (learning_rate)", 0.01, 0.3, 0.1)

    if st.button(f"Entraîner le Modèle ({model_choice})", type="primary", use_container_width=True):
        if not st.session_state.data_loaded:
            st.error("❌ Veuillez d'abord charger un fichier CSV")
        else:
            with st.spinner("Entraînement en cours..."):
                X = st.session_state.df.drop(columns=['DEATH_EVENT'])
                y = st.session_state.df['DEATH_EVENT']

                Xp, scaler = preprocess_data(X)
                model, metrics, features = train_model(Xp, y, model_choice, params)

                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.metrics = metrics
                st.session_state.model_feature_names = features
                st.session_state.model_trained = True
                st.session_state.model_name = model_choice
                
            st.success("✅ Modèle entraîné avec succès!")
            st.balloons()

# =========================================================
# MAIN
# =========================================================
st.title("🫀 Prédiction du Risque Cardiaque")

if st.session_state.model_trained:
    # Afficher les métriques du modèle
    st.subheader(f"📊 Performances du Modèle ({st.session_state.model_name})")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{st.session_state.metrics['Accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{st.session_state.metrics['Precision']:.2%}")
    with col3:
        st.metric("Recall", f"{st.session_state.metrics['Recall']:.2%}")
    with col4:
        st.metric("F1-Score", f"{st.session_state.metrics['F1']:.2%}")
    with col5:
        st.metric("ROC-AUC", f"{st.session_state.metrics['ROC-AUC']:.2%}")

    st.divider()

    # Section de prédiction individuelle
    st.subheader("🔮 Prédiction Individuelle")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("##### Informations du Patient")
        age = st.number_input("Âge", min_value=40, max_value=100, value=60, step=1)
        sc = st.number_input("Créatinine sérique", min_value=0.5, max_value=9.4, value=1.5, step=0.1)
        ef = st.slider("Fraction d'éjection (%)", min_value=10, max_value=80, value=40, step=1)
        time = st.number_input("Temps de suivi (jours)", min_value=4, max_value=285, value=150, step=1)
    
    with col_b:
        st.markdown("##### Autres Paramètres")
        anaemia = st.selectbox("Anémie", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
        diabetes = st.selectbox("Diabète", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
        high_bp = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
        sex = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
        smoking = st.selectbox("Fumeur", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")

    input_data = {
        "age": age,
        "anaemia": anaemia,
        "creatinine_phosphokinase": 500,
        "diabetes": diabetes,
        "ejection_fraction": ef,
        "high_blood_pressure": high_bp,
        "platelets": 250000,
        "serum_creatinine": sc,
        "serum_sodium": 136,
        "sex": sex,
        "smoking": smoking,
        "time": time
    }

    if st.button("🔍 Prédire le Risque", type="primary", use_container_width=True):
        pred, prob = predict_single(input_data)
        
        st.divider()
        st.subheader("Résultat de la Prédiction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if pred == 1:
                st.error("### ❌ RISQUE ÉLEVÉ DE DÉCÈS")
            else:
                st.success("### ✅ RISQUE FAIBLE DE DÉCÈS")
        
        with col2:
            st.metric(
                "Probabilité de décès", 
                f"{prob*100:.2f}%",
                delta=f"{prob*100 - 50:.2f}%" if prob > 0.5 else f"{prob*100 - 50:.2f}%"
            )
        
        # Barre de progression pour visualiser la probabilité
        st.progress(prob)
        
        if prob > 0.7:
            st.warning("⚠️ Probabilité très élevée. Consultation médicale urgente recommandée.")
        elif prob > 0.5:
            st.info("ℹ️ Probabilité modérée. Surveillance médicale conseillée.")
        else:
            st.success("✅ Probabilité faible. Continuer le suivi médical régulier.")

else:
    st.info("ℹ️ Veuillez d'abord charger les données et entraîner le modèle via la barre latérale.")
    
    st.markdown("""
    ### 📋 Instructions:
    1. **Chargez votre fichier CSV** contenant les données des patients
    2. **Sélectionnez un modèle** (Random Forest, Logistic Regression, ou Gradient Boosting)
    3. **Ajustez les hyperparamètres** selon vos besoins
    4. **Cliquez sur "Entraîner le Modèle"** pour lancer l'apprentissage
    5. Une fois le modèle entraîné, vous pourrez faire des **prédictions individuelles**
    """)