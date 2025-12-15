import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import io

# --- 1. CONFIGURATION ET LAYOUT PROFESSIONNEL (Sidebars/Tabs) ---
st.set_page_config(
    page_title="Modèle de Prédiction des Maladies Cardiaques",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour calculer les limites d'outliers (pour le Capping)
def calculate_outlier_limits(df):
    outlier_cols = ['creatinine_phosphokinase', 'serum_creatinine', 'platelets']
    outlier_limits = {} 
    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        outlier_limits[col] = upper_limit
    return outlier_limits

# Fonction d'application du prétraitement (Capping, FE et Scaling)
# Appliqué à l'entraînement et à la prédiction individuelle
def apply_preprocessing(df_raw, scaler, outlier_limits, fit_scaler=False):
    df = df_raw.copy()
    
    # 1. Capping des Outliers
    for col, limit in outlier_limits.items():
        df[col] = np.where(df[col] > limit, limit, df[col])
        
    # 2. Feature Engineering brut (Kidney_Heart_Risk_Raw)
    df['Kidney_Heart_Risk_Raw'] = df['age'] * df['serum_creatinine']
    
    numerical_cols_to_scale = [
        'age', 'creatinine_phosphokinase', 'ejection_fraction', 
        'platelets', 'serum_creatinine', 'serum_sodium', 'time', 'Kidney_Heart_Risk_Raw'
    ]
    
    # 3. Scaling
    if fit_scaler:
        df[numerical_cols_to_scale] = scaler.fit_transform(df[numerical_cols_to_scale])
    else:
        # Utiliser .values pour éviter les avertissements de SettingWithCopy
        df[numerical_cols_to_scale] = scaler.transform(df[numerical_cols_to_scale].values)

    # 4. Création et suppression de la colonne intermédiaire
    df['Kidney_Heart_Risk'] = df['Kidney_Heart_Risk_Raw']
    df = df.drop(columns=['Kidney_Heart_Risk_Raw'])
    
    return df

# Fonction d'entraînement du modèle
def train_model(X, y, model_name, hyperparameters):
    if model_name == 'Random Forest':
        model = RandomForestClassifier(
            n_estimators=hyperparameters['n_estimators'], 
            max_depth=hyperparameters['max_depth'], 
            random_state=42
        )
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(
            C=hyperparameters['C'], 
            max_iter=1000, 
            random_state=42
        )
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(
            n_estimators=hyperparameters['n_estimators'], 
            learning_rate=hyperparameters['learning_rate'], 
            random_state=42
        )
    
    # Séparation des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else 'N/A',
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }
    
    return model, metrics, X_test.columns.tolist()

# --- 2. FONCTION DE PRÉDICTION EN TEMPS RÉEL (CORRIGÉE) ---
def make_single_prediction(model, scaler, feature_names, input_data, outlier_limits):
    """
    Effectue une prédiction unique en s'assurant que les étapes de prétraitement
    (Capping des outliers, FE, Scaling) et l'ordre des features correspondent
    à l'entraînement.
    """
    df_single = pd.DataFrame([input_data])
    
    # Appliquer le Prétraitement (SANS FIT)
    df_processed = apply_preprocessing(df_single, scaler, outlier_limits, fit_scaler=False)
    
    # Ajouter les features binaires qui n'ont PAS été scalées (elles sont dans df_single)
    binary_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
    for col in binary_cols:
        # Les binaires n'ont pas été transformées, les récupérer directement de l'input
        df_processed[col] = df_single[col]
        
    # Réordonner les features (TRÈS IMPORTANT)
    X_predict = df_processed[feature_names]
    
    # Prédiction
    prediction = model.predict(X_predict)[0]
    proba = model.predict_proba(X_predict)[0]
    
    return prediction, proba

# --- 3. GESTION DES SESSIONS ET DONNÉES PRÉ-CHARGÉES ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None # DataFrame brut
    st.session_state.df_processed_visual = None # DataFrame cappé (pour visualisation)
    st.session_state.target_col = 'DEATH_EVENT'
    st.session_state.feature_cols = [
        'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
        'ejection_fraction', 'high_blood_pressure', 'platelets', 
        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
    ]
    st.session_state.model = None
    st.session_state.scaler = StandardScaler() # Initialiser le scaler
    st.session_state.outlier_limits = None 
    st.session_state.metrics = None
    st.session_state.model_feature_names = None
    st.session_state.model_name = None

# Tentez de charger le dataset de Heart Failure (pré-chargé)
try:
    df_default = pd.read_csv('heart_failure_clinical_records_dataset.csv')
except FileNotFoundError:
    st.warning("Fichier 'heart_failure_clinical_records_dataset.csv' non trouvé. Veuillez l'uploader.")
    df_default = pd.DataFrame()


# --- TITRE DE L'APPLICATION ---
st.title("Système Interactif d'Analyse et de Prédiction Cardiaque")
st.markdown("Application d'apprentissage automatique basée sur les données d'insuffisance cardiaque (HF).")

# --- BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    st.header("1. Chargement des Données")
    
    # 3. Upload fichier CSV ou dataset pré-chargé
    data_source = st.radio(
        "Sélectionnez la source de données :",
        ["Dataset pré-chargé (HF)", "Uploader un fichier CSV"],
        index=0 if not df_default.empty else 1
    )
    
    df_uploaded = None
    if data_source == "Uploader un fichier CSV":
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])
        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.data_loaded = True
            st.session_state.model = None # Réinitialiser le modèle si nouvelles données
            st.success("Fichier CSV chargé avec succès.")
    elif data_source == "Dataset pré-chargé (HF)" and not df_default.empty:
        st.session_state.df = df_default
        st.session_state.data_loaded = True
        st.success("Dataset Heart Failure chargé.")
    
    # Si les données sont chargées, calculer les limites pour les stocker et la visualisation
    if st.session_state.data_loaded and st.session_state.df is not None and st.session_state.outlier_limits is None:
        st.session_state.outlier_limits = calculate_outlier_limits(st.session_state.df)
        
        # Préparer le DataFrame pour la visualisation (Capping uniquement, SANS Scaling)
        df_visual = st.session_state.df.copy()
        for col, limit in st.session_state.outlier_limits.items():
            df_visual[col] = np.where(df_visual[col] > limit, limit, df_visual[col])
        st.session_state.df_processed_visual = df_visual
        # Ne pas oublier la FE pour la visualisation
        st.session_state.df_processed_visual['Kidney_Heart_Risk'] = st.session_state.df_processed_visual['age'] * st.session_state.df_processed_visual['serum_creatinine']


    # --- 5. SÉLECTION DU MODÈLE ET HYPERPARAMÈTRES ---
    st.header("2. Configuration du Modèle")
    model_choice = st.selectbox(
        "Sélectionnez le Modèle:",
        ['Random Forest', 'Logistic Regression', 'Gradient Boosting'],
        index=0
    )
    
    st.subheader("Hyperparamètres")
    hyperparameters = {}
    if model_choice == 'Random Forest':
        hyperparameters['n_estimators'] = st.slider("Nombre d'Estimateurs (n_estimators)", 50, 500, 100, 50)
        hyperparameters['max_depth'] = st.slider("Profondeur Max (max_depth)", 5, 20, 10, 1)
    elif model_choice == 'Logistic Regression':
        hyperparameters['C'] = st.slider("Force de Régularisation (C)", 0.01, 10.0, 1.0, 0.01)
    elif model_choice == 'Gradient Boosting':
        hyperparameters['n_estimators'] = st.slider("Nombre d'Estimateurs (n_estimators)", 50, 500, 100, 50)
        hyperparameters['learning_rate'] = st.slider("Taux d'Apprentissage (learning_rate)", 0.01, 0.3, 0.1, 0.01)

    if st.session_state.data_loaded:
        if st.button(f"Entraîner le Modèle ({model_choice})"):
            with st.spinner(f"Entraînement du modèle {model_choice} en cours..."):
                
                # 1. Préparation des données d'entraînement brutes
                X_train_data_raw = st.session_state.df.drop(columns=[st.session_state.target_col])
                y = st.session_state.df[st.session_state.target_col]

                # 2. Appliquer le preprocessing (FIT SCALER = True)
                X = apply_preprocessing(
                    X_train_data_raw, 
                    st.session_state.scaler, 
                    st.session_state.outlier_limits,
                    fit_scaler=True # Ajuste le scaler
                )
                
                # 3. Entraînement
                model, metrics, feature_names = train_model(X, y, model_choice, hyperparameters)
                
                # 4. Sauvegarde dans la session
                st.session_state.model = model
                st.session_state.metrics = metrics
                st.session_state.model_feature_names = feature_names
                st.session_state.model_name = model_choice
                st.success(f"Modèle {model_choice} entraîné et prêt pour la prédiction!")
                st.balloons()
            
    # --- 6. PRÉDICTIONS EN TEMPS RÉEL SUR NOUVEAUX EXEMPLES ---
    st.header("3. Prédiction Individuelle")
    
    if st.session_state.model:
        st.subheader(f"Input pour {st.session_state.model_name}")
        
        # Champs d'entrée pour les features (simulés avec des valeurs par défaut)
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Âge", min_value=40, max_value=100, value=60)
            anaemia = st.selectbox("Anémie (0/1)", options=[0, 1], index=0)
            cpk = st.number_input("CPK (creatinine_phosphokinase)", min_value=23, max_value=7861, value=500)
            diabetes = st.selectbox("Diabète (0/1)", options=[0, 1], index=0)
            ejection_fraction = st.slider("Fraction d'Éjection (%)", 10, 80, 40)
            
        with col2:
            hbp = st.selectbox("Hypertension (0/1)", options=[0, 1], index=0)
            platelets = st.number_input("Plaquettes (kilo/mL)", min_value=25000, max_value=850000, value=250000)
            sc = st.number_input("Créatinine Sérique (mg/dL)", min_value=0.5, max_value=9.4, value=1.5, format="%.2f")
            sn = st.number_input("Sodium Sérique (mEq/L)", min_value=113, max_value=148, value=136)
            sex = st.selectbox("Sexe (0=Femme, 1=Homme)", options=[0, 1], index=1)
            smoking = st.selectbox("Fumeur (0/1)", options=[0, 1], index=0)
            time = st.number_input("Période de Suivi (jours)", min_value=4, max_value=285, value=150)
            
        input_data = {
            'age': age, 'anaemia': anaemia, 'creatinine_phosphokinase': cpk, 
            'diabetes': diabetes, 'ejection_fraction': ejection_fraction, 
            'high_blood_pressure': hbp, 'platelets': platelets, 
            'serum_creatinine': sc, 'serum_sodium': sn, 'sex': sex, 
            'smoking': smoking, 'time': time
        }
        
        if st.button("Prédir le Risque"):
            try:
                prediction, proba = make_single_prediction(
                    st.session_state.model, 
                    st.session_state.scaler, 
                    st.session_state.model_feature_names,
                    input_data,
                    st.session_state.outlier_limits 
                )
                
                risk = "❌ RISQUE DE DÉCÈS ÉLEVÉ" if prediction == 1 else "✅ FAIBLE RISQUE DE DÉCÈS (Survie)"
                proba_risk = proba[1] * 100
                
                st.subheader("Résultat de la Prédiction")
                st.metric(label="Statut Prédit", value=risk)
                st.info(f"Probabilité de Décès: **{proba_risk:.2f}%**")
                
            except Exception as e:
                st.error(f"Erreur lors de la prédiction. Erreur : {e}")
    else:
        st.info("Veuillez d'abord charger les données et entraîner un modèle.")


# --- CONTENU PRINCIPAL (TABS) ---
tab1, tab2, tab3 = st.tabs([
    "4. Vue d'ensemble des Données et Statistiques", 
    "7. & 8. Visualisations et Performance du Modèle", 
    "9. Export des Résultats"
])

# --- TAB 1 : VUE D'ENSEMBLE DES DONNÉES ---
with tab1:
    st.header("Prévisualisation et Statistiques des Données")
    
    if st.session_state.data_loaded:
        df = st.session_state.df
        st.subheader(f"Aperçu des Données brutes (5 premières lignes)")
        st.dataframe(df.head())
        
        st.subheader("Statistiques Descriptives")
        st.dataframe(df.describe().T.style.format("{:.2f}"))
        
        # 4. Statistiques des données
        st.subheader("Informations Clés du Dataset")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nombre de Lignes", df.shape[0])
        col2.metric("Nombre de Colonnes", df.shape[1])
        col3.metric("Valeurs Manquantes (Total)", df.isnull().sum().sum())
        
        target_counts = df[st.session_state.target_col].value_counts(normalize=True) * 100
        col4.metric(
            f"Taux d'Événements Cibles (1=Décès)", 
            f"{target_counts.get(1, 0):.2f}%"
        )
        
    else:
        st.info("Veuillez charger ou sélectionner un dataset dans la barre latérale pour voir la prévisualisation et les statistiques.")

# --- TAB 2 : VISUALISATIONS ET PERFORMANCE ---
with tab2:
    st.header("Visualisations Dynamiques et Évaluation du Modèle")
    
    if st.session_state.model:
        st.subheader(f"8. Métriques de Performance du Modèle : {st.session_state.model_name}")
        
        metrics = st.session_state.metrics
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        col2.metric("Precision", f"{metrics['Precision']:.4f}")
        col3.metric("Recall", f"{metrics['Recall']:.4f}")
        col4.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
        col5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")
        
        # Matrice de Confusion
        st.subheader("Matrice de Confusion")
        cm = metrics['Confusion Matrix']
        st.code(
            f"[[{cm[0, 0]} (Vrais Négatifs)  {cm[0, 1]} (Faux Positifs)]\n"
            f" [{cm[1, 0]} (Faux Négatifs)  {cm[1, 1]} (Vrais Positifs)]]"
        )
        st.markdown(
            "*(Vrais Positifs = Décès prédits correctement, Faux Négatifs = Décès manqués)*"
        )

        # 7. Visualisations dynamiques (Feature Importance)
        if hasattr(st.session_state.model, 'feature_importances_'):
            st.subheader("Importance des Caractéristiques (Feature Importance)")
            
            # Récupérer les noms des features depuis le modèle (scalées/featurisées)
            feature_names = st.session_state.model_feature_names
            
            # Si le modèle est LogReg (pas de feature_importances_)
            if st.session_state.model_name != 'Logistic Regression':
                importances = pd.Series(
                    st.session_state.model.feature_importances_, 
                    index=feature_names
                ).sort_values(ascending=False).head(10)
                
                fig_importance = px.bar(
                    importances,
                    x=importances.values,
                    y=importances.index,
                    orientation='h',
                    title="Top 10 Caractéristiques les plus importantes",
                    labels={'x': 'Importance', 'y': 'Caractéristique'},
                    height=400,
                )
                fig_importance.update_layout(xaxis_title="Importance du Modèle", yaxis_title="")
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info("L'importance des caractéristiques n'est pas directement disponible pour la Régression Logistique (utilisez les coefficients du modèle si nécessaire).")
            
            st.info(
                "L'importance des features après l'entraînement est affichée ici. "
                "Ceci confirme généralement que la durée de suivi (`time`), la fonction cardiaque (`ejection_fraction`) et les marqueurs rénaux (`serum_creatinine`) sont les plus critiques."
            )

    # 7. Visualisations dynamiques (Distribution par Variable Cible)
    st.subheader("Analyse de la Distribution des Variables Clés")
    
    if st.session_state.data_loaded and st.session_state.df_processed_visual is not None:
        # Utiliser le DF préparé pour la visualisation (cappé/FE, mais non scalé)
        df_viz = st.session_state.df_processed_visual.copy()
        df_viz[st.session_state.target_col] = st.session_state.df[st.session_state.target_col]
        
        # Liste de toutes les colonnes disponibles pour la visualisation
        viz_options = df_viz.columns.drop(st.session_state.target_col, errors='ignore').tolist()
        
        viz_col = st.selectbox(
            "Sélectionnez une variable pour voir sa distribution par événement de décès (target=DEATH_EVENT) :",
            options=viz_options
        )
        
        fig_dist = px.box(
            df_viz,
            x=st.session_state.target_col,
            y=viz_col,
            color=st.session_state.target_col,
            title=f"Distribution de '{viz_col}' par Issue (0=Survie, 1=Décès)",
            labels={st.session_state.target_col: "Événement de Décès (0:Survie, 1:Décès)", viz_col: viz_col},
            color_discrete_map={0: 'blue', 1: 'red'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown(
            "*(Note: Les valeurs extrêmes (outliers) des colonnes CPK, Créatinine Sérique et Plaquettes sont plafonnées dans ce graphique pour des raisons de cohérence avec le prétraitement du modèle.)*"
        )
    else:
        st.info("Veuillez charger un dataset pour afficher les visualisations de distribution.")


# --- TAB 3 : EXPORT DES RÉSULTATS ---
with tab3:
    st.header("9. Export des Résultats")
    
    # 9. Export résultats (CSV/JSON)
    if st.session_state.metrics:
        st.subheader("Export des Métriques du Modèle")
        metrics_df = pd.DataFrame({
            'Metric': list(st.session_state.metrics.keys()),
            'Value': [
                f"{v:.4f}" if isinstance(v, float) else str(v).replace('\n', ' ') 
                for v in st.session_state.metrics.values()
            ]
        }).iloc[:-1] # Exclut la matrice de confusion pour l'export CSV
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        @st.cache_data
        def convert_df_to_json(df):
            return df.to_json(orient="records").encode('utf-8')
        
        csv_export = convert_df_to_csv(metrics_df)
        json_export = convert_df_to_json(metrics_df)

        st.dataframe(metrics_df)
        
        col_csv, col_json = st.columns(2)
        with col_csv:
            st.download_button(
                label="Télécharger les Métriques en CSV",
                data=csv_export,
                file_name=f'{st.session_state.model_name}_metrics.csv',
                mime='text/csv',
            )
        with col_json:
            st.download_button(
                label="Télécharger les Métriques en JSON",
                data=json_export,
                file_name=f'{st.session_state.model_name}_metrics.json',
                mime='application/json',
            )
    else:
        st.info("Entraînez un modèle pour activer les options d'export.")