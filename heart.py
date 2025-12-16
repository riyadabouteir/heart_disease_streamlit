import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import StandardScaler
import pickle # Ajout pour sérialiser le scaler et les modèles

# Configuration de la page (Reste inchangé)
st.set_page_config(
    page_title="Dashboard BI - Insuffisance Cardiaque",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé (Reste inchangé)
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 1rem;
        border: 2px dashed #1f77b4;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🫀 Dashboard BI - Analyse d\'Insuffisance Cardiaque</h1>', 
            unsafe_allow_html=True)

# Fonction pour le feature engineering (Reste inchangée)
def feature_engineering(df):
    """Applique le feature engineering sur le dataset Heart Failure"""
    df_engineered = df.copy()
    
    # Création de nouvelles features
    df_engineered['Age_Group'] = pd.cut(df_engineered['age'], 
                                        bins=[0, 50, 60, 70, 100], 
                                        labels=['<50', '50-60', '60-70', '70+'])
    
    df_engineered['Kidney_Heart_Risk'] = (
        df_engineered['serum_creatinine'] * df_engineered['high_blood_pressure']
    )
    
    df_engineered['Anemia_Diabetes'] = (
        df_engineered['anaemia'] & df_engineered['diabetes']
    ).astype(int)
    
    return df_engineered

# Fonction pour entraîner les modèles
@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    """Entraîne les trois modèles de classification et retourne les modèles, scaler et résultats"""
    models = {}
    results = {}
    
    # Standardisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf
    
    # 3. Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train_scaled, y_train)
    models['Gradient Boosting'] = gb
    
    # Calcul des métriques pour chaque modèle
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    return models, results, scaler, X_test, y_test

# Fonction pour la prédiction d'un nouveau patient
def predict_new_patient(model, scaler, patient_data):
    """
    Prédit le risque de décès pour un nouveau patient.
    patient_data est un DataFrame 1 ligne avec les features.
    """
    # 1. Feature Engineering (pour Kidney_Heart_Risk et Anemia_Diabetes)
    patient_data['Kidney_Heart_Risk'] = (
        patient_data['serum_creatinine'] * patient_data['high_blood_pressure']
    )
    patient_data['Anemia_Diabetes'] = (
        patient_data['anaemia'] & patient_data['diabetes']
    ).astype(int)
    
    # 2. Sélection et Ordre des features (doit correspondre à l'entraînement)
    features_to_use = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                       'ejection_fraction', 'high_blood_pressure', 'platelets',
                       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
                       'Kidney_Heart_Risk', 'Anemia_Diabetes']
    
    X_new = patient_data[features_to_use]
    
    # 3. Scaling (Utiliser le scaler FIT sur les données d'entraînement)
    X_new_scaled = scaler.transform(X_new)
    
    # 4. Prédiction
    prediction = model.predict(X_new_scaled)[0]
    proba = model.predict_proba(X_new_scaled)[0][1] # Probabilité de décès (classe 1)
    
    return prediction, proba

# Sidebar avec upload de fichier
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=100)
    st.title("Navigation")
    
    # Section d'upload de fichier
    st.markdown("### 📁 Charger les Données")
    
    uploaded_file = st.file_uploader(
        "Uploader le fichier CSV",
        type=['csv'],
        help="Téléchargez le fichier heart_failure_clinical_records_dataset.csv"
    )
    
    st.divider()
    
    # Navigation (seulement si les données sont chargées)
    if uploaded_file is not None or 'df_failure' in st.session_state:
        page = st.radio(
            "Sélectionnez une section:",
            ["🏠 Accueil",
             "📊 Exploration des Données (EDA)",
             "🔬 Feature Engineering",
             "🤖 Modélisation & Prédictions",
             "🧪 Prédictions Individuelles", # Nouvelle page
             "📈 Comparaison des Modèles",
             "💡 Insights & Recommandations"]
        )
    else:
        page = "🏠 Accueil"
    
    st.divider()
    st.markdown("### À propos")
    st.info("""
    **Dashboard BI - Analyse Prédictive**
    
    Ce tableau de bord analyse le dataset Heart Failure Clinical Records pour prédire les événements de décès chez les patients atteints d'insuffisance cardiaque.
    """)

# Gestion du chargement des données
if uploaded_file is not None:
    try:
        df_failure = pd.read_csv(uploaded_file)
        st.session_state['df_failure'] = df_failure
        st.sidebar.success(f"✅ Données chargées : {len(df_failure)} patients")
    except Exception as e:
        st.sidebar.error(f"❌ Erreur lors du chargement : {str(e)}")
        df_failure = None
elif 'df_failure' in st.session_state:
    df_failure = st.session_state['df_failure']
else:
    df_failure = None

# Si pas de données, afficher la page d'upload (Reste inchangé)
if df_failure is None:
    st.markdown("""
    <div class="upload-section">
        <h2 style="text-align: center; color: #1f77b4;">📤 Bienvenue !</h2>
        <p style="text-align: center; font-size: 1.2rem;">
            Pour commencer l'analyse, veuillez uploader votre fichier de données.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ### 📋 Format du fichier attendu
        
        Le fichier doit être un CSV contenant les colonnes suivantes :
        
        - **age** : Âge du patient
        - **anaemia** : Anémie (0 ou 1)
        - **creatinine_phosphokinase** : Niveau de CPK
        - **diabetes** : Diabète (0 ou 1)
        - **ejection_fraction** : Fraction d'éjection
        - **high_blood_pressure** : Hypertension (0 ou 1)
        - **platelets** : Plaquettes
        - **serum_creatinine** : Créatinine sérique
        - **serum_sodium** : Sodium sérique
        - **sex** : Sexe (0=F, 1=M)
        - **smoking** : Fumeur (0 ou 1)
        - **time** : Période de suivi
        - **DEATH_EVENT** : Décès (0 ou 1)
        
        ---
        
        ### 🔍 Exemple de données
        
        Si vous n'avez pas le fichier, vous pouvez :
        1. Télécharger depuis [Kaggle](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)
        2. Ou générer des données de démonstration avec le script fourni
        
        ---
        
        👈 **Utilisez le bouton d'upload dans la barre latérale pour commencer !**
        """)
        
        # Bouton pour télécharger un exemple
        st.markdown("### 📥 Télécharger un template")
        
        # Créer un template CSV
        template_data = {
            'age': [65.0, 70.0, 60.0],
            'anaemia': [0, 1, 0],
            'creatinine_phosphokinase': [582, 231, 200],
            'diabetes': [0, 1, 0],
            'ejection_fraction': [20, 25, 35],
            'high_blood_pressure': [0, 0, 1],
            'platelets': [265000.0, 194000.0, 250000.0],
            'serum_creatinine': [1.9, 1.2, 1.1],
            'serum_sodium': [130, 136, 137],
            'sex': [1, 0, 1],
            'smoking': [0, 0, 1],
            'time': [4, 10, 20],
            'DEATH_EVENT': [1, 0, 0]
        }
        template_df = pd.DataFrame(template_data)
        
        csv = template_df.to_csv(index=False)
        st.download_button(
            label="📄 Télécharger un template CSV",
            data=csv,
            file_name="template_heart_failure.csv",
            mime="text/csv",
            help="Téléchargez ce template pour voir le format attendu"
        )

else:
    # Les données sont chargées, afficher les pages
    
    # Préparation des données pour le ML (nécessaire pour plusieurs pages)
    df_model = feature_engineering(df_failure)
    
    # Définir les features
    features_to_use = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                       'ejection_fraction', 'high_blood_pressure', 'platelets',
                       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
                       'Kidney_Heart_Risk', 'Anemia_Diabetes']
    
    X = df_model[features_to_use]
    y = df_model['DEATH_EVENT']
    
    # Split initial (pour l'entraînement)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entraînement et récupération des modèles et du scaler
    with st.spinner("⏳ Préparation des modèles en arrière-plan..."):
        models, results, scaler, X_test_df, y_test_series = train_models(X_train, X_test, y_train, y_test)
        
    st.session_state['models'] = models
    st.session_state['results'] = results
    st.session_state['scaler'] = scaler
    st.session_state['X_test_df'] = X_test_df
    st.session_state['y_test_series'] = y_test_series
    
    
    # PAGE 1: ACCUEIL (Reste inchangé)
    if page == "🏠 Accueil":
        st.header("Bienvenue sur le Dashboard BI")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("👥 Nombre de Patients", df_failure.shape[0])
        
        with col2:
            st.metric("📊 Nombre de Variables", df_failure.shape[1])
        
        with col3:
            death_rate = (df_failure['DEATH_EVENT'].sum() / len(df_failure) * 100)
            st.metric("💀 Taux de Mortalité", f"{death_rate:.1f}%")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Aperçu des Données")
            st.dataframe(df_failure.head(10), use_container_width=True)
        
        with col2:
            st.subheader("📊 Statistiques Descriptives")
            st.dataframe(df_failure.describe(), use_container_width=True)
        
        st.divider()
        
        # Distribution de DEATH_EVENT
        st.subheader("🎯 Distribution des Événements de Décès")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.pie(df_failure, names='DEATH_EVENT', 
                         title='Répartition Survie vs Décès',
                         color='DEATH_EVENT',
                         color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'},
                         labels={'DEATH_EVENT': 'Événement'})
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Détails")
            value_counts = df_failure['DEATH_EVENT'].value_counts()
            st.metric("✅ Survie (0)", value_counts[0])
            st.metric("❌ Décès (1)", value_counts[1])
            ratio = value_counts[1] / value_counts[0]
            st.metric("📊 Ratio Décès/Survie", f"{ratio:.3f}")
        
        # Informations sur les valeurs manquantes
        st.subheader("🔍 Qualité des Données")
        missing = df_failure.isnull().sum()
        if missing.sum() == 0:
            st.success("✅ Aucune valeur manquante détectée !")
        else:
            st.warning(f"⚠️ {missing.sum()} valeurs manquantes détectées")
            st.dataframe(missing[missing > 0], use_container_width=True)
    
    # PAGE 2: EXPLORATION DES DONNÉES (Reste inchangé)
    elif page == "📊 Exploration des Données (EDA)":
        st.header("Analyse Exploratoire des Données (EDA)")
        
        # Distribution de la variable cible
        st.subheader("🎯 Distribution de DEATH_EVENT")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.histogram(df_failure, x='DEATH_EVENT', 
                               color='DEATH_EVENT',
                               title="Distribution des Événements de Décès",
                               labels={'DEATH_EVENT': 'Événement (0=Survie, 1=Décès)'},
                               color_discrete_sequence=['#4ECDC4', '#FF6B6B'],
                               text_auto=True)
            fig.update_layout(showlegend=False, bargap=0.2)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            value_counts = df_failure['DEATH_EVENT'].value_counts()
            st.metric("Survie (0)", value_counts[0], 
                      delta=f"{value_counts[0]/len(df_failure)*100:.1f}%")
            st.metric("Décès (1)", value_counts[1],
                      delta=f"{value_counts[1]/len(df_failure)*100:.1f}%")
            st.metric("Ratio", f"{value_counts[1]/value_counts[0]:.3f}")
        
        st.divider()
        
        # Distributions des variables numériques
        st.subheader("📈 Distributions des Variables Numériques")
        
        numeric_cols = df_failure.select_dtypes(include=[np.number]).columns.tolist()
        if 'DEATH_EVENT' in numeric_cols:
            numeric_cols.remove('DEATH_EVENT')
        
        selected_var = st.selectbox("Sélectionnez une variable:", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df_failure, x=selected_var, 
                               color='DEATH_EVENT',
                               marginal="box",
                               title=f"Distribution de {selected_var}",
                               color_discrete_sequence=['#4ECDC4', '#FF6B6B'],
                               labels={'DEATH_EVENT': 'Événement'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df_failure, x='DEATH_EVENT', y=selected_var,
                         color='DEATH_EVENT',
                         title=f"Boxplot de {selected_var} par classe",
                         color_discrete_sequence=['#4ECDC4', '#FF6B6B'],
                         labels={'DEATH_EVENT': 'Événement'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Matrice de corrélation
        st.subheader("🔗 Matrice de Corrélation")
        
        corr_matrix = df_failure.select_dtypes(include=[np.number]).corr()
        
        fig = px.imshow(corr_matrix, 
                        text_auto='.2f',
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Matrice de Corrélation",
                        zmin=-1, zmax=1)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top corrélations avec DEATH_EVENT
        st.subheader("🎯 Top Corrélations avec DEATH_EVENT")
        
        target_corr = corr_matrix['DEATH_EVENT'].drop('DEATH_EVENT').sort_values(key=abs, ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(x=target_corr.values, 
                         y=target_corr.index,
                         orientation='h',
                         title="Corrélations avec DEATH_EVENT",
                         labels={'x': 'Corrélation', 'y': 'Variable'},
                         color=target_corr.values,
                         color_continuous_scale='RdBu_r')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🔝 Top 5 Corrélations")
            for i, (var, corr) in enumerate(target_corr.head(5).items(), 1):
                emoji = "📈" if corr > 0 else "📉"
                st.metric(f"{i}. {var}", f"{corr:.3f}", delta=emoji)
    
    # PAGE 3: FEATURE ENGINEERING (Reste inchangé)
    elif page == "🔬 Feature Engineering":
        st.header("Feature Engineering")
        
        st.info("""
        Dans cette section, nous créons de nouvelles variables pour améliorer la performance 
        des modèles prédictifs.
        """)
        
        # Application du feature engineering
        df_engineered = feature_engineering(df_failure)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Dataset Original")
            st.write(f"Nombre de colonnes: {df_failure.shape[1]}")
            st.dataframe(df_failure.head(), use_container_width=True)
        
        with col2:
            st.subheader("✨ Dataset avec Feature Engineering")
            st.write(f"Nombre de colonnes: {df_engineered.shape[1]}")
            st.dataframe(df_engineered.head(), use_container_width=True)
        
        st.divider()
        
        # Nouvelles features créées
        st.subheader("🆕 Nouvelles Features Créées")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 1️⃣ Age_Group
            Catégorisation de l'âge en 4 groupes :
            - **<50 ans** : Jeunes
            - **50-60 ans** : Adultes
            - **60-70 ans** : Seniors
            - **70+ ans** : Âgés
            """)
        
        with col2:
            st.markdown("""
            #### 2️⃣ Kidney_Heart_Risk
            Interaction entre :
            - **serum_creatinine** (fonction rénale)
            - **high_blood_pressure** (hypertension)
            
            Risque = créatinine $\\times$ hypertension
            """)
        
        with col3:
            st.markdown("""
            #### 3️⃣ Anemia_Diabetes
            Combinaison binaire :
            - **anaemia** (anémie)
            - **diabetes** (diabète)
            
            1 si les deux conditions présentes
            """)
        
        st.divider()
        
        # Visualisation de Age_Group
        st.subheader("📊 Visualisation : Age_Group")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df_engineered, x='Age_Group', 
                               color='DEATH_EVENT',
                               title="Distribution des Groupes d'Âge par Outcome",
                               barmode='group',
                               color_discrete_sequence=['#4ECDC4', '#FF6B6B'],
                               labels={'DEATH_EVENT': 'Événement'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df_engineered, x='Age_Group', y='Kidney_Heart_Risk',
                         color='DEATH_EVENT',
                         title="Kidney_Heart_Risk par Groupe d'Âge",
                         color_discrete_sequence=['#4ECDC4', '#FF6B6B'],
                         labels={'DEATH_EVENT': 'Événement'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Impact des nouvelles features
        st.subheader("🔗 Impact des Nouvelles Features")
        
        new_features = ['Kidney_Heart_Risk', 'Anemia_Diabetes']
        corr_new = df_engineered[new_features + ['DEATH_EVENT']].corr()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.imshow(corr_new, 
                            text_auto='.3f',
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            title="Corrélations des Nouvelles Features avec DEATH_EVENT",
                            zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Résultats")
            for feature in new_features:
                corr_value = corr_new.loc[feature, 'DEATH_EVENT']
                st.metric(feature, f"{corr_value:.4f}",
                          delta="Positive" if corr_value > 0 else "Négative")
            
            st.markdown("""
            ---
            ### 💡 Interprétation
            
            - **Kidney_Heart_Risk** : Montre une corrélation significative avec la mortalité
            - **Anemia_Diabetes** : Combinaison utile pour identifier les patients à risque
            """)
    
    # PAGE 4: MODÉLISATION & PRÉDICTIONS (Mise à jour pour utiliser les session_state)
    elif page == "🤖 Modélisation & Prédictions":
        st.header("Modélisation & Prédictions")
        
        st.info("""
        Cette section présente les performances des modèles entraînés sur le dataset pour prédire le risque de décès.
        
        - ⚙️ Régression Logistique
        - 🌲 Random Forest (Recommandé)
        - 📈 Gradient Boosting
        """)
        
        results = st.session_state['results']
        models = st.session_state['models']
        
        st.divider()
        
        # Performances
        st.subheader("📊 Performances des Modèles")
        
        results_df = pd.DataFrame({
            name: {
                'Accuracy': res['Accuracy'],
                'Precision': res['Precision'],
                'Recall': res['Recall'],
                'F1-Score': res['F1-Score'],
                'ROC-AUC': res['ROC-AUC']
            }
            for name, res in results.items()
        }).T
        
        # Formatter avec couleurs
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: lightgreen' if v else '' for v in is_max]
        
        st.dataframe(results_df.style.apply(highlight_max, axis=0), 
                     use_container_width=True)
        
        st.divider()
        
        # Visualisations
        col1, col2 = st.columns(2)
        
        with col1:
            # Comparaison des métriques
            st.subheader("📊 Comparaison des Métriques")
            
            fig = go.Figure()
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            
            for model_name in results.keys():
                values = [results[model_name][metric] for metric in metrics]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=model_name
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature Importance (Random Forest)
            st.subheader("🌲 Feature Importance (Random Forest)")
            
            rf_model = models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': features_to_use,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            fig = px.bar(feature_importance, 
                         x='importance', 
                         y='feature',
                         orientation='h',
                         title="Top 10 Variables Importantes",
                         color='importance',
                         color_continuous_scale='Viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Matrice de confusion
        st.subheader("🎯 Matrice de Confusion (Random Forest)")
        
        best_model = 'Random Forest'
        cm = confusion_matrix(st.session_state['y_test_series'], results[best_model]['y_pred'])
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            fig = px.imshow(cm, 
                            text_auto=True,
                            labels=dict(x="Prédiction", y="Réalité", color="Count"),
                            x=['Survie', 'Décès'],
                            y=['Survie', 'Décès'],
                            color_continuous_scale='Blues',
                            title=f"Matrice de Confusion - {best_model}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Rapport de classification
        with st.expander("📄 Rapport de Classification Détaillé"):
            report = classification_report(st.session_state['y_test_series'], 
                                           results[best_model]['y_pred'], 
                                           target_names=['Survie', 'Décès'])
            st.text(report)

    # NOUVELLE PAGE 5: PRÉDICTIONS INDIVIDUELLES
    elif page == "🧪 Prédictions Individuelles":
        st.header("Prédiction du Risque de Décès pour un Nouveau Patient")
        
        st.info("""
        Utilisez les curseurs pour définir les paramètres d'un nouveau patient et obtenir une estimation de son risque de décès, 
        basée sur le modèle le plus performant (**Random Forest**).
        """)
        
        # Utiliser les modèles et scaler stockés
        models = st.session_state['models']
        scaler = st.session_state['scaler']
        best_model_name = max(st.session_state['results'].items(), key=lambda x: x[1]['ROC-AUC'])[0]
        model = models[best_model_name]
        
        # Trouver les min/max pour les inputs
        df_desc = df_failure.describe().T
        
        # Formulaire de saisie
        with st.form("patient_form"):
            st.subheader("Paramètres Démographiques et Cliniques")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.slider("Âge (années)", min_value=int(df_desc.loc['age', 'min']), 
                                max_value=int(df_desc.loc['age', 'max']), value=60)
                sex = st.selectbox("Sexe", options=[1, 0], format_func=lambda x: 'Homme' if x == 1 else 'Femme', index=0)
                smoking = st.selectbox("Fumeur", options=[0, 1], format_func=lambda x: 'Non' if x == 0 else 'Oui', index=0)
            
            with col2:
                diabetes = st.selectbox("Diabète", options=[0, 1], format_func=lambda x: 'Non' if x == 0 else 'Oui', index=0)
                anaemia = st.selectbox("Anémie", options=[0, 1], format_func=lambda x: 'Non' if x == 0 else 'Oui', index=0)
                high_blood_pressure = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: 'Non' if x == 0 else 'Oui', index=0)
                
            with col3:
                time = st.slider("Période de Suivi (jours)", min_value=int(df_desc.loc['time', 'min']), 
                                 max_value=int(df_desc.loc['time', 'max']), value=150)
            
            st.subheader("Résultats des Biomarqueurs")
            col4, col5, col6 = st.columns(3)
            
            with col4:
                ejection_fraction = st.slider("Fraction d'Éjection (%)", 
                                              min_value=int(df_desc.loc['ejection_fraction', 'min']), 
                                              max_value=int(df_desc.loc['ejection_fraction', 'max']), value=35)
                
            with col5:
                serum_creatinine = st.slider("Créatinine Sérique (mg/dL)", 
                                             min_value=float(df_desc.loc['serum_creatinine', 'min']), 
                                             max_value=float(df_desc.loc['serum_creatinine', 'max']), value=1.4, step=0.1)
                
            with col6:
                serum_sodium = st.slider("Sodium Sérique (mEq/L)", 
                                         min_value=int(df_desc.loc['serum_sodium', 'min']), 
                                         max_value=int(df_desc.loc['serum_sodium', 'max']), value=136)
            
            col7, col8 = st.columns(2)
            with col7:
                creatinine_phosphokinase = st.slider("CPK (mcg/L)", 
                                                     min_value=int(df_desc.loc['creatinine_phosphokinase', 'min']), 
                                                     max_value=int(df_desc.loc['creatinine_phosphokinase', 'max']), value=500)
            with col8:
                platelets = st.slider("Plaquettes (kiloplaquettes/mL)", 
                                      min_value=int(df_desc.loc['platelets', 'min']), 
                                      max_value=int(df_desc.loc['platelets', 'max']), value=250000)
                
            submitted = st.form_submit_button("Calculer le Risque de Décès")

        if submitted:
            # Créer un DataFrame pour le nouveau patient
            new_patient_data = pd.DataFrame({
                'age': [age],
                'anaemia': [anaemia],
                'creatinine_phosphokinase': [creatinine_phosphokinase],
                'diabetes': [diabetes],
                'ejection_fraction': [ejection_fraction],
                'high_blood_pressure': [high_blood_pressure],
                'platelets': [platelets],
                'serum_creatinine': [serum_creatinine],
                'serum_sodium': [serum_sodium],
                'sex': [sex],
                'smoking': [smoking],
                'time': [time]
            })
            
            # Faire la prédiction
            prediction, proba = predict_new_patient(model, scaler, new_patient_data)
            
            risk_percent = proba * 100
            
            st.divider()
            st.subheader(f"Résultats de la Prédiction ({best_model_name})")
            
            if prediction == 1:
                st.error(f"❌ Patient à **Risque Élevé** de Décès.")
                st.markdown(f"**Probabilité estimée de décès :** **{risk_percent:.2f}%**")
                
                if ejection_fraction < 30:
                    st.warning("🚨 Alerte : La Fraction d'Éjection est très faible (< 30%), un facteur de risque majeur.")
                if serum_creatinine > 1.5:
                    st.warning("🚨 Alerte : La Créatinine Sérique est élevée (> 1.5), indiquant un risque rénal/cardiaque accru.")
                if time < 50:
                    st.warning("🚨 Alerte : Le temps de suivi est court (< 50 jours), le risque est maximal en début de suivi.")
                    
            else:
                st.success(f"✅ Patient à **Faible Risque** de Décès.")
                st.markdown(f"**Probabilité estimée de décès :** **{risk_percent:.2f}%**")
                if risk_percent > 30:
                    st.info("💡 Note : Bien que la prédiction soit 'Survie', la probabilité reste modérée. Une surveillance est conseillée.")
            
            # Jauge de risque
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_percent,
                title = {'text': "Score de Risque de Décès (%)"},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkgray"},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_percent}}
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    # PAGE 6: COMPARAISON DES MODÈLES (Reste inchangé)
    elif page == "📈 Comparaison des Modèles":
        st.header("Comparaison Avancée des Modèles")
        
        results = st.session_state['results']
        y_test = st.session_state['y_test_series']
        
        # Courbes ROC
        st.subheader("📉 Courbes ROC")
        
        fig = go.Figure()
        
        # Ligne aléatoire
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Aléatoire (AUC = 0.50)',
            line=dict(dash='dash', color='gray', width=2)
        ))
        
        # Courbes des modèles
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        for (name, res), color in zip(results.items(), colors):
            fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f"{name} (AUC = {res['ROC-AUC']:.4f})",
                line=dict(color=color, width=3)
            ))
        
        fig.update_layout(
            title='Courbes ROC - Comparaison des Modèles',
            xaxis_title='Taux de Faux Positifs (1 - Spécificité)',
            yaxis_title='Taux de Vrais Positifs (Sensibilité)',
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Comparaison des matrices
        st.subheader("🎯 Matrices de Confusion")
        
        cols = st.columns(3)
        
        for i, (name, res) in enumerate(results.items()):
            with cols[i]:
                cm = confusion_matrix(y_test, res['y_pred'])
                
                fig = px.imshow(cm, 
                                text_auto=True,
                                labels=dict(x="Prédiction", y="Réalité"),
                                x=['Survie', 'Décès'],
                                y=['Survie', 'Décès'],
                                color_continuous_scale='Blues',
                                title=name)
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Tableau récapitulatif
        st.subheader("📋 Tableau Récapitulatif")
        
        results_df = pd.DataFrame({
            name: {
                'Accuracy': f"{res['Accuracy']:.4f}",
                'Precision': f"{res['Precision']:.4f}",
                'Recall': f"{res['Recall']:.4f}",
                'F1-Score': f"{res['F1-Score']:.4f}",
                'ROC-AUC': f"{res['ROC-AUC']:.4f}"
            }
            for name, res in results.items()
        }).T
        
        st.dataframe(results_df, use_container_width=True)
        
        # Recommandation
        best_model = max(results.items(), key=lambda x: x[1]['ROC-AUC'])[0]
        st.success(f"""
        ### 🏆 Modèle Recommandé : {best_model}
        
        Avec un ROC-AUC de **{results[best_model]['ROC-AUC']:.4f}**, ce modèle offre 
        le meilleur compromis entre sensibilité et spécificité pour la prédiction des 
        événements de décès.
        """)
    
    # PAGE 7: INSIGHTS & RECOMMANDATIONS
    elif page == "💡 Insights & Recommandations":
        st.header("Insights & Recommandations Cliniques")
        
        # Section 1: Variables clés
        st.subheader("🔍 Variables Clés Identifiées")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 📊 Top 3 Facteurs Prédictifs
            
            #### 1️⃣ Time (Période de Suivi)
            - **Corrélation** : -0.53 avec mortalité
            - **Impact** : Les patients avec un suivi plus court ont un risque élevé
            - **Importance** : Variable la plus discriminante dans tous les modèles
            
            #### 2️⃣ Ejection Fraction (Fonction Cardiaque)
            - **Seuil critique** : < 30%
            - **Impact** : Distribution nettement différente entre survivants et décédés
            - **Importance** : Indicateur direct de la santé cardiaque
            
            #### 3️⃣ Serum Creatinine (Fonction Rénale)
            - **Seuil d'alerte** : > 1.5 mg/dL
            - **Impact** : Forte corrélation avec la mortalité
            - **Interaction** : Effet amplifié avec l'hypertension (Kidney_Heart_Risk)
            """)
        
        with col2:
            # Boxplots des variables clés
            key_vars = ['time', 'ejection_fraction', 'serum_creatinine']
            
            for var in key_vars:
                fig = px.box(df_model, x='DEATH_EVENT', y=var,
                             color='DEATH_EVENT',
                             color_discrete_sequence=['#4ECDC4', '#FF6B6B'],
                             labels={'DEATH_EVENT': 'Événement'})
                fig.update_layout(height=200, showlegend=False, 
                                  title=dict(text=var, font=dict(size=12)))
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Section 2: Protocole de stratification
        st.subheader("🏥 Protocole de Stratification du Risque") 
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔴 Risque ÉLEVÉ
            
            **Critères** (Approximatifs, basés sur l'analyse) :
            - **Time** < 50 jours
            - **Ejection Fraction** < 30%
            - **Serum Creatinine** > 1.5 mg/dL
            
            **Actions** :
            - 🚨 Surveillance intensive et continue.
            - 📅 Consultations hebdomadaires ou mensuelles.
            - 💊 Optimisation thérapeutique agressive.
            - 🏥 Évaluation pour hospitalisation ou soins spécialisés.
            """)
        
        with col2:
            st.markdown("""
            ### 🟡 Risque MODÉRÉ
            
            **Critères** :
            - **Time** 50-100 jours
            - **Ejection Fraction** 30-40%
            - **Serum Creatinine** 1.0-1.5 mg/dL
            
            **Actions** :
            - ⚠️ Surveillance standard mais rapprochée.
            - 📅 Consultations trimestrielles.
            - 💊 Suivi thérapeutique régulier et ajustements.
            - 📊 Monitoring actif des biomarqueurs clés.
            """)
        
        with col3:
            st.markdown("""
            ### 🟢 Risque FAIBLE
            
            **Critères** :
            - **Time** > 100 jours
            - **Ejection Fraction** > 40%
            - **Serum Creatinine** < 1.0 mg/dL
            
            **Actions** :
            - ✅ Surveillance légère.
            - 📅 Consultations semestrielles ou annuelles.
            - 💊 Traitement de maintien.
            - 🏃 Encouragement aux changements de mode de vie (non-fumeur, activité physique).
            """)
        
        st.divider()
        
        # Section 3: Recommandations
        st.subheader("🎯 Actions Prioritaires")
        
        tab1, tab2, tab3 = st.tabs(["🩺 Cliniques", "📊 Système", "🔬 Recherche"])
        
        with tab1:
            st.markdown("""
            ### 👨‍⚕️ Recommandations Cliniques
            
            * **Focus sur l'EF et la Créatinine :** Ces deux marqueurs sont les plus prédictifs après le temps de suivi. Une surveillance et une intervention rapide sont cruciales pour les patients ayant une **fraction d'éjection (EF) faible** et une **créatinine sérique élevée**.
            * **Gestion des Comorbidités :** L'interaction entre l'**hypertension** et la **créatinine sérique** (nouvelle feature `Kidney_Heart_Risk`) montre un risque accru. Une gestion agressive de l'hypertension est recommandée pour les patients ayant une fonction rénale déjà compromise.
            * **Consultations Précoces :** Les décès se produisant majoritairement au début du suivi (*Time* faible), un protocole d'urgence et des consultations très rapprochées devraient être mis en place dans les 1 à 2 premiers mois pour les patients nouvellement diagnostiqués ou en phase aiguë.
            """)
        
        with tab2:
            st.markdown("""
            ### 💻 Recommandations Système et BI
            
            * **Déploiement du Modèle :** Le modèle **Random Forest** devrait être intégré dans le Système d'Information Hospitalier (SIH) pour fournir un score de risque en temps réel.
            * **Alerte Automatique :** Mise en place d'alertes automatiques pour les patients dont le score de risque prédit dépasse un seuil critique (ex: > 70%), afin de notifier le personnel soignant immédiatement.
            * **Amélioration des Données :** Collecter des données supplémentaires sur les facteurs environnementaux, les antécédents familiaux plus détaillés, ou les résultats d'examens (BNP, troponine) pour affiner la précision du modèle.
            """)
        
        with tab3:
            st.markdown("""
            ### 🔬 Pistes de Recherche ML
            
            * **Optimisation :** Tester l'optimisation des hyperparamètres des modèles (Grid Search/Bayesian Optimization) pour Random Forest et Gradient Boosting afin de maximiser le ROC-AUC ou le F1-Score.
            * **Gestion du Déséquilibre :** Expérimenter des techniques de rééchantillonnage (SMOTE) ou de pondération des classes pour améliorer la prédiction de la classe minoritaire (**Décès**), qui est cruciale.
            * **Modèles d'Interprétabilité :** Utiliser des outils comme **SHAP** ou **LIME** pour obtenir une interprétabilité locale (par patient), au-delà de l'importance globale des variables, renforçant la confiance clinique.
            """)
        
        st.divider()
        
        # Conclusion et appel à l'action
        st.markdown("""
        <div style="text-align: center; padding: 20px; border: 1px solid #1f77b4; border-radius: 10px; margin-top: 30px;">
            <h2>🎉 Analyse Complète !</h2>
            <p>
                Ce tableau de bord offre une vue complète, de l'exploration des données à la prédiction du risque de mortalité, 
                permettant une prise de décision basée sur les données pour améliorer les soins aux patients atteints d'insuffisance cardiaque.
            </p>
            <strong>Passez à l'onglet "🧪 Prédictions Individuelles" pour tester un scénario !</strong>
        </div>
        """, unsafe_allow_html=True)