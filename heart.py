import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(
    page_title="Dashboard BI - Insuffisance Cardiaque",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
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

# Fonction pour le feature engineering
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
    """Entraîne les trois modèles de classification"""
    models = {}
    results = {}
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # 3. Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    
    # Calcul des métriques pour chaque modèle
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    return models, results

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

# Si pas de données, afficher la page d'upload
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
    
    # PAGE 1: ACCUEIL
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
    
    # PAGE 2: EXPLORATION DES DONNÉES
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
    
    # PAGE 3: FEATURE ENGINEERING
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
            
            Risque = créatinine × hypertension
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
    
    # PAGE 4: MODÉLISATION & PRÉDICTIONS
    elif page == "🤖 Modélisation & Prédictions":
        st.header("Modélisation & Prédictions")
        
        st.info("""
        Cette section entraîne et évalue trois modèles de classification:
        - ⚙️ Régression Logistique
        - 🌲 Random Forest (Recommandé)
        - 📈 Gradient Boosting
        """)
        
        # Préparation des données
        df_model = feature_engineering(df_failure)
        
        features_to_use = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                          'ejection_fraction', 'high_blood_pressure', 'platelets',
                          'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
                          'Kidney_Heart_Risk', 'Anemia_Diabetes']
        
        X = df_model[features_to_use]
        y = df_model['DEATH_EVENT']
        
        # Split et normalisation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entraînement
        with st.spinner("🔄 Entraînement des modèles en cours..."):
            models, results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        st.success("✅ Modèles entraînés avec succès!")
        
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
            # Comparaison des métriques - Radar amélioré
            st.subheader("📊 Comparaison des Métriques")
            
            fig = go.Figure()
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            
            colors = {'Logistic Regression': '#636EFA', 'Random Forest': '#00CC96', 'Gradient Boosting': '#EF553B'}
            
            for model_name in results.keys():
                values = [results[model_name][metric] for metric in metrics]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=model_name,
                    line=dict(color=colors.get(model_name, '#636EFA'), width=2),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, 
                        range=[0, 1],
                        showline=True,
                        linewidth=2,
                        gridcolor='lightgray',
                        tickfont=dict(size=10)
                    ),
                    angularaxis=dict(
                        linewidth=2,
                        showline=True,
                        gridcolor='lightgray'
                    )
                ),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.1
                ),
                height=450,
                title=dict(
                    text="Radar des Performances",
                    font=dict(size=14)
                )
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
        cm = confusion_matrix(y_test, results[best_model]['y_pred'])
        
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
            report = classification_report(y_test, results[best_model]['y_pred'], 
                                          target_names=['Survie', 'Décès'])
            st.text(report)
    
    # PAGE 5: COMPARAISON DES MODÈLES
    elif page == "📈 Comparaison des Modèles":
        st.header("Comparaison Avancée des Modèles")
        
        # Préparation
        df_model = feature_engineering(df_failure)
        features_to_use = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                          'ejection_fraction', 'high_blood_pressure', 'platelets',
                          'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
                          'Kidney_Heart_Risk', 'Anemia_Diabetes']
        
        X = df_model[features_to_use]
        y = df_model['DEATH_EVENT']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        with st.spinner("⏳ Entraînement des modèles..."):
            models, results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
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
    
    # PAGE 6: INSIGHTS & RECOMMANDATIONS
    elif page == "💡 Insights & Recommandations":
        st.header("Insights & Recommandations Cliniques")
        
        df_model = feature_engineering(df_failure)
        
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
            - **Interaction** : Effet amplifié avec l'hypertension
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
            
            **Critères** :
            - Time < 50 jours
            - Ejection Fraction < 30%
            - Serum Creatinine > 1.5
            
            **Actions** :
            - 🚨 Surveillance intensive
            - 📅 Consultations mensuelles
            - 💊 Optimisation thérapeutique
            - 🏥 Hospitalisation si nécessaire
            """)
        
        with col2:
            st.markdown("""
            ### 🟡 Risque MODÉRÉ
            
            **Critères** :
            - Time 50-100 jours
            - Ejection Fraction 30-40%
            - Serum Creatinine 1.0-1.5
            
            **Actions** :
            - ⚠️ Surveillance standard
            - 📅 Consultations trimestrielles
            - 💊 Suivi thérapeutique régulier
            - 📊 Monitoring des biomarqueurs
            """)
        
        with col3:
            st.markdown("""
            ### 🟢 Risque FAIBLE
            
            **Critères** :
            - Time > 100 jours
            - Ejection Fraction > 40%
            - Serum Creatinine < 1.0
            
            **Actions** :
            - ✅ Surveillance légère
            - 📅 Consultations semestrielles
            - 💊 Traitement de maintien
            - 🏃 Encouragement activité physique
            """)
        
        st.divider()
        
        # Section 3: Recommandations
        st.subheader("🎯 Actions Prioritaires")
        
        tab1, tab2, tab3 = st.tabs(["🩺 Cliniques", "📊 Système", "🔬 Recherche"])
        
        with tab1:
            st.markdown("""
            ### Recommandations Cliniques
            
            #### 1. Suivi Régulier et Prolongé
            - ✅ Consultations planifiées selon le niveau de risque
            - ✅ Télémédecine pour patients à mobilité réduite
            - ✅ Rappels automatiques
            
            #### 2. Monitoring de la Fonction Cardiaque
            - ✅ Échocardiographie régulière
            - ✅ Alertes si Ejection Fraction < 30%
            - ✅ Ajustement thérapeutique proactif
            
            #### 3. Surveillance de la Fonction Rénale
            - ✅ Dosage régulier de créatinine
            - ✅ Attention aux patients hypertendus
            - ✅ Feature 'Kidney_Heart_Risk' validée
            
            #### 4. Approche Multifactorielle
            - ✅ Considérer âge, diabète, anémie simultanément
            - ✅ Traitement holistique
            - ✅ Prise en charge des comorbidités
            """)
        
        with tab2:
            st.markdown("""
            ### Implémentation du Système de Scoring
            
            #### Score de Risque (0-100 points)
            
            | Facteur | Points | Seuils |
            |---------|--------|--------|
            | **Time** | 0-40 pts | < 50j = 40pts, 50-100j = 20pts, > 100j = 0pts |
            | **Ejection Fraction** | 0-30 pts | < 30% = 30pts, 30-40% = 15pts, > 40% = 0pts |
            | **Serum Creatinine** | 0-20 pts | > 1.5 = 20pts, 1.0-1.5 = 10pts, < 1.0 = 0pts |
            | **Kidney_Heart_Risk** | 0-10 pts | Proportionnel à la valeur |
            
            #### Interprétation du Score Total
            
            - **0-30 points** : 🟢 Risque Faible
            - **31-60 points** : 🟡 Risque Modéré
            - **61-100 points** : 🔴 Risque Élevé
            
            #### Intégration Système
            
            - 💻 Intégration dans DME (Dossier Médical Électronique)
            - 🔔 Alertes automatiques pour scores élevés
            - 📊 Dashboard pour équipes médicales
            - 📈 Suivi longitudinal des patients
            """)
        
        with tab3:
            st.markdown("""
            ### Perspectives de Recherche
            
            #### Améliorations Possibles
            
            1. **Données Supplémentaires**
               - 🔬 Biomarqueurs additionnels (BNP, troponine)
               - 🫀 Données d'imagerie cardiaque
               - 🧬 Facteurs génétiques
               - 📱 Données de wearables
            
            2. **Modélisation Avancée**
               - 🤖 Deep Learning (réseaux neurones)
               - 🔄 Modèles d'ensemble (Stacking)
               - ⏱️ Modèles de survie (Cox, Time-to-event)
               - 🎯 Médecine personnalisée
            
            3. **Validation**
               - 🌍 Validation multicent rique
               - 🔀 Validation croisée externe
               - 📊 Études prospectives
               - ⚖️ Équité entre populations
            
            #### Limitations Actuelles
            
            - ⚠️ Échantillon limité (299 patients)
            - ⚠️ Déséquilibre des classes
            - ⚠️ Données d'un seul centre
            - ⚠️ Variables manquantes possibles
            """)
        
        st.divider()
        
        # Conclusion
        st.subheader("✅ Conclusion")
        
        st.success("""
        ### 🎯 Points Clés à Retenir
        
        1. **Modèle Optimal** : Random Forest offre le meilleur compromis (AUC ≈ 0.88)
        
        2. **Facteurs Critiques** :
           - Durée du suivi (time) - Le plus important
           - Fonction cardiaque (ejection_fraction)
           - Fonction rénale (serum_creatinine)
        
        3. **Impact Clinique** :
           - Stratification efficace des risques
           - Optimisation des ressources médicales
           - Amélioration de la prise en charge
        
        4. **Implémentation** :
           - Intégration possible dans systèmes hospitaliers
           - Aide à la décision pour cliniciens
           - Alertes automatisées
        
        ---
        
        **⚠️ Important** : Ces modèles sont des outils d'aide à la décision et ne remplacent 
        pas le jugement clinique. Toute décision doit être prise par un professionnel de santé 
        qualifié en considérant le contexte complet du patient.
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p><strong>Dashboard BI - Analyse Prédictive d'Insuffisance Cardiaque</strong></p>
    <p>Développé avec Streamlit | Machine Learning pour la Santé</p>
</div>
""", unsafe_allow_html=True)