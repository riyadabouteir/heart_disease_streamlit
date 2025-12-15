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
    page_title="Dashboard BI - Maladies Cardiovasculaires",
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
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🫀 Dashboard Business Intelligence - Maladies Cardiovasculaires</h1>', 
            unsafe_allow_html=True)

# Fonction de chargement des données
@st.cache_data
def load_data():
    """Charge les datasets de maladies cardiovasculaires"""
    try:
        df_uci = pd.read_csv("heart_disease_data.csv")
        df_failure = pd.read_csv("heart_failure_clinical_records_dataset.csv")
        return df_uci, df_failure
    except FileNotFoundError:
        st.error("⚠️ Fichiers de données non trouvés. Veuillez vous assurer que les fichiers CSV sont dans le même répertoire.")
        return None, None

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

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=100)
    st.title("Navigation")
    
    page = st.radio(
        "Sélectionnez une section:",
        ["🏠 Accueil",
         "📊 Exploration des Données (EDA)",
         "🔬 Feature Engineering",
         "🤖 Modélisation & Prédictions",
         "📈 Comparaison des Modèles",
         "💡 Insights & Recommandations"]
    )
    
    st.divider()
    st.markdown("### À propos")
    st.info("""
    **Dashboard BI - Analyse Prédictive**
    
    Ce tableau de bord analyse deux datasets de maladies cardiovasculaires:
    - Heart Disease UCI (606 patients)
    - Heart Failure Records (299 patients)
    """)

# Chargement des données
df_uci, df_failure = load_data()

if df_uci is not None and df_failure is not None:
    
    # PAGE 1: ACCUEIL
    if page == "🏠 Accueil":
        st.header("Bienvenue sur le Dashboard BI")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Dataset 1: Heart Disease UCI")
            st.markdown(f"""
            - **Nombre de patients**: {df_uci.shape[0]}
            - **Nombre de variables**: {df_uci.shape[1]}
            - **Variables cibles**: Présence de maladie cardiaque (target)
            """)
            
            with st.expander("Aperçu des données"):
                st.dataframe(df_uci.head(10), use_container_width=True)
            
            with st.expander("Informations détaillées"):
                st.text(f"Valeurs manquantes:\n{df_uci.isnull().sum()}")
        
        with col2:
            st.subheader("🏥 Dataset 2: Heart Failure Clinical Records")
            st.markdown(f"""
            - **Nombre de patients**: {df_failure.shape[0]}
            - **Nombre de variables**: {df_failure.shape[1]}
            - **Variables cibles**: Événement de décès (DEATH_EVENT)
            """)
            
            with st.expander("Aperçu des données"):
                st.dataframe(df_failure.head(10), use_container_width=True)
            
            with st.expander("Informations détaillées"):
                st.text(f"Valeurs manquantes:\n{df_failure.isnull().sum()}")
        
        st.divider()
        
        # Statistiques descriptives
        st.subheader("📊 Statistiques Descriptives")
        
        tab1, tab2 = st.tabs(["Heart Disease UCI", "Heart Failure Records"])
        
        with tab1:
            st.dataframe(df_uci.describe(), use_container_width=True)
        
        with tab2:
            st.dataframe(df_failure.describe(), use_container_width=True)
    
    # PAGE 2: EXPLORATION DES DONNÉES
    elif page == "📊 Exploration des Données (EDA)":
        st.header("Analyse Exploratoire des Données (EDA)")
        
        dataset_choice = st.selectbox(
            "Choisissez le dataset à explorer:",
            ["Heart Disease UCI", "Heart Failure Clinical Records"]
        )
        
        df_selected = df_uci if dataset_choice == "Heart Disease UCI" else df_failure
        target_col = 'target' if dataset_choice == "Heart Disease UCI" else 'DEATH_EVENT'
        
        # Distribution de la variable cible
        st.subheader("🎯 Distribution de la Variable Cible")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.histogram(df_selected, x=target_col, 
                             color=target_col,
                             title=f"Distribution de {target_col}",
                             labels={target_col: 'Classe'},
                             color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            value_counts = df_selected[target_col].value_counts()
            st.metric("Classe 0", value_counts[0])
            st.metric("Classe 1", value_counts[1])
            st.metric("Ratio", f"{value_counts[1]/value_counts[0]:.2f}")
        
        st.divider()
        
        # Distributions des variables numériques
        st.subheader("📈 Distributions des Variables Numériques")
        
        numeric_cols = df_selected.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        selected_var = st.selectbox("Sélectionnez une variable:", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df_selected, x=selected_var, 
                             color=target_col,
                             marginal="box",
                             title=f"Distribution de {selected_var}",
                             color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df_selected, x=target_col, y=selected_var,
                        color=target_col,
                        title=f"Boxplot de {selected_var} par classe",
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Matrice de corrélation
        st.subheader("🔗 Matrice de Corrélation")
        
        corr_matrix = df_selected.select_dtypes(include=[np.number]).corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       aspect="auto",
                       color_continuous_scale='RdBu_r',
                       title="Matrice de Corrélation")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top corrélations avec la cible
        if target_col in corr_matrix.columns:
            st.subheader("🎯 Top Corrélations avec la Variable Cible")
            
            target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
            
            fig = px.bar(x=target_corr.values, 
                        y=target_corr.index,
                        orientation='h',
                        title=f"Corrélations avec {target_col}",
                        labels={'x': 'Corrélation', 'y': 'Variable'},
                        color=target_corr.values,
                        color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 3: FEATURE ENGINEERING
    elif page == "🔬 Feature Engineering":
        st.header("Feature Engineering")
        
        st.info("""
        Dans cette section, nous appliquons des transformations et créons de nouvelles features 
        pour améliorer la performance des modèles prédictifs.
        """)
        
        # Application du feature engineering sur Heart Failure dataset
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
        
        st.markdown("""
        1. **Age_Group**: Catégorisation de l'âge en groupes (<50, 50-60, 60-70, 70+)
        2. **Kidney_Heart_Risk**: Interaction entre créatinine sérique et hypertension
        3. **Anemia_Diabetes**: Combinaison binaire de l'anémie et du diabète
        """)
        
        # Visualisation de Age_Group
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df_engineered, x='Age_Group', 
                             color='DEATH_EVENT',
                             title="Distribution des Groupes d'Âge par Outcome",
                             barmode='group',
                             color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df_engineered, x='Age_Group', y='Kidney_Heart_Risk',
                        color='DEATH_EVENT',
                        title="Kidney_Heart_Risk par Groupe d'Âge",
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Corrélations des nouvelles features
        st.subheader("🔗 Impact des Nouvelles Features")
        
        new_features = ['Kidney_Heart_Risk', 'Anemia_Diabetes']
        corr_new = df_engineered[new_features + ['DEATH_EVENT']].corr()
        
        fig = px.imshow(corr_new, 
                       text_auto='.3f',
                       aspect="auto",
                       color_continuous_scale='RdBu_r',
                       title="Corrélations des Nouvelles Features avec DEATH_EVENT")
        st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 4: MODÉLISATION & PRÉDICTIONS
    elif page == "🤖 Modélisation & Prédictions":
        st.header("Modélisation & Prédictions")
        
        st.info("""
        Cette section entraîne et évalue trois modèles de classification:
        - Régression Logistique
        - Random Forest
        - Gradient Boosting
        """)
        
        # Préparation des données pour Heart Failure dataset
        df_model = feature_engineering(df_failure)
        
        # Sélection des features numériques pour la modélisation
        features_to_use = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                          'ejection_fraction', 'high_blood_pressure', 'platelets',
                          'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
                          'Kidney_Heart_Risk', 'Anemia_Diabetes']
        
        X = df_model[features_to_use]
        y = df_model['DEATH_EVENT']
        
        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entraînement des modèles
        with st.spinner("Entraînement des modèles en cours..."):
            models, results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        st.success("✅ Modèles entraînés avec succès!")
        
        # Affichage des performances
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
        
        st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'), 
                    use_container_width=True)
        
        # Visualisation des métriques
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=list(results.keys()),
                    y=[results[model][metric] for model in results.keys()]
                ))
            
            fig.update_layout(
                title="Comparaison des Métriques par Modèle",
                xaxis_title="Modèle",
                yaxis_title="Score",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature Importance pour Random Forest
            if 'Random Forest' in models:
                rf_model = models['Random Forest']
                feature_importance = pd.DataFrame({
                    'feature': features_to_use,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig = px.bar(feature_importance.head(10), 
                           x='importance', 
                           y='feature',
                           orientation='h',
                           title="Top 10 Features Importantes (Random Forest)",
                           color='importance',
                           color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Matrice de confusion pour le meilleur modèle
        st.subheader("🎯 Matrice de Confusion (Random Forest)")
        
        best_model = 'Random Forest'
        cm = confusion_matrix(y_test, results[best_model]['y_pred'])
        
        fig = px.imshow(cm, 
                       text_auto=True,
                       labels=dict(x="Prédiction", y="Réalité", color="Count"),
                       x=['Survie', 'Décès'],
                       y=['Survie', 'Décès'],
                       color_continuous_scale='Blues',
                       title=f"Matrice de Confusion - {best_model}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Rapport de classification
        with st.expander("📄 Rapport de Classification Détaillé"):
            report = classification_report(y_test, results[best_model]['y_pred'], 
                                          target_names=['Survie', 'Décès'])
            st.text(report)
    
    # PAGE 5: COMPARAISON DES MODÈLES
    elif page == "📈 Comparaison des Modèles":
        st.header("Comparaison Avancée des Modèles")
        
        # Préparation des données
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
        
        with st.spinner("Entraînement des modèles..."):
            models, results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Courbes ROC
        st.subheader("📉 Courbes ROC")
        
        fig = go.Figure()
        
        # Courbe aléatoire
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Aléatoire (AUC = 0.50)',
            line=dict(dash='dash', color='gray')
        ))
        
        # Courbes pour chaque modèle
        colors = ['blue', 'green', 'red']
        for (name, res), color in zip(results.items(), colors):
            fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f"{name} (AUC = {res['ROC-AUC']:.4f})",
                line=dict(color=color, width=2)
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
        
        # Comparaison radar des métriques
        st.subheader("🕸️ Diagramme Radar des Performances")
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig = go.Figure()
        
        for name, res in results.items():
            values = [res[metric] for metric in metrics]
            values.append(values[0])  # Pour fermer le radar
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=name
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Comparaison Multi-Métriques des Modèles",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Comparaison des matrices de confusion
        st.subheader("🎯 Comparaison des Matrices de Confusion")
        
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
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Tableau récapitulatif
        st.subheader("📋 Tableau Récapitulatif des Performances")
        
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
    
    # PAGE 6: INSIGHTS & RECOMMANDATIONS
    elif page == "💡 Insights & Recommandations":
        st.header("Insights & Recommandations Cliniques")
        
        # Préparation des données pour l'analyse
        df_model = feature_engineering(df_failure)
        
        # Section 1: Insights de l'EDA
        st.subheader("🔍 Insights de l'Analyse Exploratoire")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 Variables Clés Identifiées
            
            **1. Time (Période de Suivi)**
            - Corrélation la plus forte avec la mortalité (-0.53)
            - Les patients avec un suivi plus court ont un risque plus élevé
            - Variable la plus importante dans tous les modèles
            
            **2. Ejection Fraction**
            - Deuxième facteur le plus important
            - Distribution nettement différente entre survivants et décédés
            - Indicateur critique de la fonction cardiaque
            
            **3. Serum Creatinine**
            - Marqueur rénal crucial
            - Forte corrélation avec la mortalité
            - Interaction significative avec l'hypertension
            """)
        
        with col2:
            # Visualisation des variables clés
            key_vars = ['time', 'ejection_fraction', 'serum_creatinine']
            
            for var in key_vars:
                fig = px.box(df_model, x='DEATH_EVENT', y=var,
                           color='DEATH_EVENT',
                           title=f"Distribution de {var} par Outcome",
                           color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Section 2: Performance des Modèles
        st.subheader("🤖 Insights de la Modélisation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Comparaison des Modèles
            
            **Random Forest - Modèle Recommandé**
            - **AUC**: 0.8779 (le plus élevé)
            - **Precision**: 0.7895 (excellent taux de prédictions positives correctes)
            - **Recall**: 0.6250 (bon équilibre de détection)
            - **Avantages**:
              - Meilleur compromis entre précision et rappel
              - Robuste aux outliers
              - Interprétable via feature importance
            
            **Logistic Regression**
            - **AUC**: 0.8766 (très bon)
            - **Recall**: 0.7083 (meilleur taux de détection)
            - **Usage**: Quand la détection maximale est prioritaire
            
            **Gradient Boosting**
            - **AUC**: 0.8538 (bon)
            - **Precision**: 0.7143
            - **Usage**: Alternative solide pour validation croisée
            """)
        
        with col2:
            st.info("""
            ### 💊 Seuil de Décision
            
            **Recommandation**: Ajuster le seuil selon le contexte clinique
            
            - **Dépistage**: Seuil bas (0.3-0.4) pour maximiser la détection
            - **Traitement**: Seuil standard (0.5) pour équilibrer
            - **Ressources limitées**: Seuil élevé (0.6-0.7) pour cibler les cas critiques
            """)
        
        st.divider()
        
        # Section 3: Recommandations Cliniques
        st.subheader("🏥 Recommandations Cliniques")
        
        st.markdown("""
        ### 📋 Protocole de Stratification du Risque
        
        #### 🔴 Patients à Haut Risque (Surveillance Intensive)
        - **Time** < 50 jours de suivi
        - **Ejection Fraction** < 30%
        - **Serum Creatinine** > 1.5 mg/dL
        - **Action**: Consultations mensuelles, monitoring cardiaque rapproché
        
        #### 🟡 Patients à Risque Modéré (Surveillance Standard)
        - **Time** 50-100 jours
        - **Ejection Fraction** 30-40%
        - **Serum Creatinine** 1.0-1.5 mg/dL
        - **Action**: Consultations trimestrielles, évaluation régulière
        
        #### 🟢 Patients à Faible Risque (Surveillance Légère)
        - **Time** > 100 jours
        - **Ejection Fraction** > 40%
        - **Serum Creatinine** < 1.0 mg/dL
        - **Action**: Consultations semestrielles, suivi standard
        
        ---
        
        ### 🎯 Actions Prioritaires
        
        1. **Suivi Régulier et Prolongé**
           - Importance critique démontrée par la variable 'time'
           - Mise en place de rappels automatiques pour les consultations
           - Télémédecine pour les patients à mobilité réduite
        
        2. **Monitoring de la Fonction Cardiaque**
           - Échocardiographie régulière pour suivre l'ejection fraction
           - Alertes automatiques si EF < 30%
           - Ajustement thérapeutique proactif
        
        3. **Surveillance de la Fonction Rénale**
           - Dosage régulier de la créatinine sérique
           - Attention particulière aux patients hypertendus
           - Feature 'Kidney_Heart_Risk' validée par l'analyse
        
        4. **Approche Multifactorielle**
           - Considérer l'âge, le diabète et l'anémie simultanément
           - Feature 'Anemia_Diabetes' montre une interaction significative
           - Traitement holistique plutôt que symptomatique
        
        ---
        
        ### 📊 Implémentation du Système de Scoring
        
        **Score de Risque Calculé**:
        - Time (0-40 points): Plus court = Plus de points
        - Ejection Fraction (0-30 points): Plus bas = Plus de points
        - Serum Creatinine (0-20 points): Plus élevé = Plus de points
        - Kidney_Heart_Risk (0-10 points): Interaction significative
        
        **Interprétation**:
        - **0-30 points**: Risque Faible ✅
        - **31-60 points**: Risque Modéré ⚠️
        - **61-100 points**: Risque Élevé 🚨
        """)
        
        st.divider()
        
        # Section 4: Limitations et Perspectives
        st.subheader("⚠️ Limitations et Perspectives Futures")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.warning("""
            ### 📉 Limitations de l'Étude
            
            - **Taille de l'échantillon**: 299 patients (Heart Failure dataset)
            - **Déséquilibre des classes**: Plus de survivants que de décès
            - **Variables manquantes**: Certains facteurs de risque non disponibles
            - **Validation externe**: Nécessité de valider sur d'autres populations
            - **Biais de sélection**: Patients d'un seul centre médical
            """)
        
        with col2:
            st.success("""
            ### 🚀 Perspectives d'Amélioration
            
            - **Données supplémentaires**: Intégrer plus de patients et centres
            - **Features additionnelles**: ECG, imagerie médicale, génétique
            - **Deep Learning**: Réseaux de neurones pour patterns complexes
            - **Monitoring temps réel**: Wearables et IoT médical
            - **Médecine personnalisée**: Modèles adaptés par sous-populations
            """)
        
        st.divider()
        
        # Section 5: Conclusion
        st.subheader("✅ Conclusion")
        
        st.success("""
        ### 🎯 Points Clés à Retenir
        
        1. **Modèle Optimal**: Random Forest avec AUC de 0.8779 offre le meilleur compromis
        
        2. **Facteurs Critiques**: 
           - Durée du suivi (time)
           - Fonction cardiaque (ejection_fraction)
           - Fonction rénale (serum_creatinine)
        
        3. **Impact Clinique**: 
           - Stratification efficace des risques
           - Priorisation des ressources médicales
           - Amélioration de la prise en charge
        
        4. **Implémentation**: 
           - Intégration possible dans les systèmes hospitaliers
           - Aide à la décision pour les cliniciens
           - Alertes automatisées pour les cas critiques
        
        ---
        
        **📌 Note Importante**: Ces modèles sont des outils d'aide à la décision et ne remplacent 
        pas le jugement clinique d'un professionnel de santé. Toute décision thérapeutique doit 
        être prise par un médecin qualifié en considérant l'ensemble du contexte médical du patient.
        """)

else:
    st.error("""
    ⚠️ Impossible de charger les données. 
    
    Veuillez vous assurer que les fichiers suivants sont présents:
    - heart_disease_data.csv
    - heart_failure_clinical_records_dataset.csv
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p>Dashboard BI - Analyse Prédictive des Maladies Cardiovasculaires</p>
    <p>Développé avec Streamlit | Données: Heart Disease UCI & Heart Failure Clinical Records</p>
</div>
""", unsafe_allow_html=True)