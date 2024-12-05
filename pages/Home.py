
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
import os
import warnings
import datetime


# Configuration de la page Streamlit
st.set_page_config(page_title="Visualisation de Données", page_icon=":bar_chart:", layout="wide")

# Styles CSS inline (à améliorer avec un fichier CSS externe pour de plus grandes applications)
st.markdown(
    """
    <style>
    body {
        font-family: sans-serif;
    }
    .title {
        text-align: center;
        color: #306609; /* Couleur du titre */
    }
    .subtitle {
        text-align: center;
        color: #6699CC; /* Couleur du sous-titre */
    }
    .section-header {
        background-color: #1864B8; /* Couleur de fond des sections */
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# Titre et sous-titre
st.markdown('<h1 class="title">Bienvenue sur notre plateforme de visualisation de données</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subtitle">Explorez nos données et découvrez des informations précieuses.</h3>', unsafe_allow_html=True)


# Section d'introduction
st.markdown("""
<div>
Cette application vous permet de visualiser des données de manière interactive.  
Vous trouverez ci-dessous des exemples de visualisations.
</div>
""", unsafe_allow_html=True)


# Données d'exemple
data = pd.DataFrame({
    'Catégorie': ['A', 'B', 'C', 'A', 'B', 'C'],
    'Valeur': [10, 15, 12, 18, 20, 25]
})

# Visualisations (Matplotlib et Plotly - inchangées)
st.markdown('<div class="section-header"><h3>Exemple de visualisation avec Matplotlib</h3></div>', unsafe_allow_html=True)
fig, ax = plt.subplots()
ax.bar(data['Catégorie'], data['Valeur'])
ax.set_xlabel("Catégorie")
ax.set_ylabel("Valeur")
ax.set_title("Diagramme en barres")
st.pyplot(fig)

st.markdown('<div class="section-header"><h3>Exemple de visualisation interactive avec Plotly</h3></div>', unsafe_allow_html=True)
fig_plotly = px.pie(data, values='Valeur', names='Catégorie', title='Diagramme circulaire')
st.plotly_chart(fig_plotly)


# Section de contact
st.markdown("---")
st.markdown('<div class="section-header"><h3>Contactez-nous</h3></div>', unsafe_allow_html=True)
st.write("Pour toute question ou suggestion, n'hésitez pas à nous contacter.")
st.write("Email: team_alpha@gmail.com")
