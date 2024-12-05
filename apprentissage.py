
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Data visualisation avec streamlit")
st.subheader(body="Auteur: TEAM ALPHA")
st.markdown("***Cette applicaton affiche different type de graphique sur le thème Emploi et Insertion Professionnelle en Afrique***")


#data=pd.read_excel("./personna.xlsx",sheet_name="Sheet1")
# st.write(data.head())


simul=np.random.normal(size=1000)
st.line_chart(simul)

# Diagramme à barres

bar_data= pd.DataFrame(
    [100,10,15,69],
    ["A","B","C","D"]
)

st.bar_chart(bar_data)
