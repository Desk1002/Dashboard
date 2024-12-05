
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
import os
import warnings
import datetime
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.offline as py
import plotly.tools as tls
import plotly
import time
import statsmodels.formula.api as smf

#==========================================================
#==== Base et Traitement ==================================
#==========================================================

#

#def load_data(excel_path, shapefile_path):
    #Charge les données Excel et Shapefile, avec gestion des erreurs.
#    st.write("Chargement des données...")
#    start_time = time.time()
#    try:
#        data = pd.read_excel(excel_path)
#        geo_Africa_df = gpd.read_file(shapefile_path)
#        end_time = time.time()
 #       st.success(f"Données chargées en {end_time - start_time:.2f} secondes.")
 #       return data, geo_Africa_df
 #   except FileNotFoundError as e:
#        st.error(f"Erreur : Fichier introuvable - {e}")
#        return None, None
#    except Exception as e:
#        st.exception(f"Une erreur s'est produite : {e}")
#        return None, None

 
#excel_path = "Indicateur.xlsx"
#shapefile_path = "GeoData2023.shp"

#data, geo_Africa_df = load_data(excel_path, shapefile_path)
@st.cache_data(persist="disk") # Persist pour garder le cache entre les sessions
def load_data(excel_path):
    data=pd.read_excel(excel_path)
    return data
excel_path = "Indicateur.xlsx"
#shapefile_path = "GeoData2023.shp"

data = load_data(excel_path)
st.sidebar.image("Logo.png") # Remplacez par le chemin de votre logo
st.sidebar.title("Membre de l'Equipe")
df_model = pd.read_excel( "baseFINALE_panel.xlsx")
data_to_grph=data[data["time"]==2023]
#==========================================================
#==== FONCTION DE VISUALISATION ===========================
#==========================================================

def make_multy_bar(data,value,group):
    df_dg=data[data['indicator']==indicateur]
    df_dg=df_dg.groupby(by=[group,"Region"])["Valeur"].sum().reset_index()
    fig = px.bar(df_dg, x=group, y=value, color="Region")
    st.plotly_chart(fig) 

def make_line(data,value,group,titre=""):
    #df_dg=data[(data['indicator']==indicateur)]
    df_dg=data[(data['indicator']==indicateur)]
    #a_tes=list(df_dg["Gender"].unique())
    #if len(a_tes)!=0:
        #df_dg=data[(data['indicator']==indicateur) & (data["Gender"] == gender)]
    df_dg=df_dg.groupby(by=[group,"time"])["Valeur"].sum().reset_index()
    fig = px.line(df_dg, x="time", y=value, color=group, markers=True,title=titre)
    fig.update_layout(
        title_font=dict(size=16, color="white", family="Arial"), #Titre blanc
        xaxis_title="Temps",
        yaxis_title=value,
        legend=dict(font=dict(size=12)),
        font=dict(size=12)
    )
    st.plotly_chart(fig) 

def make_chloropleth(input_geodf, indicator, value, width=800, height=600):

    #df_grph = input_geodf[(input_geodf["indicator"] == indicator)&(input_geodf["Gender"] == gender)&(input_geodf["Critere"] == crit)].copy() #Copie pour éviter les modifications in situ
    df_grph=input_geodf[(input_geodf["indicator"] == indicator) ]
    fig = px.choropleth_mapbox(
        df_grph,
        geojson=df_grph.geometry,
        locations=df_grph.index,  # ou l'ID de vos polygones
        color=value,  # Colonne contenant les valeurs à représenter
        hover_name="country", #Affichage du nom du pays au survol (si la colonne existe)
        hover_data=[value], #Affichage de la valeur au survol
        color_continuous_scale="Blues",  # Palette de couleurs plus esthétique
        range_color=[df_grph[value].min(), df_grph[value].max()], #Ajustement automatique de l'échelle de couleur
        width=width,
        height=height,
        zoom=1.5,  # Niveau de zoom (ajuster selon votre zone géographique)
        opacity=1,  # Opacité légèrement augmentée pour une meilleure visibilité
        mapbox_style="carto-darkmatter",  # Style de carte plus clair et lisible
        title=f"Carte Choroplèthe de {indicator}", #Titre plus informatif
        labels={value: indicator} #Label plus explicite
    )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 20, "b": 0},  # Marge supérieure augmentée pour le titre
        template='plotly_dark', #Template plus clair
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(size=12), #Taille de police augmentée
        title_font=dict(size=16, color="white", family="Arial"), #Style du titre
        legend=dict(font=dict(size=12)) #Taille de la légende augmentée
    )

    st.plotly_chart(fig)

def make_combined_map(input_geodf, pib_column, population_column, width=800, height=600):
    """Crée une carte combinant choroplèthe et points."""
    input_geodf=input_geodf[input_geodf['indicator']==indicateur]
    fig = go.Figure()

    # Choroplèthe pour le PIB
    fig.add_trace(go.Choroplethmapbox(
        geojson=input_geodf.geometry,
        locations=input_geodf.index,
        z=input_geodf[pib_column],
        colorscale="Blues",
        zmin=input_geodf[pib_column].min(),
        zmax=input_geodf[pib_column].max(),
        width=width,
        height=height,
        marker_opacity=1,
        name="PIB",
        hoverinfo='text',
        hovertemplate="%{hovertext}<extra></extra>",
    ))
    input_geodf['hovertext_pib'] = input_geodf['country'] + '<br>PIB: ' + input_geodf[pib_column].astype(str)


    # Points pour la population
    fig.add_trace(go.Scattermapbox(
        lat=input_geodf.geometry.centroid.y,  # Latitude des centroïdes
        lon=input_geodf.geometry.centroid.x,  # Longitude des centroïdes
        mode='markers',
        marker=dict(
            size=input_geodf[population_column] / 100000,  # Taille proportionnelle à la population (ajuster le diviseur selon l'échelle)
            color='red',
            opacity=1
        ),
        name="Population",
        hoverinfo='text',
        hovertemplate="%{hovertext}<extra></extra>",
    ))
    input_geodf['hovertext_pop'] = input_geodf['country'] + '<br>Population: ' + input_geodf[population_column].astype(str)


    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            zoom=1.5,
            center=dict(lat=0, lon=0)
        ),
        margin={"r": 0, "t": 0, "l": 20, "b": 0},
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(size=12),
        title_font=dict(size=16, color="white", family="Arial"),
        legend=dict(font=dict(size=12)),
        title= indicateur + " and Population (2017)"
    )

    st.plotly_chart(fig)

def make_histogram(input_geodf, indicator, width=800, height=600):
    
    #df = input_geodf[(input_geodf["indicator"] == indicator)&(input_geodf["Gender"] == gender)&(input_geodf["Critere"] == crit)].copy() #Copie pour éviter les modifications in situ
    df = input_geodf[(input_geodf["indicator"] == indicator)].copy() #Copie pour éviter les modifications in situ
    #df = input_geodf[input_geodf['indicator'] == indicator].copy()  # Copie pour éviter les modifications in situ
    df.sort_values("Valeur", inplace=True) #Tri sur place pour plus d'efficacité

    # Création de la couleur des barres en fonction de la valeur
    df['color'] = df['Valeur'].apply(lambda x: 'red' if x < 0 else 'blue')

    fig = px.histogram(
        df,
        x="Region",
        y="Valeur",
        color='color',  # Couleur des barres basée sur la colonne 'color'
        color_discrete_sequence=['blue', 'red'], #Séquence pour gérer les cas où il n'y a que des valeurs positives ou négatives
        labels={'Valeur': 'Valeur de l\'indicateur', 'Region': 'Région'}, #Labels plus explicites
        title=f'{indicator}', #Titre plus informatif
        width=width,
        height=height
    )

    fig.update_layout(
        barmode='group',  # Regroupement des barres pour une meilleure lisibilité
        xaxis_title='Valeur de l\'indicateur', #Titre de l'axe X
        yaxis_title='Région', #Titre de l'axe Y
        xaxis = dict(showgrid=True, gridwidth=1, gridcolor='lightgray'), #Ajout d'une grille pour une meilleure lisibilité
        yaxis = dict(showgrid=True, gridwidth=1, gridcolor='lightgray') #Ajout d'une grille pour une meilleure lisibilité

    )
    st.plotly_chart(fig)

def display_single_metric_advanced(label, value, delta, unit="", caption="", color_scheme="blue"):
    """Affiche une seule métrique avec un style avancé et personnalisable."""

    color = {
        "blue": {"bg": "#e6f2ff", "text": "#336699", "delta_pos": "#007bff", "delta_neg": "#dc3545"},
        "green": {"bg": "#e6ffe6", "text": "#28a745", "delta_pos": "#28a745", "delta_neg": "#dc3545"},
        "red": {"bg": "#ffe6e6", "text": "#dc3545", "delta_pos": "#28a745", "delta_neg": "#dc3545"},
    }.get(color_scheme, {"bg": "#f0f0f0", "text": "#333", "delta_pos": "#28a745", "delta_neg": "#dc3545"})

    delta_color = color["delta_pos"] if delta >= 0 else color["delta_neg"]

    st.markdown(
        f"""
        <div style="background-color: {color['bg']}; padding: 3px; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); text-align: center;">
            <h3 style="color: {color['text']}; margin-bottom: 3px;">{label}</h3>
            <div style="font-size: 2.5em; font-weight: bold; color: {color['text']};">{value} {unit}</div>
            <div style="font-size: 1.5em; color: {delta_color};">{'▲' if delta >= 0 else '▼'} {abs(delta)}</div>
            <p style="font-size: 1em; color: {color['text']};">{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def make_bar_polar_chart(data, names, values, title="Diagramme Polaire en Barres", width=800, height=600, colors=None):
    df_dg=data[data['indicator']==indicateur]
    data=df_dg.groupby(by=["Region"])["Valeur"].sum().reset_index()
    
    if isinstance(data, pd.DataFrame):
        fig = px.bar_polar(data, r=values, theta=names, title=title, width=width, height=height, color=names)
    else:
        fig = px.bar_polar(r=values, theta=names, title=title, width=width, height=height, color=names)

    if colors:
        fig.update_traces(marker=dict(color=colors, line=dict(color='#000000', width=1)))
    else:
        fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))

    fig.update_layout(
        title_font=dict(size=16, color="white", family="Arial"),
        legend=dict(font=dict(size=12)),
        plot_bgcolor='rgba(0,0,0,0)',  # Arrière-plan transparent
        paper_bgcolor='rgba(0,0,0,0)',  # Arrière-plan transparent
        margin=dict(l=50, r=50, b=50, t=80, pad=4)
    )
    st.plotly_chart(fig)

def afficher_profil_sidebar(photo, nom, prenom, email, telephone):
   
    st.sidebar.image(photo, width=100) # Ajustez la largeur selon vos besoins
    st.sidebar.write(f"Nom : {nom}")
    st.sidebar.write(f"Prénom : {prenom}")
    st.sidebar.write(f"Email : {email}")
    st.sidebar.write(f"Téléphone : {telephone}")

afficher_profil_sidebar("Landry.jpg", "KENGNE", "Bienvenu Landry", "landrykengne99@gmail.com", "+237 6 98 28 05 37")

afficher_profil_sidebar("Merveil.jpg", "NOULAYE", "Merveille", "noulayemerveille@gmail.com", "+237 6 77 39 32 86")

afficher_profil_sidebar("Deselie.jpg", "DONFACK", "Deselie", "deseliengoufack@gmail.com", "+237 6 98 16 03 61")

afficher_profil_sidebar("Rinel.jpg", "TAGNE", "Rinel", "cezangue@gmail.com", "+237 6 73 83 11 57")

afficher_profil_sidebar("Anaba.jpg", "ANABA", "Rodrigue", "student.rodrigue.anaba@issea-cemac.org", "+237 6 96 26 90 77")

def chloro_cart(df,indicator,value):
    fig = px.choropleth(
        df[df['indicator']==indicator].sort_values(by='time'),
        locations='ISO_A3',
        color=value,
        hover_name='country',
        color_continuous_scale=px.colors.sequential.Plasma,
        title=indicator,
        labels={value: "Value"},
        
    )
    fig.update_geos(
        projection_type="mercator",
        showframe=False,
        showcoastlines=False,
        fitbounds="locations",  # Automatically fits the map to the data
        lataxis=dict(range=[-35, 40]),  # Latitude range for Africa
        lonaxis=dict(range=[-20, 55]),  # Longitude range for Africa
    )
    # Update layout to remove margins
    fig.update_layout(
        title=dict(x=0.5),  # Center the title
        margin=dict(l=0, r=0, t=30, b=0),  # Remove margins
        coloraxis_colorbar=dict(title="Value"),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        
    )
    st.plotly_chart(fig)


st.title("EMPLOI ET INSERTION PROFESSIONNELLE EN AFRIQUE")
st.subheader(body="Concepteur: GROUPE ALPHA")
st.markdown("***Cette applicaton affiche different type de graphique***")

tables = st.tabs(["Données", "Description des différentes bases", "Tableau de Bord","Modélisation","Interprétations"])
with tables[0]:
    st.text("Mettre Les différentes Base de Donnée ici")
    #if data is not None and geo_Africa_df is not None:
        # Traitement et affichage des données
        #st.write("Données Excel :")
        #st.dataframe(data)
        #st.write("Données Shapefile :")
        #st.map(geo_Africa_df) # ou autre affichage de geo_Africa_df
    #else:
        #st.warning("Le chargement des données a échoué. Veuillez vérifier les chemins de fichiers.")
with tables[1]:
    st.text("Décrire les différentes Bases ")
        # ...votre code pour afficher les données...
        
        
#==============================================================
#==============Tableau de Bord Proprement dit==================
#==============================================================
with tables[2]:
    st.text("Visuels Sur les Indicateurs")
    cl1,cl2,cl3,cl4=st.columns(4)
    ci_df1 = data[data["indicator"]=="Unemployment rate 2024 (%)"]
    metric_chom=round(ci_df1['Valeur'].mean(),2)
    delta_chom = round(ci_df1['Valeur'].median(),2)
            
    ci_df2 = data[data["indicator"]=="GDP (millions 2017 PPP$)"]
    metric_val_GDP=round(ci_df2['Valeur'].sum(),2)
    delta_GDP = round(ci_df2['Valeur'].mean(),2)
    
    ci_df3 = data[data["indicator"]=="Underemployment rate (%)"]
    metric_val_under=round(ci_df3['Valeur'].mean(),2)
    delta_under = round(ci_df3['Valeur'].max(),2)
    
    ci_df4 = data[(data["indicator"]=="Employment by sex and age (thousands)")&(data["Gender"]=="Sex: Total")]
    metric_val_emp=round(ci_df4['Valeur'].sum(),2)
    delta_emp = round(ci_df4['Valeur'].mean(),2)
    
    with cl1:
        display_single_metric_advanced("PIB TOTAL", metric_val_GDP, delta_GDP, unit="$", caption="Moyenne", color_scheme="green")
    with cl2:
        display_single_metric_advanced("Taux de chômage Moyen", metric_chom, delta_chom, unit="%", caption="Maximun", color_scheme="red")
    with cl3: 
        display_single_metric_advanced("Taux Global de Sous Emplois", metric_val_under, delta_under, unit="%", caption="Maximum")
    with cl4: 
        display_single_metric_advanced("Nombre d'Emploi", metric_val_emp, delta_emp, unit="", caption="Moyenne",color_scheme="orange")
    
    with st.container():
        col1, col2 = st.columns([5, 1])
        
        with col2:
            st.write("Indicateur:")
            
            #
            
            
            
            #display_single_metric_advanced("PIB", metric_val2, delta2, unit="€", caption="Variation annuelle")
            st.metric(label="PIB", value=metric_val_GDP, delta=delta_GDP)
            #st.metric(label="Chômage", value=metric_val1, delta=delta1)
            #st.metric(label="Sous emploi", value=metric_value, delta=delta)
            #st.metric(label="metric_label", value=metric_value, delta=delta)
        
        with col1:
            with st.expander("Pour l'Afrique entièrement", expanded=True):
                tabs = st.tabs(["Carte", "Par sous Région", "Statistiques"])
            with tabs[0]:
                st.text("Carte Globale")
                subcol1,subcol2 =st.columns(2)
                with subcol1:
                    indicateur = st.selectbox("Choisisez un indicateur",list(data['indicator'].unique()))
                    gender=st.selectbox("Genre",list(data[data['indicator']==indicateur]['Gender'].unique()))
                    crit = st.selectbox("Choisisez une tranche d'age",list(data[(data['indicator']==indicateur)&(data['Gender']==gender)]['Critere'].unique()))
                    if len(indicateur)!=0:    
                        data_ex = {'Category': ['A', 'B', 'C', 'D'], 'Value': [25, 15, 30, 30]}
                        df_ex = pd.DataFrame(data_ex)
                        make_bar_polar_chart(data_to_grph, 'Region', 'Valeur',width=300, height=300,colors=['red','green','blue','orange'])                     
                        make_histogram(data_to_grph,indicateur,width=450,height=450)  
                with subcol2:
                    #make_chloropleth(geo_Africa_df,indicateur,"Valeur",width=600,height=350)
                    chloro_cart(data_to_grph,indicateur,"Valeur")   
                    if gender!="nan":
                        dfplot=data[data['Gender']==gender]
                    else:
                        dfplot=data
                    make_line(dfplot,"Valeur","Region",titre="Evolution of " + indicateur)
            with tabs[1]:
                st.text("Par sous Région")
                #make_chloropleth_by(geo_Africa_df,'POP_EST',"Region",width=700,height=350)
                
                
            with tabs[2]:
                st.text("Afficher les statistiques ici")
            
            g_pays = st.multiselect("Choisisez un ou plusieurs pays",list(data['country'].unique())) 
            s_col1,s_col2=st.columns(2)
            with s_col1:
                
                if len(g_pays)!=0:
                    df_pay=data[data['country'].isin(g_pays)]
                    make_line(df_pay,"Valeur",'country',titre= "Evolution of " + indicateur)
            with s_col2:
                if len(g_pays)!=0:
                    df_pay=data[data['country'].isin(g_pays)]
                    make_multy_bar(df_pay,"Valeur",'country')

                #st.dataframe(data_grph)
            
            

#==============================================================
        # ...votre code pour afficher les statistiques...
with tables[3]:
    st.text("Analyses Statistiques Avancée")
        # ...votre code pour afficher les statistiques...
with tables[4]:
    st.text("Interprétation")
    df_model=df_model[['pays','annee','Unemployment Rate','GDP','Population_Female','Population_Male','Working Poverty Rate_Total','Output per Worker','POP_Female_15-24','POP_Female_25-34','POP_Female_35-44','POP_Female_45-54','POP_Female_55-64','POP_Female_65+','POP_Male_15-24','POP_Male_25-34','POP_Male_35-44','POP_Male_45-54','POP_Male_55-64','POP_Male_65+']]
    def panel_model(df):
        import matplotlib.pyplot as plt
        model_contraint = smf.ols(" df['Unemployment Rate'] ~ np.log(GDP) + np.log(Population_Female) + np.log(Population_Male) + df['Working Poverty Rate_Total'] + np.log(df['Output per Worker']) + np.log(df['POP_Female_15-24']) + np.log(df['POP_Female_25-34']) + np.log(df['POP_Female_35-44']) + np.log(df['POP_Female_45-54']) + np.log(df['POP_Female_55-64']) +  np.log(df['POP_Female_65+']) + np.log(df['POP_Male_15-24']) + np.log(df['POP_Male_25-34']) + np.log(df['POP_Male_35-44']) + np.log(df['POP_Male_45-54']) + np.log(df['POP_Male_55-64']) + np.log(df['POP_Male_65+'])", data=df).fit()
    # print(model_contraint.summary())
        
        summary = model_contraint.summary()
        
        # Récupérer les variables significatives
        alpha = 0.05
        significant_vars = summary.tables[1].data[1:]  # Ignore the first row which is the headers
        
        # Stocker les résultats dans un DataFrame
        results = []
        for var in significant_vars:
            variable_name = var[0]
            coefficient = var[1]
            p_value = var[4]
            
            if float(p_value) < alpha:
                results.append((variable_name, coefficient, p_value))
        
        # Créer un DataFrame pour les variables significatives
        results_df = pd.DataFrame(results, columns=['Variable', 'Coefficient', 'p-value'])
        
        # Afficher le DataFrame des résultats
        #print("Variables significatives :")
        #print(results_df)
        
        # R carré du modèle
        r_squared = model_contraint.rsquared
        #print(f"\nR carré du modèle : {r_squared:.4f}")
        
        # Récupérer la p-value globale du modèle
        p_value_global = model_contraint.f_pvalue
        #print(f"\nP-value globale du modèle : {p_value_global:.4f}")
        dic = {'np.log(GDP)': 'GDP',
        "np.log(Population_Female)":'Population_Female',
        "np.log(Population_Male)":'Population_Male',
        "np.log(df['Output per Worker'])":'Output per Worker',
        "np.log(df['POP_Female_15-24'])":'POP_Female_15-24',
        "np.log(df['POP_Female_25-34'])":'POP_Female_25-34',
        "np.log(df['POP_Female_35-44'])":'POP_Female_35-44',
        "np.log(df['POP_Female_45-54'])":'POP_Female_45-54',
        "np.log(df['POP_Female_55-64'])":'POP_Female_55-64',
        "np.log(df['POP_Female_65+'])":'POP_Female_65+',
        "np.log(df['POP_Male_15-24'])":'POP_Male_15-24',
        "np.log(df['POP_Male_25-34'])":'POP_Male_25-34',
        "np.log(df['POP_Male_35-44'])":'POP_Male_35-44',
        "np.log(df['POP_Male_45-54'])":'POP_Male_45-54',
        "np.log(df['POP_Male_55-64'])":'POP_Male_55-64',
        "np.log(df['POP_Male_65+'])":'POP_Male_65+'
        }
        X = []
        
        for i in results_df["Variable"].tolist()[1:]:
            if i == "df['Working Poverty Rate_Total']":
                pass
            else: X.append(dic[i])
        X
        return results_df,r_squared,p_value_global,X

    
        # ...votre code pour afficher les statistiques...
    
    def plot_scatter(df, x, year):
        df_filtered = df[df['annee'] == year]
        fig = px.scatter(
            df_filtered,
            x=x,
            y='Unemployment Rate',
            color='pays',
            size_max=10
        )
        st.plotly_chart(fig)
     
    results_df,r_squared,p_value_global,X=panel_model(df_model)  
    
    st.dataframe(results_df)
    st.write('R_carré:' + str(r_squared) )
    st.write('P_value:' + str(p_value_global) )
    cm1, cm2=st.columns(2)
    with cm1:
        plot_scatter(df_model, X[0], 2021)
        plot_scatter(df_model, X[1], 2021)
        plot_scatter(df_model, X[2], 2021)
        plot_scatter(df_model, X[3], 2021)
    with cm2:
        plot_scatter(df_model, X[4], 2021)
        plot_scatter(df_model, X[5], 2021)
        plot_scatter(df_model, X[6], 2021)
        plot_scatter(df_model, X[7], 2021)
st.markdown("---")
colon1, colon2=st.columns(2)

def creer_graphique_area(df, titre, x_col, y_cols, couleurs=None):
    

    fig = go.Figure()

    if couleurs is None:
        couleurs = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A'] # Couleurs par défaut

    for i, col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[col],
            hoverinfo='x+y',  # Afficher x et y au survol
            mode='lines',
            fill='tonexty' if i == 0 else 'tozeroy', # Remplissage de l'aire
            line=dict(width=2, color=couleurs[i % len(couleurs)]), # Largeur et couleur des lignes
            name=col,
            stackgroup='one' # Empilement des aires
        ))

    fig.update_layout(
        title=titre,
        xaxis_title=x_col,
        yaxis_title="Valeur",
        plot_bgcolor='#f2f2f2',  # Couleur d'arrière-plan du graphique
        paper_bgcolor='#f2f2f2',  # Couleur d'arrière-plan de la page
        font=dict(family="Arial", size=12, color="#333"), # Police
        hovermode='x unified', # Survol unifié sur l'axe x
        legend=dict(
            orientation="h", # Légende horizontale
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_layout(
        title_font=dict(size=16, color="white", family="Arial"),
        legend=dict(font=dict(size=12)),
        plot_bgcolor='rgba(0,0,0,0)',  # Arrière-plan transparent
        paper_bgcolor='rgba(0,0,0,0)',  # Arrière-plan transparent
        margin=dict(l=50, r=50, b=50, t=80, pad=4)
    )
    
    st.plotly_chart(fig)

with colon1:
    Pays=st.selectbox("Choisisez un pays", data['country'].unique())
    data_country=data[data['country']==Pays]
    
    data = {'Date': pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-15', '2024-01-22', '2024-01-29']),
        'Série A': [10, 15, 12, 18, 22],
        'Série B': [5, 8, 10, 12, 15],
            'Série C': [2, 4, 6, 8, 10]}
    df_n = pd.DataFrame(data)
    df_ct_PIB=data_country[data_country["indicator"]=="GDP (millions 2017 PPP$)"]
    df_ct_unemp=data_country[data_country["indicator"]=="Underemployment rate (%)"]
    df_ct_emp=data_country[data_country["indicator"]=="Employment by sex and age (thousands)"]
    df_ct_lab=data_country[data_country["indicator"]=="Labour income share (%)"]
    
    creer_graphique_area(df_ct_PIB, "Evolution du PIB", "time", ["Valeur"])

with colon2:
    
    fig = px.line(df_ct_emp, x="time", y="Valeur", markers=True,title="Evolution de l'emploi")
    st.plotly_chart(fig)
    
