import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
#import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
import json
import plotly.express as px
import altair as alt
import seaborn as sns



def identifiant_client():
        SK_ID_CURR=st.sidebar.selectbox('SK_ID_CURR',(X.SK_ID_CURR))

        data={' SK_ID_CURR ': SK_ID_CURR }
    
        ID_client = pd.DataFrame(data,index=[0])
        return ID_client




def numeric(col):
    fig = plt.figure(figsize=(15,5))
    plt.title("Distribution of " +col)
    ax = sns.distplot(X[col])
    plt.axline((donnees_client[col].iloc[0], 0), (donnees_client[col].iloc[0], 0.00005), c='darkorange', ls='dashed')
    st.pyplot(fig)
        
        
def relation_entre_variables (var1, var2):
    fig = plt.figure(figsize=(15,5))
    plt.title(var2 +" en fonction de " + var1)
    sns.scatterplot(data=X, x=var1, y=var2)
    st.pyplot(fig)
        

if __name__=="__main__":
    st.set_page_config(
        page_title="Streamlit basics app",
        layout="centered"
    )

    st.title("Application qui prédit l'accord du crédit")

    st.write("Auteur : Brahim AIT OUALI  - Data Scientist")
  

    # Display the LOGO
    img = Image.open("LOGO.png")
    st.sidebar.image(img, width=300)

    #Collecter le profil d'entrée
    st.sidebar.header("Identifiant du client")




    X = pd.read_csv('X_test_init_sample_saved.csv')
    
    

    # Variables sélectionnées
    df_vars_selected = pd.read_csv('df_vars_selected_saved.csv')
    vars_selected = df_vars_selected['feature'].to_list()

      
   # Afficher les données du client:
    vars_selected.insert(0, 'SK_ID_CURR') # Ajout de l'identifiant aux features 
    st.subheader('1. Les données du client')

    
    input_df=identifiant_client().iloc[0,0]
    X = X[vars_selected]    
    donnees_client = X[X['SK_ID_CURR']==input_df] # ligne du dataset qui concerne le client
    st.write(donnees_client)
    

    st.subheader("2. Distribution des variables")
    option = st.selectbox(
      'Choisissez la variable dont vous voulez représenter la distribution', (vars_selected[1:]))

    st.write("Distribution de ", option)
    numeric(option)
  
    st.write("Valeur pour le client sélectionné = ", donnees_client[option].iloc[0])

    st.write("-------------------------------------------------------------------------------------------")
    st.subheader("3. Relation entre les variables")
    option1 = st.selectbox(
      'Choisissez la variable en abscisse', (vars_selected[1:]))
    st.write("________________________________________")
    option2 = st.selectbox(
      'Choisissez la variable en ordonnée', (vars_selected[1:]))
        
    relation_entre_variables (option1, option2)

    # Importer le modèle
    from joblib import dump, load
    pipeline_loaded = load('pipeline_credit.joblib')
    pipeline = pipeline_loaded




    # importing the requests library
    import requests
  
        
    API_ENDPOINT = "http://13.37.154.218/predict"



    # Envois de la requête à la Fastapi
  
 
    # data to be sent to api

    data ={"SK_ID_CURR": float(input_df)}
        
    #st.write(data)
    # sending post request and saving response as response object
        
    r = requests.post(url = API_ENDPOINT, json = data)
    prevision = r.text
    #st.write(prevision)  
    prevision = json.loads(prevision)
        
    #st.write(prevision["reponse"])
    #st.write(type(prevision))
    
    # Appliquer le modèle sur le profil d'entrée

    st.subheader("4. Interprétation de la prévision")

    st.write("* #### Client positif = client en défaut ")
    st.write("* #### Client négatif = bon client ")
        
    st.write("##### Les 2 types d'erreurs possibles minimisées par notre modèle: ")
    st.write("           - FN = faux négatif (mauvais client prédit bon client : donc crédit accordé et perte en capital)")
    st.write("           - FP = faux positif (bon client prédit mauvais : donc refus crédit et manque à gagner en marge)")

    st.write("##### Un seuil S= 0.3783783783783784  a été calculé pour minimiser au maximum la perte en capital des FP :")
        
    st.write (" ##### P = probablité que le client soit positif")
    st.write (" * ###### Si P > S alors le client est positif")
    st.write (" * ###### Si P < S alors le client est négatif")

    st.subheader("5. Résultat pour ce client ")
   
    st.write("P = ", prevision["reponse"])
    seuil = 0.3783783783783784
    
    
    #if y_train_pred_proba[:,1] > seuil:    
    if prevision["reponse"] > seuil:
        st.write("###### Crédit refusé")
    else:
        st.write("###### crédit accordé")


    # Model Explainer
    st.subheader("6. L'explication du résltat ")
    import lime
    import lime.lime_tabular
    
    
        
    predict_fn_rf = lambda x: pipeline.predict_proba(x).astype(float)

    explainer = lime.lime_tabular.LimeTabularExplainer((X[vars_selected].drop(['SK_ID_CURR'],axis=1)).values, 
                                    feature_names = (X[vars_selected].drop(['SK_ID_CURR'],axis=1)).columns,class_names=['Négatif', 'Positif'],kernel_width=5)

    choosen_instance = (donnees_client.drop(['SK_ID_CURR'],axis=1)).values[0]
    exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=15)
    exp.show_in_notebook(show_all=False)

    new_exp = exp.as_list()
    fig = plt.figure()

    label_limits = [i[0] for i in new_exp]
    #st.write(label_limits)
    label_scores = [i[1] for i in new_exp]
    #st.write(label_scores)

    #plt.barh(label_limits, label_scores)
      
    #st.pyplot(fig)
    html = exp.as_html()
    import streamlit.components.v1 as components
    components.html(html, height=800)

    with plt.style.context("ggplot"):
                exp.as_pyplot_figure()
        
    
    st.write (" * En orange les facteurs  *défavorables*  à l'octroi du crédit")
    st.write (" * En bleu les facteurs  *favorables*  à l'octroi du crédit")
   
    

