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



def identifiant_client():
        SK_ID_CURR=st.sidebar.selectbox('SK_ID_CURR',(X.SK_ID_CURR))

        data={' SK_ID_CURR ': SK_ID_CURR }
    
        ID_client = pd.DataFrame(data,index=[0])
        return ID_client


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
    

    # Importer le modèle
    from joblib import dump, load
    pipeline_loaded = load('pipeline_credit.joblib')
    pipeline = pipeline_loaded




    # importing the requests library
    import requests
    #def sendrequest_to_fastapi():
    API_ENDPOINT = "http://15.236.121.236/predict"


    # Envois de la requête à la Fastapi
    #prevision = sendrequest_to_fastapi


 
    # data to be sent to api

    data ={
       "SK_ID_CURR":input_df
    }

    # sending post request and saving response as response object
    r = requests.post(url = API_ENDPOINT, data = data)
    #r = requests.post(url = API_ENDPOINT, data = json.dumps(data))
    prevision = r.text
    st.write(prevision)  
    prevision = json.loads(prevision)
    st.write(prevision)
    prevision = prevision['detail']
    st.write(prevision[0])
    st.write(type(prevision))
    # extracting response text
    # Appliquer le modèle sur le profil d'entrée

    #prevision = pipeline.predict_proba(donnees_client.drop(['SK_ID_CURR'],axis=1))
    st.subheader("2. Résultat de la prévision")
    st.write((prevision)[0][0])
    S=0.38

    if prevision > S:
        st.write("Crédit refusé")
    else:
        st.write("crédit accordé")


    # Model Explainer
    st.subheader("3. L'explication")
    import lime
    import lime.lime_tabular

   


    predict_fn_rf = lambda x: pipeline.predict_proba(x).astype(float)

    explainer = lime.lime_tabular.LimeTabularExplainer((X[vars_selected].drop(['SK_ID_CURR'],axis=1)).values, 
                                    feature_names = (X[vars_selected].drop(['SK_ID_CURR'],axis=1)).columns,class_names=['Refusé','Accordé'],kernel_width=5)

    choosen_instance = (donnees_client.drop(['SK_ID_CURR'],axis=1)).values[0]
    exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=15)
    exp.show_in_notebook(show_all=False)

    new_exp = exp.as_list()
    fig = plt.figure()

    label_limits = [i[0] for i in new_exp]
    st.write(label_limits)
    label_scores = [i[1] for i in new_exp]
    st.write(label_scores)

    plt.barh(label_limits, label_scores)
      
    st.pyplot(fig)
    st.write(fig)
   
    
