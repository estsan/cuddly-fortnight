from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from data_processing import DataProcessor
from model_training import ModelTrainer


def run_app():
    """ Get the model to take input from an application, and guess the 
    penguin species.

    input:
    culmen_length_mm:      32 < 60
    culmen_depth_mm:       13 < 22
    flipper_length_mm:     172 < 231
    body_mass_g:           2700 < 6300
    female:                yes = 1 or no = 0
    island: 2 = Torgersen, 0 = Biscoe, 1 = Dream island

    output:
    species: 0 = Adelie, 1 = Chinstrap, 2 = Gentoo
    """

    st.title('What penguin?')

    train = st.selectbox('Do you want to train the model? Otherwise the app ' \
            'will use an already trained model', options=['Choose', 'Yes', 'No'])

    if train != 'Choose':
        if train == 'Yes':
            processor = DataProcessor()
            x_train, y_train = processor.process_data()
            trainer = ModelTrainer('SVC')
            trainer.train_model(x_train, y_train)

        scaler = joblib.load('scaler.pkl')
        model = joblib.load('model.pkl')

        culmen_length = st.slider('How long is the culmen in millimeter?', 
                min_value=30, max_value=62)
        culmen_depth = st.slider('How deep is the culmen in millimeter?', 
                min_value=11, max_value=24)
        flipper_length = st.slider('How long are the flippers in millimeter?', 
                min_value=170, max_value=235)
        body_mass = st.slider('How much does the penguin weigh in grams?', 
                min_value=2650, max_value=6350)
        female = 1 if st.radio('Is it female or male?', 
                            options=['Female', 'Male']) == 'Female' else 0
        island = st.radio('On what island did you find it?', 
                options=['Torgersen', 'Biscoe', 'Dream'])

        if island == 'Torgersen':
            islandnr = 2
        elif island == 'Biscoe':
            islandnr = 0
        else:
            islandnr = 1

        indata = pd.DataFrame(columns=['culmen_length_mm', 
                                       'culmen_depth_mm', 
                                       'flipper_length_mm', 
                                       'body_mass_g', 
                                       'female', 
                                       'island_encoded'
                                       ], 
                            data=[[culmen_length, 
                                   culmen_depth, 
                                   flipper_length, 
                                   body_mass, 
                                   female, 
                                   islandnr
                                   ]])

        indata_scaled = scaler.transform(indata)
        y_prediction = model.predict(indata_scaled)

        path_img = Path(__file__).resolve().parent / 'assets'

        if y_prediction[0] == 0:
            species = 'an Adelie'
            img = path_img / 'adelie.jpg'
        elif y_prediction[0] == 1:
            species = 'a Chinstrap'
            img = path_img / 'chinstrap.jpg'
        else:
            species = 'a Gentoo'
            img = path_img / 'gentoo.jpg'

        st.markdown(f'## :rainbow[I guess that the penguin is {species}!]')
        st.image(img)
