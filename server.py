import streamlit as st
import json 
import joblib
import numpy as np
from streamlit_lottie import st_lottie

st.set_page_config(layout="wide")



def load_lottifiel(filepath:str):
    with open(filepath,'r') as f:return json.load(f)

st_lottie( load_lottifiel("Animation - 1706023950044.json"),height=250)

st.title(f"Electrical Grid Stability prediction")

voting_classifier_with_f_e = joblib.load('model/voting_classifier_with_f_e.job')
stabf = joblib.load('pipeline/LabelEncoder_stabf.job')

tau1 = st.number_input('Reaction time of participant (real from the range [0.5,10]s), tau1(τ)')
tau2 = st.number_input('Reaction time of participant (real from the range [0.5,10]s), tau2(τ)')
tau3 = st.number_input('Reaction time of participant (real from the range [0.5,10]s), tau3(τ)')
tau4 = st.number_input('Reaction time of participant (real from the range [0.5,10]s), tau4(τ)')
g1 = st.number_input('Coefficient (gamma) proportional to price elasticity (real from the range [0.05,1]s^-1), g1(γ)')
g2 = st.number_input('Coefficient (gamma) proportional to price elasticity (real from the range [0.05,1]s^-1), g2(γ)')
g3 = st.number_input('Coefficient (gamma) proportional to price elasticity (real from the range [0.05,1]s^-1), g3(γ)')
g4 = st.number_input('Coefficient (gamma) proportional to price elasticity (real from the range [0.05,1]s^-1), g4(γ)')

if st.button('Submit'):
    predict = voting_classifier_with_f_e.predict(np.array([[tau1,tau2,tau3,tau4,g1,g2,g3,g4]]))[0]
    st.title(f"Your Electrical Grid is {stabf.inverse_transform([predict])[0]}")