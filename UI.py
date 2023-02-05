import streamlit as st
import pandas as pd
import pickle
import spacy
import time
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_model():
    pkl_filename = "RF2.sav"
    with open(pkl_filename, 'rb') as file:
        countVect, pickle_model = pickle.load(file)

    encoder = LabelEncoder()
    encoder.classes_ = np.load('classes.npy', allow_pickle=True)
    return countVect, pickle_model, encoder;


def run():
    countVect, model, encoder = load_model()
    st.title("Ticket Assignment System")
    st.header('Enter ticket details to get the assignee group')
    caller = st.text_input('Caller')
    short_desc = st.text_area('Short Description')
    descr = st.text_area('Description')
    output = ""
    if st.button("Assign"):
        output = model.predict(countVect.transform([descr]))
        assigned_group = encoder.inverse_transform(output)[0]
        st.success(f"Ticket assign to {assigned_group}")

if __name__ == "__main__":
    run()