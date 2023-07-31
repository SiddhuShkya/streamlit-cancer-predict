import os
import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import numpy as np

# streamlit run ./Application/main.py

def add_predictions(data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    feature_names = list(data.keys())
    feature_values = np.array(list(data.values())).reshape(1, -1)
    input_df = pd.DataFrame(feature_values, columns=feature_names)
    input_array_scaled = scaler.transform(input_df)
    prediction = model.predict(input_array_scaled)
    if prediction[0] == 0:
        st.header('Benign 😁')
    else:
        st.header('Malignant 😥')
    st.write("Probability of being Benign : ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being Malignant : ", model.predict_proba(input_array_scaled)[0][1])
    st.write("This model can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")
    return

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(columns=['diagnosis'], axis=1)
    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val)/(max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict

def get_radar_chart(data):
    data = get_scaled_values(data)
    categories = ['Radius', 'Texture', 'Perimeter', 
                'Area', 'Smoothness', 'Compactness', 'Concavity', 
                'Concave Points', 'Symmetry', 'Fractal Dimension']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r = [
            data['radius_mean'], data['texture_mean'], data["perimeter_mean"], 
            data['area_mean'], data['smoothness_mean'], data['compactness_mean'], 
            data['concavity_mean'], data['concave points_mean'], data['symmetry_mean'], 
            data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r = [
            data['radius_se'], data['texture_se'], data["perimeter_se"], 
            data['area_se'], data['smoothness_se'], data['compactness_se'], 
            data['concavity_se'], data['concave points_se'], data['symmetry_se'], 
            data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r = [
            data['radius_worst'], data['texture_worst'], data["perimeter_worst"], 
            data['area_worst'], data['smoothness_worst'], data['compactness_worst'], 
            data['concavity_worst'], data['concave points_worst'], data['symmetry_worst'], 
            data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0,1]
            )
        ),
        showlegend=True
    )
    return fig

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(columns=['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M' : 1, 'B' : 0})
    return data

def add_sidebar():
    st.sidebar.header('Cell Nuclei Measurements')
    data = get_clean_data()
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst")]
    input_dict = {}
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label=label,
            min_value=0.0,
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    
    return input_dict

def main():
    st.set_page_config(
        page_title='Breast Cancer Predictor',
        page_icon='👩‍⚕️',
        layout="wide",
        initial_sidebar_state='expanded'
    )
    with open("./Application/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    input_data = add_sidebar()
    with st.container():
        st.title('Breast Cancer Predictor 👩‍⚕️')
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app predicts using a machine learning model whether a breast mass is bening or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.")
    col1, col2 = st.columns([4, 1])
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)
    return


if __name__ == '__main__':
    main()