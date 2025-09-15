import streamlit as st
import pickle
#from price_by_multifeatures  import clean_data, model_predictions, wrangle
import pandas as pd
import json
import numpy as np

st.set_page_config(page_title="Housing Price Prediction App", page_icon="🏠", layout= "centered")

@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("pricemodel.pkl", 'rb'))
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'pricemodel.pkl' exists")
        return None

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Administrator\Desktop\Data Expert\Data Analytics\Projects\World Quant\Housing in Buenos Aires\data\mexico-cleaned-data.csv")
    return df


def sidebar(data):
    st.sidebar.header("Input Parameters")
    data= load_data()
    
    slider_labels = [
        ("Size", "surface_covered_in_m2"), 
        ("Latitude", "lat"),
        ("Longitude", "lon"),
         #"Neighborhood", "neighborhood"
    ]
    
    input_data = {}
    for label, column in slider_labels:
        input_data[column] = st.sidebar.slider(label, 
                                               float(data[column].min()), 
                                               float(data[column].max()), 
                                               float(data[column].mean()))
    neighborhoods = sorted(data["neighbourhood"].unique()) if "neighbourhood" in data.columns else ["Default"]
    input_data["neighbourhood"] = st.sidebar.selectbox(
        "Neighbourhood",
        neighborhoods,
        index=0 
    )
    
    #return pd.DataFrame(input_data, index=[0])
    return input_data

def apply_model(model,df):
    model = pickle.load(open("pricemodel.pkl", 'rb'))
    input = np.array(list(df.values())).reshape(1,-1)
    model.predict(input)
   

def main(model,df):
    if model is None:
        return None
    try:
        expected_columns = ['surface_total_in_m2', 'lat', 'lon', 'neighbourhood']
        if not all(col in df.columns for col in expected_columns):
            missing = [col for col in expected_columns if col not in df.columns]
            st.error(f"Missing columns for model: {missing}")
            return None
        df = df[expected_columns]
        
        input_array = np.array(df.values).reshape(1, -1)
        prediction = model.predict(input_array)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None
    
def main():
    model = load_model()
    data = load_data()
    
    if data.empty or model is None:
        st.error("Failed to load data or model.")
        return
    
    #st.write("Data Columns for Reference:", data.columns.tolist())
    df_input = sidebar(data)
    
    st.title("Housing Price Prediction App 🏠")
    st.write("Interactive model to estimate apartment prices based on size(m²), location and neighborhood features.")
    
    if st.button("Predict Price"):
        prediction = apply_model(model, df_input)
        if prediction is not None:
            st.subheader("Predicted Price")
            st.write(f"${prediction:,.2f} MXN")
    
    st.subheader("Input Parameters")
    st.write(df_input)
    
    # st.subheader("Data Sample")
    # st.write(data.head())

if __name__ == '__main__':
    main()    