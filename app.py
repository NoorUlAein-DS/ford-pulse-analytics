import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Ford Analytics")

# --- CUSTOM CSS FOR HIGH-END UI ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }
    div[data-testid="stBlock"] > div:first-child {
        background-color: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid #eaeaea;
    }
    h1 {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 1.5rem;
    }
    h2, h3 {
        color: #333;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        margin-top: 1rem;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #10B981 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.3);
        transform: translateY(-2px);
    }
    .plotly-graph-div {
        border-radius: 15px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA & MODEL LOADING ---
@st.cache_resource
def load_assets():
    try:
        with open('ford_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model_columns.pkl', 'rb') as f:
            model_columns = pickle.load(f)
        data = pd.read_csv('ford.csv')
        return model, model_columns, data
    except FileNotFoundError:
        st.error("Error: Required files not found.")
        return None, None, None

model, model_columns, data = load_assets()

# --- 2. LAYOUT: TWO COLUMNS ---
col1, col2 = st.columns([1, 2], gap="large")

if model and data is not None:
    with col1:
        st.title("🚗 ValuEngine")
        st.markdown("### Professional Ford Price Prediction")
        
        with st.container():
            st.subheader("Vehicle Specifications")
            unique_models = sorted([m.strip() for m in data['model'].unique()])
            unique_transmissions = sorted(data['transmission'].unique())
            unique_fuels = sorted(data['fuelType'].unique())

            model_input = st.selectbox("Model", unique_models, index=unique_models.index('Fiesta'))
            year = st.slider("Year", min_value=1996, max_value=2020, value=2018)
            transmission_input = st.radio("Transmission", unique_transmissions)
            fuel_input = st.selectbox("Fuel Type", unique_fuels)

            c1, c2 = st.columns(2)
            with c1:
                mileage = st.number_input("Mileage", min_value=0, value=15000, step=1000)
                tax = st.number_input("Annual Tax (£)", min_value=0, value=145)
            with c2:
                mpg = st.number_input("MPG (Combined)", min_value=10.0, value=55.4)
                engineSize = st.number_input("Engine Size (L)", min_value=0.0, value=1.0, step=0.1)

            predict_btn = st.button("Generate Valuation", use_container_width=True)

            if predict_btn:
                input_data = pd.DataFrame([{
                    'model': ' ' + model_input,
                    'year': year,
                    'transmission': transmission_input,
                    'mileage': mileage,
                    'fuelType': fuel_input,
                    'tax': tax,
                    'mpg': int(mpg),
                    'engineSize': int(engineSize)
                }])

                input_encoded = pd.get_dummies(input_data)
                final_features = pd.DataFrame(0, index=[0], columns=model_columns)
                for col in input_encoded.columns:
                    if col in final_features.columns:
                        final_features[col] = input_encoded[col].values

                prediction = model.predict(final_features)
                
                st.markdown("---")
                st.subheader("Market Valuation")
                st.metric(label=f"Estimated Price for {year} Ford {model_input}", value=f"£{prediction[0]:,.2f}")

    with col2:
        st.markdown("## 📊 Ford Market Pulse")
        tab1, tab2 = st.tabs(["Market Drivers", "Model Deep-Dive"])

        with tab1:
            st.subheader("What Drives Ford Prices?")
            t1_col1, t1_col2 = st.columns(2)

            with t1_col1:
                avg_price_year = data.groupby('year')['price'].mean().reset_index()
                fig_year = px.line(avg_price_year, x='year', y='price', 
                                    title='Average Price vs. Year (Depreciation Curve)',
                                    color_discrete_sequence=['#2563EB'])
                fig_year.update_layout(plot_bgcolor="white")
                st.plotly_chart(fig_year, use_container_width=True)

            with t1_col2:
                fig_engine = px.box(data, x='engineSize', y='price',
                                    title='Price Distribution by Engine Size')
                fig_engine.update_layout(plot_bgcolor="white")
                st.plotly_chart(fig_engine, use_container_width=True)

        with tab2:
            st.subheader("Model-Specific Segmentation")
            models_to_plot = st.multiselect("Filter Models", unique_models, default=['Fiesta', 'Focus', 'Kuga'])
            filtered_data = data[data['model'].str.strip().isin(models_to_plot)]

            fig_model = px.box(filtered_data, x='model', y='price', color='model', title='Price Segmentation')
            fig_model.update_layout(plot_bgcolor="white", showlegend=False)
            st.plotly_chart(fig_model, use_container_width=True)
