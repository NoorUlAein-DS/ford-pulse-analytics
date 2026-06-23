import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="DriveValue Analytics")

# --- CUSTOM CSS FOR MODERN CAR-VALUATION UI ---
st.markdown("""
    <style>
    /* Premium Minimalist Background */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }
    /* Modern Glassmorphism Card Style */
    div[data-testid="stBlock"] > div:first-child {
        background-color: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.04);
        border: 1px solid #eaeaea;
    }
    /* Clean Typography */
    h1 {
        color: #111827;
        font-family: '-apple-system', BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
    }
    h2, h3 {
        color: #1f2937;
        font-family: '-apple-system', sans-serif;
        font-weight: 700;
    }
    /* Premium Big Valuation Display */
    .metric-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1e3a8a;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02);
    }
    .metric-title {
        font-size: 1.1rem;
        color: #6b7280;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2.8rem;
        color: #111827;
        font-weight: 800;
        margin-top: 0.2rem;
    }
    </style>
""", unsafe_with_html=True)

# --- LOAD MODELS & DATA ---
@st.cache_resource
def load_resources():
    with open('ford_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    data = pd.read_csv('ford.csv')
    return model, model_columns, data

try:
    model, model_columns, data = load_resources()
except Exception as e:
    st.error("Required files not found. Please ensure all model and data files are uploaded.")
    st.stop()

# --- HEADER ---
st.title("Specifications")
st.markdown("---")

# --- MAIN LAYOUT ---
col_inputs, col_graphs = st.columns([1, 2], gap="large")

with col_inputs:
    # Form Clean Fields
    car_model = st.selectbox("Model", sorted(data['model'].unique()))
    year = st.slider("Year", int(data['year'].min()), 2025, 2018)
    transmission = st.radio("Transmission", sorted(data['transmission'].unique()))
    fuel_type = st.selectbox("Fuel Type", sorted(data['fuelType'].unique()))
    mileage = st.number_input("Mileage", min_value=0, value=15000, step=1000)
    mpg = st.number_input("MPG (Combined)", min_value=0.0, value=55.4, step=0.1)
    tax = st.number_input("Annual Tax ($)", min_value=0, value=145)
    engine_size = st.number_input("Engine Size (L)", min_value=0.0, value=1.0, step=0.1)
    
    st.markdown("<br>", unsafe_with_html=True)
    submit = st.button("Generate Valuation", use_container_width=True, type="primary")

with col_graphs:
    # Filter data dynamically according to the user input
    filtered_data = data[data['model'] == car_model]
    
    graph_col1, graph_col2 = st.columns(2)
    
    with graph_col1:
        # Dynamic Graph 1: Price Trend Over Years
        trend_data = filtered_data.groupby('year')['price'].mean().reset_index()
        fig_year = px.line(trend_data, x='year', y='price', 
                           title=f"Price Trend: {car_model}",
                           template="simple_white",
                           labels={'price': 'Price ($)', 'year': 'Year'})
        fig_year.update_traces(line_color='#1e3a8a', line_width=3)
        fig_year.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_year, use_container_width=True)

    with graph_col2:
        # Dynamic Graph 2: Engine Size vs Price Scatter
        fig_engine = px.scatter(filtered_data, x='engineSize', y='price',
                                title="Price vs Engine Size Distribution",
                                template="simple_white",
                                color_discrete_sequence=['#1e3a8a'],
                                labels={'price': 'Price ($)', 'engineSize': 'Engine Size (L)'})
        fig_engine.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_engine, use_container_width=True)
        
    st.markdown("<br><br>", unsafe_with_html=True)
    st.subheader("Valuation")
    
    # --- PREDICTION LOGIC ---
    if submit or True: # Keeps valuation visible
        # Prepare input row with exact same columns as One-Hot Encoding
        input_data = pd.DataFrame([{
            'year': year,
            'mileage': mileage,
            'tax': tax,
            'mpg': int(mpg),
            'engineSize': int(engine_size),
            f'model_{car_model}': 1,
            f'transmission_{transmission}': 1,
            f'fuelType_{fuel_type}': 1
        }])
        
        # Reindex to match training columns, filling missing with 0
        input_encoded = input_data.reindex(columns=model_columns, fill_value=0)
        
        # Predict
        predicted_price = model.predict(input_encoded)[0]
        
        # Output block
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Estimated Market Value ({car_model.strip()} - {year})</div>
                <div class="metric-value">${predicted_price:,.2f} <span style="font-size:1.5rem; color:#10b981;">▲ + USD</span></div>
            </div>
        """, unsafe_with_html=True)
