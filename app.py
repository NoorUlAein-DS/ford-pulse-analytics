import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    layout="wide", 
    page_title="DriveValue Analytics",
    page_icon="🦋",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS WITH FOREST MIST COLOR PALETTE ---
st.markdown("""
<style>
    :root {
        --primary: #2F3E46;
        --secondary: #354F52;
        --accent: #04A98C;
        --light: #CAD2C5;
        --dark: #2F3E46;
        --text-primary: #2F3E46;
        --text-secondary: #354F52;
        --bg-light: #F5F7F4;
    }
    
    .stApp {
        background: var(--bg-light);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--light);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: var(--accent);
        border-radius: 10px;
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 1.8rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(202, 210, 197, 0.2);
    }
    
    .dashboard-title {
        color: white;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 600;
        font-size: 2.2rem;
        margin: 0;
        letter-spacing: -0.3px;
    }
    
    .dashboard-title span {
        color: var(--accent);
    }
    
    .dashboard-subtitle {
        color: rgba(255,255,255,0.75);
        font-size: 1rem;
        margin-top: 0.3rem;
        font-weight: 300;
    }
    
    /* Metric card */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(202, 210, 197, 0.3);
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(47, 62, 70, 0.06);
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: var(--secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
        margin-top: 0.2rem;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 0.3rem;
    }
    
    .metric-badge.success {
        background: rgba(4, 169, 140, 0.12);
        color: var(--accent);
    }
    
    .metric-badge.info {
        background: rgba(53, 79, 82, 0.08);
        color: var(--secondary);
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--primary);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 10px;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: var(--secondary);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(47, 62, 70, 0.15);
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(202, 210, 197, 0.3);
    }
    
    /* Valuation result */
    .valuation-box {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        color: white;
        border: 1px solid rgba(202, 210, 197, 0.2);
    }
    
    /* Section headers */
    .section-title {
        color: var(--primary);
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: var(--secondary);
        font-size: 0.85rem;
        border-top: 1px solid rgba(202, 210, 197, 0.3);
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

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
st.markdown("""
<div class="dashboard-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 class="dashboard-title">DriveValue <span>Analytics</span></h1>
            <p class="dashboard-subtitle">Intelligent vehicle valuation platform</p>
            <p class="dashboard-subtitle">Developed By Aein</p>
        </div>
        <div style="display: flex; gap: 1.5rem; color: rgba(255,255,255,0.6); font-size: 0.85rem;">
            <span>● Live</span>
            <span>|</span>
            <span>ML Powered</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
col_left, col_right = st.columns([0.4, 0.6], gap="large")

with col_left:
    # Custom native box container instead of buggy HTML div to remove the empty white box
    with st.container(border=True):
        st.markdown('<p class="section-title">▸ Vehicle Details</p>', unsafe_allow_html=True)
        
        car_model = st.selectbox(
            "Model",
            sorted(data['model'].unique())
        )
        
        year = st.slider(
            "Year",
            int(data['year'].min()),
            2025,
            2018
        )
        
        transmission = st.radio(
            "Transmission",
            sorted(data['transmission'].unique()),
            horizontal=True
        )
        
        fuel_type = st.selectbox(
            "Fuel Type",
            sorted(data['fuelType'].unique())
        )
        
        col_mileage, col_mpg = st.columns(2)
        with col_mileage:
            mileage = st.number_input(
                "Mileage",
                min_value=0,
                value=15000,
                step=1000
            )
        with col_mpg:
            mpg = st.number_input(
                "MPG",
                min_value=0,
                value=55,
                step=1
            )
        
        col_tax, col_engine = st.columns(2)
        with col_tax:
            tax = st.number_input(
                "Tax ($)",
                min_value=0,
                value=145
            )
        with col_engine:
            engine_size = st.number_input(
                "Engine (L)",
                min_value=0.5,
                value=1.0,
                step=0.1,
                format="%.1f"
            )
        
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        submit = st.button(
            "Generate Valuation",
            use_container_width=True,
            type="primary"
        )

with col_right:
    filtered_data = data[data['model'] == car_model]
    
    # Metrics row
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        avg_price = filtered_data['price'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Price</div>
            <div class="metric-value">${avg_price:,.0f}</div>
            <div class="metric-badge success">▲ +12%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        avg_mileage = filtered_data['mileage'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Mileage</div>
            <div class="metric-value">{avg_mileage:,.0f}</div>
            <div class="metric-badge info">mi</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        count_models = len(filtered_data)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Listings</div>
            <div class="metric-value">{count_models:,}</div>
            <div class="metric-badge info">active</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<br>', unsafe_allow_html=True)
    
    # Charts
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            trend_data = filtered_data.groupby('year')['price'].mean().reset_index()
            fig_trend = px.line(
                trend_data,
                x='year',
                y='price',
                title="Price Trend",
                template="plotly_white",
                labels={'price': 'Price ($)', 'year': 'Year'}
            )
            fig_trend.update_traces(
                line_color='#04A98C',
                line_width=3,
                fill='tozeroy',
                fillcolor='rgba(4, 169, 140, 0.08)'
            )
            fig_trend.update_layout(
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=30, r=20, t=40, b=30),
                font=dict(size=11)
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with chart_col2:
            fig_engine = px.scatter(
                filtered_data,
                x='engineSize',
                y='price',
                title="Price vs Engine",
                template="plotly_white",
                labels={'price': 'Price ($)', 'engineSize': 'Engine (L)'},
                color_discrete_sequence=['#04A98C']
            )
            fig_engine.update_traces(
                marker=dict(size=8, opacity=0.6, line=dict(width=1, color='white'))
            )
            fig_engine.update_layout(
                hovermode='closest',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=30, r=20, t=40, b=30),
                font=dict(size=11)
            )
            st.plotly_chart(fig_engine, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<br>', unsafe_allow_html=True)
    
    # Valuation result
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    input_encoded = pd.DataFrame(0, index=[0], columns=model_columns)
    
    input_encoded['year'] = int(year)
    input_encoded['mileage'] = int(mileage)
    input_encoded['tax'] = int(tax)
    input_encoded['mpg'] = int(mpg)
    input_encoded['engineSize'] = float(engine_size)
    
    model_col = f"model_{car_model}"
    trans_col = f"transmission_{transmission}"
    fuel_col = f"fuelType_{fuel_type}"
    
    if model_col in input_encoded.columns:
        input_encoded[model_col] = 1
    if trans_col in input_encoded.columns:
        input_encoded[trans_col] = 1
    if fuel_col in input_encoded.columns:
        input_encoded[fuel_col] = 1
        
    input_encoded = input_encoded.astype(float)
    
    # Always execute prediction logic to keep block stable, but conditionally display results
    predicted_price = model.predict(input_encoded)[0]
    market_avg = filtered_data['price'].mean()
    price_diff = ((predicted_price - market_avg) / market_avg) * 100
    
    if submit:
        st.markdown(f"""
        <div class="valuation-box">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <div style="font-size: 0.8rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.5px;">
                        Estimated Value
                    </div>
                    <div style="font-size: 3rem; font-weight: 700; margin-top: 0.2rem;">
                        ${predicted_price:,.2f}
                    </div>
                    <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.3rem;">
                        {car_model} · {year} · {fuel_type}
                    </div>
                </div>
                <div style="text-align: right; background: rgba(255,255,255,0.08); padding: 1rem 1.5rem; border-radius: 12px;">
                    <div style="font-size: 0.75rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.5px;">
                        Market Comparison
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 600; color: #CAD2C5;">
                        {price_diff:+.1f}%
                    </div>
                    <div style="font-size: 0.75rem; opacity: 0.6;">
                        avg ${market_avg:,.0f}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(".")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div class="footer">
    DriveValue Analytics · Powered by Machine Learning
</div>
""", unsafe_allow_html=True)
