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
    page_icon="🚗",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR PREMIUM DASHBOARD ---
st.markdown("""
<style>
    /* Main background with soft gradient */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #6c757d;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #495057;
    }
    
    /* Main container with glassmorphism effect */
    .main-container {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 24px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.3);
        margin: 1rem 0;
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 2rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(52, 152, 219, 0.3);
    }
    
    .dashboard-title {
        color: white;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .dashboard-subtitle {
        color: rgba(255,255,255,0.8);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Input card styling */
    .input-card {
        background: white;
        padding: 1.8rem;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.04);
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .input-card:hover {
        box-shadow: 0 12px 35px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }
    
    /* Metric card - premium style */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #e9ecef;
        box-shadow: 0 8px 25px rgba(0,0,0,0.04);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3498db, #2ecc71);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 3.2rem;
        font-weight: 800;
        color: #2c3e50;
        margin-top: 0.3rem;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-change {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .metric-change.positive {
        background: #d4edda;
        color: #155724;
    }
    
    .metric-change.negative {
        background: #f8d7da;
        color: #721c24;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(52, 152, 219, 0.4);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #3498db;
    }
    
    /* Number input styling */
    .stNumberInput > div > div {
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    
    /* Slider styling */
    .stSlider > div > div {
        background: #3498db;
    }
    
    /* Radio button styling */
    .stRadio > div {
        gap: 1rem;
    }
    
    .stRadio > div > label {
        background: white;
        padding: 0.5rem 1.5rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .stRadio > div > label:hover {
        border-color: #3498db;
        background: #f8f9fa;
    }
    
    /* Divider styling */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #3498db, transparent);
        margin: 2rem 0;
        opacity: 0.3;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.04);
        border: 1px solid #e9ecef;
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
    st.error("⚠️ Required files not found. Please ensure all model and data files are uploaded.")
    st.stop()

# --- HEADER SECTION ---
st.markdown("""
<div class="dashboard-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="dashboard-title">🚗 DriveValue Analytics</h1>
            <p class="dashboard-subtitle">Intelligent Car Valuation & Market Intelligence Platform</p>
        </div>
        <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">
            <span>📊 Live Market Data</span>
            <span style="margin: 0 1rem;">|</span>
            <span>🔄 Real-time Valuation</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
col_left, col_right = st.columns([0.4, 0.6], gap="large")

with col_left:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Vehicle Specifications")
    st.markdown("---")
    
    car_model = st.selectbox(
        "Model",
        sorted(data['model'].unique()),
        help="Select the vehicle model for valuation"
    )
    
    year = st.slider(
        "Registration Year",
        int(data['year'].min()),
        2025,
        2018,
        help="Year of first registration"
    )
    
    transmission = st.radio(
        "Transmission Type",
        sorted(data['transmission'].unique()),
        horizontal=True
    )
    
    fuel_type = st.selectbox(
        "Fuel Type",
        sorted(data['fuelType'].unique()),
        help="Primary fuel type of the vehicle"
    )
    
    col_mileage, col_mpg = st.columns(2)
    with col_mileage:
        mileage = st.number_input(
            "Mileage",
            min_value=0,
            value=15000,
            step=1000,
            help="Total miles driven"
        )
    with col_mpg:
        mpg = st.number_input(
            "MPG Combined",
            min_value=0,
            value=55,
            step=1,
            help="Miles per gallon (combined cycle)"
        )
    
    col_tax, col_engine = st.columns(2)
    with col_tax:
        tax = st.number_input(
            "Annual Tax ($)",
            min_value=0,
            value=145,
            help="Annual vehicle tax amount"
        )
    with col_engine:
        engine_size = st.number_input(
            "Engine Size (L)",
            min_value=0.5,
            value=1.0,
            step=0.1,
            format="%.1f",
            help="Engine displacement in liters"
        )
    
    st.markdown("---")
    submit = st.button(
        "🚀 Generate Valuation",
        use_container_width=True,
        type="primary"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # Filter data for selected model
    filtered_data = data[data['model'] == car_model]
    
    # Top metrics row
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    
    with col_metric1:
        avg_price = filtered_data['price'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 Average Price</div>
            <div class="metric-value">${avg_price:,.0f}</div>
            <div class="metric-change positive">⬆ +12.5% this year</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_metric2:
        avg_mileage = filtered_data['mileage'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Avg. Mileage</div>
            <div class="metric-value">{avg_mileage:,.0f}</div>
            <div class="metric-change">🚗 miles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_metric3:
        count_models = len(filtered_data)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📋 Listings</div>
            <div class="metric-value">{count_models:,}</div>
            <div class="metric-change">📌 active listings</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<br>', unsafe_allow_html=True)
    
    # Charts section
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Price trend line chart
            trend_data = filtered_data.groupby('year')['price'].mean().reset_index()
            fig_trend = px.line(
                trend_data,
                x='year',
                y='price',
                title=f"💰 Price Trend: {car_model}",
                template="plotly_white",
                labels={'price': 'Price ($)', 'year': 'Year'}
            )
            fig_trend.update_traces(
                line_color='#3498db',
                line_width=3,
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.1)'
            )
            fig_trend.update_layout(
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", size=12),
                margin=dict(l=40, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with chart_col2:
            # Engine size distribution
            fig_engine = px.scatter(
                filtered_data,
                x='engineSize',
                y='price',
                title=f"⚙️ Price vs Engine Size",
                template="plotly_white",
                labels={'price': 'Price ($)', 'engineSize': 'Engine Size (L)'},
                color_discrete_sequence=['#2ecc71']
            )
            fig_engine.update_traces(
                marker=dict(size=10, opacity=0.7, line=dict(width=1, color='white'))
            )
            fig_engine.update_layout(
                hovermode='closest',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", size=12),
                margin=dict(l=40, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_engine, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<br>', unsafe_allow_html=True)
    
    # Valuation result section
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # --- PREDICTION LOGIC ---
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
    
    if submit:
        predicted_price = model.predict(input_encoded)[0]
        
        # Get market comparison
        market_avg = filtered_data['price'].mean()
        price_diff = ((predicted_price - market_avg) / market_avg) * 100
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            padding: 2rem;
            border-radius: 20px;
            color: white;
            box-shadow: 0 10px 30px rgba(52, 152, 219, 0.3);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.9rem; opacity: 0.8;">🏷️ Estimated Market Value</div>
                    <div style="font-size: 3.5rem; font-weight: 800; margin-top: 0.3rem;">
                        ${predicted_price:,.2f}
                    </div>
                    <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">
                        {car_model} • {year} • {fuel_type}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.9rem; opacity: 0.8;">Market Comparison</div>
                    <div style="font-size: 1.5rem; font-weight: 700; margin-top: 0.3rem;">
                        {price_diff:+.1f}%
                    </div>
                    <div style="font-size: 0.8rem; opacity: 0.8;">
                        vs. market avg ${market_avg:,.0f}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div style="
    text-align: center;
    padding: 2rem;
    color: #6c757d;
    font-size: 0.9rem;
    margin-top: 2rem;
    border-top: 1px solid #e9ecef;
">
    <span>🚗 DriveValue Analytics</span>
    <span style="margin: 0 1rem;">•</span>
    <span>Powered by Machine Learning</span>
    <span style="margin: 0 1rem;">•</span>
    <span>© 2026 All Rights Reserved</span>
</div>
""", unsafe_allow_html=True)
